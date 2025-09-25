# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# dpt head implementation for DUST3R
# Downstream heads assume inputs of size B x N x C (where N is the number of tokens) ;
# or if it takes as input the output at every layer, the attribute return_all_layers should be set to True
# the forward function also takes as input a dictionnary img_info with key "height" and "width"
# for PixelwiseTask, the output will be of dimension B x num_channels x H x W
# --------------------------------------------------------
from einops import rearrange
from typing import List
import torch
import torch.nn as nn
from dust3r.heads.postprocess import postprocess
import dust3r.utils.path_to_croco  # noqa: F401
from models.dpt_block import DPTOutputAdapter  # noqa
# from dust3r.utils.camera import pose_encoding_to_camera, PoseDecoder
import torch.nn.functional as F

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    quaternions = F.normalize(quaternions, p=2, dim=-1)
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def postprocess_pose(out, mode=("exp", -float("inf"), float("inf")), inverse=False):
    """
    extract pose from prediction head output
    """
    mode, vmin, vmax = mode

    no_bounds = (vmin == -float("inf")) and (vmax == float("inf"))
    assert no_bounds
    trans = out[..., 0:3]
    quats = out[..., 3:7]

    if mode == "linear":
        if no_bounds:
            return trans  # [-inf, +inf]
        return trans.clip(min=vmin, max=vmax)

    d = trans.norm(dim=-1, keepdim=True)

    if mode == "square":
        if inverse:
            scale = d / d.square().clip(min=1e-8)
        else:
            scale = d.square() / d.clip(min=1e-8)

    if mode == "exp":
        if inverse:
            scale = d / torch.expm1(d).clip(min=1e-8)
        else:
            scale = torch.expm1(d) / d.clip(min=1e-8)

    trans = trans * scale
    quats = standardize_quaternion(quats)

    return torch.cat([trans, quats], dim=-1)

class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features=1, num_frames_to_predict_for=1, stride=1):
        super().__init__()
        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.conv_squeeze = nn.Conv2d(768, 256, 1)

        self.convs_pose = []
        self.convs_pose.append(
            nn.Conv2d(num_input_features * 256, 256, 3, stride, 1))
        self.convs_pose.append(nn.Conv2d(256, 256, 3, stride, 1))
        self.convs_pose.append(
            nn.Conv2d(256, 7 * num_frames_to_predict_for, 1))
       
        self.relu = nn.ReLU()

        self.convs_pose = nn.ModuleList(list(self.convs_pose))

    def forward(self, input_features, image_size):
        # image_size = else image_size
        H, W = image_size
        # Number of patches in height and width
        N_H = H // 16
        N_W = W // 16
        # last_features = [f[-1] for f in input_features]

        # cat_features = [self.relu(self.conv_squeeze(f)) for f in last_features]
        # cat_features = torch.cat(cat_features, 1)
        last_dec_feat = rearrange(input_features[-1], 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W)
        out = self.relu(self.conv_squeeze(last_dec_feat))
        for i in range(3):
            out = self.convs_pose[i](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        pose = out.view(-1, 7)
        # pose[:, :3] = 0.1 * pose[:, :3]
        # pose[:, 3:] = 0.01 * pose[:, 3:]
        
        return postprocess_pose(pose)

class DPTOutputAdapter_fix(DPTOutputAdapter):
    """
    Adapt croco's DPTOutputAdapter implementation for dust3r:
    remove duplicated weigths, and fix forward for dust3r
    """

    def init(self, dim_tokens_enc=768):
        super().init(dim_tokens_enc)
        # these are duplicated weights
        del self.act_1_postprocess
        del self.act_2_postprocess
        del self.act_3_postprocess
        del self.act_4_postprocess

    def forward(self, encoder_tokens: List[torch.Tensor], image_size=None):
        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        # H, W = input_info['image_size']
        image_size = self.image_size if image_size is None else image_size
        H, W = image_size
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        # Hook decoder onto 4 layers from specified ViT layers
        layers = [encoder_tokens[hook] for hook in self.hooks]

        # Extract only task-relevant tokens and ignore global tokens.
        layers = [self.adapt_tokens(l) for l in layers]

        # Reshape tokens to spatial representation
        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in layers]

        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        # Project layers to chosen feature dim
        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

        # Fuse layers using refinement stages
        path_4 = self.scratch.refinenet4(layers[3])[:, :, :layers[2].shape[2], :layers[2].shape[3]]
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])

        # Output head
        out = self.head(path_1)

        return out

def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat

def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    # if rotation_mode == 'euler':
    rot_mat = euler2mat(rot)  # [B, 3, 3]
    # elif rotation_mode == 'quat':
    #     rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat

class PixelwiseTaskWithDPT(nn.Module):
    """ DPT module for dust3r, can return 3D points + confidence for all pixels"""

    def __init__(self, *, n_cls_token=0, hooks_idx=None, dim_tokens=None,
                 output_width_ratio=1, num_channels=1, postprocess=None, depth_mode=None, conf_mode=None, has_pose=False, net=None, **kwargs):
        super(PixelwiseTaskWithDPT, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.postprocess = postprocess
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        assert n_cls_token == 0, "Not implemented"
        dpt_args = dict(output_width_ratio=output_width_ratio,
                        num_channels=num_channels,
                        **kwargs)
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = DPTOutputAdapter_fix(**dpt_args)
        self.pose_head = None
        if has_pose:
            self.pose_head = PoseDecoder(net.dec_embed_dim)
        dpt_init_args = {} if dim_tokens is None else {'dim_tokens_enc': dim_tokens}
        self.dpt.init(**dpt_init_args)

    def forward(self, x, img_info):
        out = self.dpt(x, image_size=(img_info[0], img_info[1]))
        if self.pose_head is not None:
            pred_pose = self.pose_head(x, image_size=(img_info[0], img_info[1]))
            
        if self.postprocess:
            out = self.postprocess(out, self.depth_mode, self.conf_mode)
        if self.pose_head is not None:
            out['pose'] = postprocess_pose(pred_pose)
        return out


def create_dpt_head(net, has_conf=False, has_pose=False):
    """
    return PixelwiseTaskWithDPT for given net params
    """
    assert net.dec_depth > 9
    l2 = net.dec_depth
    feature_dim = 256
    last_dim = feature_dim//2
    out_nchan = 3
    ed = net.enc_embed_dim
    dd = net.dec_embed_dim
    return PixelwiseTaskWithDPT(num_channels=out_nchan + has_conf,
                                feature_dim=feature_dim,
                                last_dim=last_dim,
                                hooks_idx=[0, l2*2//4, l2*3//4, l2],
                                dim_tokens=[ed, dd, dd, dd],
                                postprocess=postprocess,
                                depth_mode=net.depth_mode,
                                conf_mode=net.conf_mode,
                                has_pose = has_pose,
                                net = net,
                                head_type='regression')
