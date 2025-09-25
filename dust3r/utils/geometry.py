# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# geometry utilitary functions
# --------------------------------------------------------
import torch
import numpy as np
from scipy.spatial import cKDTree as KDTree

from dust3r.utils.misc import invalid_to_zeros, invalid_to_nans
from dust3r.utils.device import to_numpy
from kornia.geometry.epipolar import fundamental_from_essential, essential_from_Rt, sampson_epipolar_distance
import torch.nn.functional as F

def detect_dynamic_regions_3d_distance(pointmap1, pointmap2, flow_pr, threshold=0.1, occlusion_uncertainty=0.0):
    """
    Detect dynamic regions by computing 3D distance between corresponding points.
    Sets distance to zero for regions without correspondence (e.g., occlusions).
    
    Args:
        pointmap1 (torch.Tensor): 3D point map for view1 (B, H, W, 3), in world coordinates.
        pointmap2 (torch.Tensor): 3D point map for view2 (B, H, W, 3), in world coordinates.
        flow_pr (torch.Tensor): Optical flow from view1 to view2 (B, H, W, 2).
        threshold (float): Distance threshold to classify dynamic regions (in world units).
    
    Returns:
        distance (torch.Tensor): 3D distance between corresponding points (H, W), with occluded regions set to 0.
    """
    B, H, W, _ = pointmap1.shape
    device = pointmap1.device

    # Create pixel grid for view1
    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    pixels = torch.stack([x, y], dim=-1).float()  # (H, W, 2)
    pixels = pixels.reshape(1, H, W, 2).expand(B, H, W, 2)  # (B, H, W, 2)

    # Compute corresponding pixel coordinates in view2 using flow
    pixels2 = pixels + flow_pr  # (B, H, W, 2)
    flow_norm = torch.norm(flow_pr, dim=-1, keepdim=False)
    flow_mask = flow_norm > torch.mean(flow_norm)
    # Create validity mask for correspondences within image boundaries
    valid_mask = (pixels2[:, :, :, 0] >= 0) & (pixels2[:, :, :, 0] <= W-1) & \
                 (pixels2[:, :, :, 1] >= 0) & (pixels2[:, :, :, 1] <= H-1)  # (B, H, W)

    # Sample pointmap2 at corresponding coordinates (bilinear interpolation)
    grid = pixels2 / torch.tensor([W-1, H-1], device=device).view(1, 1, 1, 2) * 2 - 1  # Normalize to [-1, 1]
    pointmap2_sampled = F.grid_sample(pointmap2.permute(0, 3, 1, 2), grid, align_corners=True, mode='bilinear')
    pointmap2_sampled = pointmap2_sampled.permute(0, 2, 3, 1)  # (B, H, W, 3)

    # Compute Euclidean distance between corresponding 3D points
    distance = torch.norm(pointmap1 - pointmap2_sampled, dim=-1)  # (B, H, W)
    valid_distance = distance * valid_mask.float()  # Zero out invalid distances
    max_distance = torch.where(valid_mask, distance, torch.zeros_like(distance)).amax(dim=(1, 2), keepdim=True) + 1e-6
    min_distance = torch.where(valid_mask, distance, torch.full_like(distance, float('inf'))).amin(dim=(1, 2), keepdim=True)
    normalized_distance = (distance - min_distance) / (max_distance - min_distance + 1e-6)  # (B, H, W)
    normalized_distance = normalized_distance * valid_mask.float()  # Apply valid mask

    # Create uncertainty map: normalized distances for valid regions, high value for occluded regions
    uncertainty = (normalized_distance + occlusion_uncertainty * (~valid_mask).float()) * flow_mask


    return uncertainty  # (H, W)

def compute_rigid_flow(world_points, w2c2, intrinsic, H, W):
    """
    Compute expected rigid flow from depth and camera poses.
    
    Args:
        depth1 (torch.Tensor): Depth map for view1 (B, H, W).
        c2w1, c2w2 (torch.Tensor): Camera-to-world matrices (B, 4, 4).
        intrinsic (torch.Tensor): Intrinsic matrix (B, 3, 3).
        H, W (int): Image height and width.
    
    Returns:
        rigid_flow (torch.Tensor): Expected flow (B, H, W, 2).
    """
    
    y, x = torch.meshgrid(torch.arange(H, device=world_points.device), 
                          torch.arange(W, device=world_points.device), indexing='ij')
    pixels = torch.stack([x, y], dim=-1).float().reshape(H, W, 2)
    
    # # Backproject pixels to 3D points
    # depth1 = depth1.unsqueeze(-1)  # (B, H, W, 1)
    # cam_points = torch.matmul(torch.inverse(intrinsic), pixels.reshape(B, H*W, 2).permute(0, 2, 1)).permute(0, 2, 1)  # (B, H*W, 3)
    # cam_points = cam_points * depth1.reshape(B, H*W, 1)  # (B, H*W, 3)
    
    # # Transform to world coordinates (view1)
    # world_points = torch.matmul(c2w1[:, :3, :3], cam_points) + c2w1[:, :3, 3].unsqueeze(1)  # (B, H*W, 3)
    world_points = world_points.reshape(H*W, 3) 
    # # Transform to view2 camera coordinates
    # w2c2 = torch.inverse(c2w2)  # World-to-camera for view2
    cam2_points = torch.matmul(w2c2[:3, :3], world_points - w2c2[:3, 3].unsqueeze(0))  # (B, H*W, 3)
    
    # Project to view2 image plane
    proj_points = torch.matmul(intrinsic, cam2_points).permute(1, 0).reshape(3, H, W)  # (B, 3, H, W)
    rigid_flow = proj_points[:2] / (proj_points[2:3] + 1e-6) - pixels.permute(2, 0, 1)[:2]
    
    return rigid_flow  # (B, 2, H, W)

def detect_dynamic_regions_depth(flow_pr, depth1, w2c2, intrinsic, H, W, threshold=1.0):
    """
    Detect dynamic regions by comparing estimated flow with rigid flow from depth.
    
    Returns:
        dynamic_mask (torch.Tensor): Binary mask for dynamic regions (B, H, W).
        flow_diff (torch.Tensor): Difference between estimated and rigid flow (B, H, W).
    """
    rigid_flow = compute_rigid_flow(depth1, w2c2, intrinsic, H, W)  # (B, 2, H, W)
    flow_pr_reshaped = flow_pr  # (B, 2, H, W)
    flow_diff = torch.norm(flow_pr_reshaped - rigid_flow, dim=0)  # (B, H, W)
    dynamic_mask = flow_diff > threshold  # Adjust threshold based on flow scale
    return dynamic_mask, flow_diff

def adaptive_thresholding(tensor, factor=3.0):
    """
    Apply adaptive thresholding to segment the object from the background.

    Args:
        tensor (torch.Tensor): The input tensor to threshold.
        factor (float): The factor to determine the threshold based on mean and std deviation.

    Returns:
        torch.Tensor: Binary mask tensor where object pixels are 1 and background pixels are 0.
    """
    # Calculate mean and standard deviation of the tensor
    mean_value = tensor.mean().item()
    std_value = tensor.std().item()

    # Set threshold as mean + factor * std
    threshold = mean_value + factor * std_value
    # threshold = tensor.median().item()
    # print(mean_value, std_value, threshold)
    # Create a binary mask based on the threshold
    mask = tensor <= threshold
    return mask

def get_matches(flow_fw, h, w):
    flow = flow_fw[0]
    x = torch.arange(w, dtype=torch.long)
    y = torch.arange(h, dtype=torch.long)
    yy, xx = torch.meshgrid(y, x)
    pts_source = torch.stack([xx, yy], dim=-1).to(flow.device)
    xx_source, yy_source = torch.split(pts_source, 1, dim=-1)
    xx_target, yy_target = torch.split(pts_source + flow, 1, dim=-1)
    xx_source, yy_source, xx_target, yy_target = xx_source.squeeze(), yy_source.squeeze(), xx_target.squeeze(), yy_target.squeeze()

    valid_regeion = torch.ones((h,w)).to(flow.device)
    valid_regeion = valid_regeion * (xx_target > 0) * (xx_target < w) * (yy_target > 0) * (yy_target < h) #* (depth > 0)
    valid_regeion = valid_regeion.bool()
    pts_source, pts_target = torch.stack([xx_source, yy_source], axis=0).float(), torch.stack([xx_target, yy_target], axis=0).float()
    matches = torch.cat([pts_source, pts_target], dim=0).unsqueeze(0).view([1, 4, -1])
    valid_regeion = valid_regeion.view(1, 1, -1)
    return matches, valid_regeion

def w2c_to_c2w(Rt_w2c):
    """
    Convert a world-to-camera (w2c) pose to camera-to-world (c2w) pose.
    
    Args:
        Rt_w2c (torch.Tensor): 4x4 transformation matrix [R|t] or 3x4 matrix [R|t] 
                              where R is the rotation (3x3) and t is the translation (3x1).
    
    Returns:
        torch.Tensor: 4x4 camera-to-world transformation matrix [R_c2w|t_c2w].
    """
    # Extract rotation and translation
    R_w2c = Rt_w2c[:3, :3]
    t_w2c = Rt_w2c[:3, 3]
    
    # Compute c2w rotation: R_c2w = R_w2c^T
    R_c2w = R_w2c.transpose(-1, -2)
    
    # Compute c2w translation: t_c2w = -R_w2c^T * t_w2c
    t_c2w = -torch.matmul(R_c2w, t_w2c.unsqueeze(-1)).squeeze(-1)
    
    # Construct 4x4 c2w transformation matrix
    Rt_c2w = torch.eye(4, dtype=Rt_w2c.dtype, device=Rt_w2c.device)
    Rt_c2w[:3, :3] = R_c2w
    Rt_c2w[:3, 3] = t_c2w
    
    return Rt_c2w

def get_fundamental_matrix(Rt_1, Rt_2, intrinsic):
    # Rt_1 = self.pose_param_net.forward(timestep_1).detach()
    R_1, t_1 = Rt_1[:3, :3], Rt_1[:3, 3]
    # Rt_2 = self.pose_param_net.forward(timestep_2).detach()
    R_2, t_2 = Rt_2[:3, :3], Rt_2[:3, 3]
    intrinsic = intrinsic.unsqueeze(0)
    essential_mat = essential_from_Rt(R_1.unsqueeze(0), t_1.reshape(1, 3, 1), \
        R_2.unsqueeze(0), t_2.reshape(1, 3, 1))
    F = fundamental_from_essential(essential_mat, intrinsic, intrinsic)
    return F

def compute_epipolar_mask(flow_fw, Rt_1, Rt_2, h, w, intrinsic):
    # fmat: [b, 3, 3] match: [b, 4, h*w] mask: [b,1,h*w]
    match, mask = get_matches(flow_fw, h, w)
    fmat = get_fundamental_matrix(Rt_1, Rt_2, intrinsic)

    points1 = match[:,:2,:].permute(0, 2, 1)
    points2 = match[:,2:,:].permute(0, 2, 1)
    
    samp_dis = sampson_epipolar_distance(points1, points2, fmat)
    sampson_dist = (samp_dis.view([h, w])).float()
    # loss = samp_dis.mean()
    rigid_mask = sampson_dist < adaptive_thresholding(sampson_dist)
    return rigid_mask, sampson_dist

def xy_grid(W, H, device=None, origin=(0, 0), unsqueeze=None, cat_dim=-1, homogeneous=False, **arange_kw):
    """ Output a (H,W,2) array of int32 
        with output[j,i,0] = i + origin[0]
             output[j,i,1] = j + origin[1]
    """
    if device is None:
        # numpy
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:
        # torch
        arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a: torch.ones(*a, device=device)

    tw, th = [arange(o, o + s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing='xy')
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid


def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')


def depthmap_to_pts3d(depth, pseudo_focal, pp=None, **_):
    """
    Args:
        - depthmap (BxHxW array):
        - pseudo_focal: [B,H,W] ; [B,2,H,W] or [B,1,H,W]
    Returns:
        pointmap of absolute coordinates (BxHxWx3 array)
    """

    if len(depth.shape) == 4:
        B, H, W, n = depth.shape
    else:
        B, H, W = depth.shape
        n = None

    if len(pseudo_focal.shape) == 3:  # [B,H,W]
        pseudo_focalx = pseudo_focaly = pseudo_focal
    elif len(pseudo_focal.shape) == 4:  # [B,2,H,W] or [B,1,H,W]
        pseudo_focalx = pseudo_focal[:, 0]
        if pseudo_focal.shape[1] == 2:
            pseudo_focaly = pseudo_focal[:, 1]
        else:
            pseudo_focaly = pseudo_focalx
    else:
        raise NotImplementedError("Error, unknown input focal shape format.")

    assert pseudo_focalx.shape == depth.shape[:3]
    assert pseudo_focaly.shape == depth.shape[:3]
    grid_x, grid_y = xy_grid(W, H, cat_dim=0, device=depth.device)[:, None]

    # set principal point
    if pp is None:
        grid_x = grid_x - (W - 1) / 2
        grid_y = grid_y - (H - 1) / 2
    else:
        grid_x = grid_x.expand(B, -1, -1) - pp[:, 0, None, None]
        grid_y = grid_y.expand(B, -1, -1) - pp[:, 1, None, None]

    if n is None:
        pts3d = torch.empty((B, H, W, 3), device=depth.device)
        pts3d[..., 0] = depth * grid_x / pseudo_focalx
        pts3d[..., 1] = depth * grid_y / pseudo_focaly
        pts3d[..., 2] = depth
    else:
        pts3d = torch.empty((B, H, W, 3, n), device=depth.device)
        pts3d[..., 0, :] = depth * (grid_x / pseudo_focalx)[..., None]
        pts3d[..., 1, :] = depth * (grid_y / pseudo_focaly)[..., None]
        pts3d[..., 2, :] = depth
    return pts3d



def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    # Mask for valid coordinates
    valid_mask = (depthmap > 0.0)
    # Invalid any depth > 80m
    valid_mask = valid_mask
    return X_cam, valid_mask


def depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose, z_far=0, **kw):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
        - camera_pose: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels."""
    X_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)
    if z_far > 0:
        valid_mask = valid_mask & (depthmap < z_far)

    X_world = X_cam # default
    if camera_pose is not None:
        # R_cam2world = np.float32(camera_params["R_cam2world"])
        # t_cam2world = np.float32(camera_params["t_cam2world"]).squeeze()
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]

        # Express in absolute coordinates (invalid depth values)
        X_world = np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]

    return X_world, valid_mask


def colmap_to_opencv_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


def opencv_to_colmap_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K


def normalize_pointcloud(pts1, pts2, norm_mode='avg_dis', valid1=None, valid2=None, ret_factor=False):
    """ renorm pointmaps pts1, pts2 with norm_mode
    """
    assert pts1.ndim >= 3 and pts1.shape[-1] == 3
    assert pts2 is None or (pts2.ndim >= 3 and pts2.shape[-1] == 3)
    norm_mode, dis_mode = norm_mode.split('_')

    if norm_mode == 'avg':
        # gather all points together (joint normalization)
        nan_pts1, nnz1 = invalid_to_zeros(pts1, valid1, ndim=3)
        nan_pts2, nnz2 = invalid_to_zeros(pts2, valid2, ndim=3) if pts2 is not None else (None, 0)
        all_pts = torch.cat((nan_pts1, nan_pts2), dim=1) if pts2 is not None else nan_pts1

        # compute distance to origin
        all_dis = all_pts.norm(dim=-1)
        if dis_mode == 'dis':
            pass  # do nothing
        elif dis_mode == 'log1p':
            all_dis = torch.log1p(all_dis)
        elif dis_mode == 'warp-log1p':
            # actually warp input points before normalizing them
            log_dis = torch.log1p(all_dis)
            warp_factor = log_dis / all_dis.clip(min=1e-8)
            H1, W1 = pts1.shape[1:-1]
            pts1 = pts1 * warp_factor[:, :W1 * H1].view(-1, H1, W1, 1)
            if pts2 is not None:
                H2, W2 = pts2.shape[1:-1]
                pts2 = pts2 * warp_factor[:, W1 * H1:].view(-1, H2, W2, 1)
            all_dis = log_dis  # this is their true distance afterwards
        else:
            raise ValueError(f'bad {dis_mode=}')

        norm_factor = all_dis.sum(dim=1) / (nnz1 + nnz2 + 1e-8)
    else:
        # gather all points together (joint normalization)
        nan_pts1 = invalid_to_nans(pts1, valid1, ndim=3)
        nan_pts2 = invalid_to_nans(pts2, valid2, ndim=3) if pts2 is not None else None
        all_pts = torch.cat((nan_pts1, nan_pts2), dim=1) if pts2 is not None else nan_pts1

        # compute distance to origin
        all_dis = all_pts.norm(dim=-1)

        if norm_mode == 'avg':
            norm_factor = all_dis.nanmean(dim=1)
        elif norm_mode == 'median':
            norm_factor = all_dis.nanmedian(dim=1).values.detach()
        elif norm_mode == 'sqrt':
            norm_factor = all_dis.sqrt().nanmean(dim=1)**2
        else:
            raise ValueError(f'bad {norm_mode=}')

    norm_factor = norm_factor.clip(min=1e-8)
    while norm_factor.ndim < pts1.ndim:
        norm_factor.unsqueeze_(-1)

    res = pts1 / norm_factor
    if pts2 is not None:
        res = (res, pts2 / norm_factor)
    if ret_factor:
        res = res + (norm_factor,)
    return res


@torch.no_grad()
def get_joint_pointcloud_depth(z1, z2, valid_mask1, valid_mask2=None, quantile=0.5):
    # set invalid points to NaN
    _z1 = invalid_to_nans(z1, valid_mask1).reshape(len(z1), -1)
    _z2 = invalid_to_nans(z2, valid_mask2).reshape(len(z2), -1) if z2 is not None else None
    _z = torch.cat((_z1, _z2), dim=-1) if z2 is not None else _z1

    # compute median depth overall (ignoring nans)
    if quantile == 0.5:
        shift_z = torch.nanmedian(_z, dim=-1).values
    else:
        shift_z = torch.nanquantile(_z, quantile, dim=-1)
    return shift_z  # (B,)


@torch.no_grad()
def get_joint_pointcloud_center_scale(pts1, pts2, valid_mask1=None, valid_mask2=None, z_only=False, center=True):
    # set invalid points to NaN
    _pts1 = invalid_to_nans(pts1, valid_mask1).reshape(len(pts1), -1, 3)
    _pts2 = invalid_to_nans(pts2, valid_mask2).reshape(len(pts2), -1, 3) if pts2 is not None else None
    _pts = torch.cat((_pts1, _pts2), dim=1) if pts2 is not None else _pts1

    # compute median center
    _center = torch.nanmedian(_pts, dim=1, keepdim=True).values  # (B,1,3)
    if z_only:
        _center[..., :2] = 0  # do not center X and Y

    # compute median norm
    _norm = ((_pts - _center) if center else _pts).norm(dim=-1)
    scale = torch.nanmedian(_norm, dim=1).values
    return _center[:, None, :, :], scale[:, None, None, None]


def find_reciprocal_matches(P1, P2):
    """
    returns 3 values:
    1 - reciprocal_in_P2: a boolean array of size P2.shape[0], a "True" value indicates a match
    2 - nn2_in_P1: a int array of size P2.shape[0], it contains the indexes of the closest points in P1
    3 - reciprocal_in_P2.sum(): the number of matches
    """
    tree1 = KDTree(P1)
    tree2 = KDTree(P2)

    _, nn1_in_P2 = tree2.query(P1, workers=8)
    _, nn2_in_P1 = tree1.query(P2, workers=8)

    reciprocal_in_P1 = (nn2_in_P1[nn1_in_P2] == np.arange(len(nn1_in_P2)))
    reciprocal_in_P2 = (nn1_in_P2[nn2_in_P1] == np.arange(len(nn2_in_P1)))
    assert reciprocal_in_P1.sum() == reciprocal_in_P2.sum()
    return reciprocal_in_P2, nn2_in_P1, reciprocal_in_P2.sum()


def get_med_dist_between_poses(poses):
    from scipy.spatial.distance import pdist
    return np.median(pdist([to_numpy(p[:3, 3]) for p in poses]))
