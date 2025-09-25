import os
import cv2
import time
import torch
import argparse
import numpy as np
import os.path as osp

from torch.utils.data import DataLoader
from dust3r.utils.geometry import inv
from dust3r.inference import inference
from dust3r.utils.geometry import geotrf
from dust3r.image_pairs import make_pairs
from dust3r.post_process import estimate_focal_knowing_depth
from dust3r.datasets.demo import Demo
from dust3r.model import Endo3R
    
def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_

def get_args_parser():
    parser = argparse.ArgumentParser('Endo3R demo', add_help=False)
    parser.add_argument('--save_path', type=str, default='./output/demo/', help='Path to experiment folder')
    parser.add_argument('--demo_path', type=str, default='./examples/s00567', help='Path to experiment folder')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/endo3r.pth', help='ckpt path')
    parser.add_argument('--scenegraph_type', type=str, default='complete', help='scenegraph type')
    parser.add_argument('--offline', action='store_true', help='offline reconstruction')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--conf_thresh', type=float, default=1e-3, help='confidence threshold')
    parser.add_argument('--kf_every', type=int, default=10, help='map every kf_every frames')
    parser.add_argument('--resolution', type=int, default=224, help='map every kf_every frames')
    parser.add_argument('--vis', action='store_true', help='visualize')
    parser.add_argument('--vis_cam', action='store_true', help='visualize camera pose')
    parser.add_argument('--save_result', action='store_true', help='save original parameters for NeRF')

    return parser

def get_transform_json(H, W, focal, poses_all, ply_file_path, ori_path=None):
    transform_dict = {
        'w': W,
        'h': H,
        'fl_x': focal.item(),
        'fl_y': focal.item(),
        'cx': W/2,
        'cy': H/2,
        'k1': 0,
        'k2': 0,
        'p1': 0,
        'p2': 0,
        'camera_model': 'OPENCV',
    }
    frames = []

    for i, pose in enumerate(poses_all):
        # CV2 GL format
        pose[:3, 1] *= -1
        pose[:3, 2] *= -1
        frame = {
            'file_path': f"imgs/img_{i:04d}.png" if ori_path is None else ori_path[i],
            'transform_matrix': pose.tolist()
        }
        frames.append(frame)
    
    transform_dict['frames'] = frames
    transform_dict['ply_file_path'] = ply_file_path

    return transform_dict

@torch.no_grad()
def main(args):

    workspace = args.save_path
    os.makedirs(workspace, exist_ok=True)

    ##### Load model
    model = Endo3R(use_feat=False).to(args.device)
    
    model.load_state_dict(torch.load(args.ckpt_path, map_location=args.device)['model'], strict=False)
    model.eval()

    ##### Load dataset
    if args.resolution == 512:
        image_size = (512, 384)
    elif args.resolution == 320:
        image_size = (320, 256)
    else:
        image_size = 224

    dataset = Demo(ROOT=args.demo_path, resolution=image_size, full_video=True, kf_every=args.kf_every)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    batch = dataloader.__iter__().__next__()

    ##### Inference
    for view in batch:
        view['img'] = view['img'].to(args.device, non_blocking=True)
           

    demo_name = args.demo_path.split("/")[-1]

    print(f'Started reconstruction for {demo_name}')
    uncertainty_check = True
    if args.offline:
        imgs_all = []
        for j, view in enumerate(batch):
            img = view['img']
            imgs_all.append(
                dict(
                    img=img,
                    true_shape=torch.tensor(img.shape[2:]).unsqueeze(0),
                    idx=j,
                    instance=str(j)
                )
            )
        pairs = make_pairs(imgs_all, scene_graph=args.scenegraph_type, prefilter=None, symmetrize=True)
        output = inference(pairs, model.dust3r, args.device, batch_size=2, verbose=True)
        preds, preds_all, idx_used = model.offline_reconstruction(batch, output) 

        ordered_batch = [batch[i] for i in idx_used]
    else:
        with torch.no_grad():
            preds, preds_all = model.forward(batch, eval=uncertainty_check, uncertainty_check=uncertainty_check) 

        ordered_batch = batch
        

    save_demo_path = osp.join(workspace, demo_name)
    os.makedirs(save_demo_path, exist_ok=True)

    pts_all = []
    images_all = []
    conf_all = []
    poses_all = []
    depths_all = []

    _, H, W, _ = preds[0]['pts3d'].shape
    pp = torch.tensor((W/2, H/2))
    focal = estimate_focal_knowing_depth(preds[0]['pts3d'].cpu(), pp, focal_mode='weiszfeld')

    intrinsic = np.eye(3)
    intrinsic[0, 0] = focal
    intrinsic[1, 1] = focal
    intrinsic[:2, 2] = pp
    save_depth_dir = os.path.join(save_demo_path, "depth_color")
    save_depth_dir2 = os.path.join(save_demo_path, "depth")
    
    os.makedirs(save_depth_dir, exist_ok=True)
    os.makedirs(save_depth_dir2, exist_ok=True)
    for j, view in enumerate(ordered_batch):
        
        image = view['img'].permute(0, 2, 3, 1).cpu().numpy()[0]
        mask = view['valid_mask'].cpu().numpy()[0]

        pts = preds[j]['pts3d' if j==0 else 'pts3d_in_other_view'].detach().cpu().numpy()[0]
        
        conf = preds[j]['conf'][0].cpu().data.numpy()

        pts_gt = view['pts3d'].cpu().numpy()[0]

        u, v = np.meshgrid(np.arange(W), np.arange(H))
        points_2d = np.stack((u, v), axis=-1)
        dist_coeffs = np.zeros(4).astype(np.float32)
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
            pts.reshape(-1, 3).astype(np.float32), 
            points_2d.reshape(-1, 2).astype(np.float32), 
            intrinsic.astype(np.float32),
            dist_coeffs, iterationsCount=100, reprojectionError=2)
    
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Extrinsic parameters (4x4 matrix)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))
        extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))
        if j != 0:
            pts3d_selfview = geotrf(extrinsic_matrix, pts[None, ...])
        else:
            pts3d_selfview = pts[None, ...]
        self_depth = pts3d_selfview[..., 2]
        depths_all.append(self_depth)
        poses_all.append(inv(extrinsic_matrix))
        images_all.append((image[None, ...] + 1.0)/2.0)
        pts_all.append(pts[None, ...])
        conf_all.append(conf[None, ...])
        if args.save_result:
            img_name = view['instance'][0].split('.')[0]
            color_depth = visualize_depth(self_depth[0])
            cv2.imwrite(
                os.path.join(save_depth_dir, f"{img_name}.png"),
                color_depth,
            )
            np.save(os.path.join(save_depth_dir2, f"{img_name}.npy"), self_depth[0])
        
    
    
    images_all = np.concatenate(images_all, axis=0)
    pts_all = np.concatenate(pts_all, axis=0)
    conf_all = np.concatenate(conf_all, axis=0)
    poses_all = np.stack(poses_all, axis=0)
    depths_all = np.stack(depths_all, axis=0)
    
    if args.save_result:
        
        save_params = dict(
            images_all=images_all,
            pts_all=pts_all,
            conf_all=conf_all,
            poses_all=poses_all,
            intrinsic=intrinsic,
            depths_all=depths_all,
            )
        np.save(os.path.join(save_demo_path, f"result.npy"), save_params)
        print("Finish reconstruction, results saved to ", save_demo_path)
        



if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)