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


def sync(device: str):
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth)  # change nan to 0
    if minmax is None:
        mi = np.min(x)
        ma = np.max(x)
    else:
        mi, ma = minmax

    x = (x - mi) / (ma - mi + 1e-8)
    x = (255 * x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_


def get_args_parser():
    parser = argparse.ArgumentParser('Endo3R demo', add_help=False)
    parser.add_argument('--save_path', type=str, default='./output/demo/', help='Path to experiment folder')
    parser.add_argument('--demo_path', type=str, default='./examples/s00567', help='Path to input sequence folder')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/endo3r.pth', help='ckpt path')
    parser.add_argument('--scenegraph_type', type=str, default='complete', help='scenegraph type')
    parser.add_argument('--offline', action='store_true', help='offline reconstruction')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--conf_thresh', type=float, default=1e-3, help='confidence threshold')

    # NOTE:
    # - In this repo, Demo(ROOT, kf_every=1) loads all frames.
    # - We subsample here before forward so kf_every affects runtime.
    parser.add_argument('--kf_every', type=int, default=1, help='use 1 of every kf_every frames for forward')
    parser.add_argument('--resolution', type=int, default=224, help='resolution preset (224 / 320 / 512)')
    parser.add_argument('--vis', action='store_true', help='visualize')
    parser.add_argument('--vis_cam', action='store_true', help='visualize camera pose')
    parser.add_argument('--save_result', action='store_true', help='save outputs (depth png/npy + result.npy)')

    # ---- ablation knobs ----
    parser.add_argument('--amp', type=int, default=0, choices=[0, 1],
                        help='use torch autocast (fp16/bf16 depending on GPU) during forward')
    parser.add_argument('--uncertainty_check', type=int, default=1, choices=[0, 1],
                        help='enable RAFT-based uncertainty check / dynamic filtering')
    parser.add_argument('--subsample', type=int, default=1,
                        help='additional subsample factor (keep 1 of every subsample frames). 1 means no extra subsample.')

    # ---- profiling helpers ----
    parser.add_argument('--profile_only', action='store_true',
                        help='only run forward and print timing, skip postprocess/save')
    parser.add_argument('--skip_post', action='store_true',
                        help='skip postprocess loop (debug). Prefer --profile_only for benchmarking.')
    parser.add_argument('--max_frames', type=int, default=-1,
                        help='only use first N frames AFTER subsampling; -1 means all')

    return parser


def get_transform_json(H, W, focal, poses_all, ply_file_path, ori_path=None):
    transform_dict = {
        'w': W,
        'h': H,
        'fl_x': focal.item(),
        'fl_y': focal.item(),
        'cx': W / 2,
        'cy': H / 2,
        'k1': 0,
        'k2': 0,
        'p1': 0,
        'p2': 0,
        'camera_model': 'OPENCV',
    }
    frames = []
    for i, pose in enumerate(poses_all):
        pose = pose.copy()
        # CV2 -> GL
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

    # ----- Load model -----
    model = Endo3R(use_feat=False).to(args.device)
    model.sf_model = model.sf_model.to(args.device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=args.device)['model'], strict=False)
    model.eval()

    # ----- Load dataset -----
    if args.resolution == 512:
        image_size = (512, 384)
    elif args.resolution == 320:
        image_size = (320, 256)
    else:
        image_size = 224

    # Load all frames (kf_every=1). We'll subsample before forward.
    dataset = Demo(ROOT=args.demo_path, resolution=image_size, full_video=True, kf_every=1)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    batch = dataloader.__iter__().__next__()  # list of views

    demo_name = args.demo_path.rstrip("/").split("/")[-1]
    print(f"Started reconstruction for {demo_name}")

    # ----- Subsample sequence BEFORE moving to device -----
    # This saves host->device bandwidth and GPU mem.
    n0 = len(batch)
    print("[PROFILE] frames before subsample:", n0)

    # Primary subsample: kf_every
    if args.kf_every is not None and args.kf_every > 1:
        batch = batch[::args.kf_every]

    # Extra subsample knob (for ablation)
    if args.subsample is not None and args.subsample > 1:
        batch = batch[::args.subsample]

    # Optional cap after subsampling
    if args.max_frames is not None and args.max_frames > 0:
        batch = batch[:args.max_frames]

    n1 = len(batch)
    print("[PROFILE] frames after  subsample:", n1)

    if n1 < 2 and not args.offline:
        raise RuntimeError(
            f"Need at least 2 frames for online forward, but got {n1}. "
            f"Reduce kf_every/subsample/max_frames."
        )

    # ----- Move imgs to device -----
    t0 = time.perf_counter()
    for view in batch:
        view['img'] = view['img'].to(args.device, non_blocking=True)
    sync(args.device)
    t1 = time.perf_counter()
    print(f"[PROFILE] to(device)   : {(t1 - t0) * 1000:.2f} ms")

    uncertainty_check = bool(args.uncertainty_check)
    use_amp = bool(args.amp)

    # ----- Forward / Offline -----
    if args.offline:
        imgs_all = []
        for j, view in enumerate(batch):
            img = view['img']
            imgs_all.append(dict(
                img=img,
                true_shape=torch.tensor(img.shape[2:]).unsqueeze(0),
                idx=j,
                instance=str(j)
            ))
        pairs = make_pairs(imgs_all, scene_graph=args.scenegraph_type, prefilter=None, symmetrize=True)

        sync(args.device); ta = time.perf_counter()
        output = inference(pairs, model.dust3r, args.device, batch_size=2, verbose=True)
        sync(args.device); tb = time.perf_counter()

        preds, preds_all, idx_used = model.offline_reconstruction(batch, output)
        sync(args.device); tc = time.perf_counter()

        print(f"[PROFILE] dust3r inference: {(tb - ta) * 1000:.2f} ms")
        print(f"[PROFILE] offline_recon   : {(tc - tb) * 1000:.2f} ms")

        ordered_batch = [batch[i] for i in idx_used]
    else:
        sync(args.device); t2 = time.perf_counter()
        with torch.no_grad():
            # autocast only when CUDA is used; safe guard for cpu runs
            if isinstance(args.device, str) and args.device.startswith("cuda") and torch.cuda.is_available():
                with torch.amp.autocast('cuda', enabled=use_amp):
                    preds, preds_all = model.forward(
                        batch,
                        eval=uncertainty_check,
                        uncertainty_check=uncertainty_check
                    )
            else:
                preds, preds_all = model.forward(
                    batch,
                    eval=uncertainty_check,
                    uncertainty_check=uncertainty_check
                )
        sync(args.device); t3 = time.perf_counter()
        print(f"[PROFILE] model.forward: {(t3 - t2) * 1000:.2f} ms (amp={use_amp}, unc={uncertainty_check})")

        ordered_batch = batch
    print("[DEBUG] preds len:", len(preds))
    print("[DEBUG] preds[0] keys:", preds[0].keys())
    for k, v in preds[0].items():
        if torch.is_tensor(v):
            print(f"  - {k}: tensor {tuple(v.shape)} {v.dtype} {v.device}")
        else:
            print(f"  - {k}: {type(v)}")
    # If profiling only, exit here to avoid postprocess / concatenate errors
    if args.profile_only:
        print("[PROFILE] profile_only=True, skip postprocess/save.")
        return

    # ----- Postprocess / Save -----
    t4 = time.perf_counter()

    save_demo_path = osp.join(workspace, demo_name)
    os.makedirs(save_demo_path, exist_ok=True)

    pts_all = []
    images_all = []
    conf_all = []
    poses_all = []
    depths_all = []

    # NOTE: pred dict might contain pts3d or pts3d_in_other_view depending on index
    if 'pts3d' in preds[0]:
        _, H, W, _ = preds[0]['pts3d'].shape
        pts0 = preds[0]['pts3d']
    else:
        _, H, W, _ = preds[0]['pts3d_in_other_view'].shape
        pts0 = preds[0]['pts3d_in_other_view']

    pp = torch.tensor((W / 2, H / 2))
    focal = estimate_focal_knowing_depth(pts0.cpu(), pp, focal_mode='weiszfeld')

    W_orig, H_orig = 924, 924
    K_orig = np.array([
        [870.7642141449198, 0.0, 386.98873045756136],
        [0.0, 876.787018860571, 491.08567174134566],
        [0.0, 0.0, 1.0]
    ])
    dist_coeffs_orig = np.array([-0.45799176, 0.43385441, 0.00508437, 0.00674396, -0.34430692])

  
    _, H_net, W_net, _ = preds[0]['pts3d'].shape

    scale_x = W_net / W_orig
    scale_y = H_net / H_orig

    intrinsic = K_orig.copy()
    intrinsic[0, 0] *= scale_x  # fx
    intrinsic[1, 1] *= scale_y  # fy
    intrinsic[0, 2] *= scale_x  # cx
    intrinsic[1, 2] *= scale_y  # cy
    print(f"original resolution: {W_orig}x{H_orig}, net resolution: {W_net}x{H_net}")
    print("intrinsics after scaling:\n", intrinsic)
    save_depth_dir = os.path.join(save_demo_path, "depth_color")
    save_depth_dir2 = os.path.join(save_demo_path, "depth")
    os.makedirs(save_depth_dir, exist_ok=True)
    os.makedirs(save_depth_dir2, exist_ok=True)

    if args.skip_post:
        print("[PROFILE] skip_post=True, returning before postprocess.")
        return

    for j, view in enumerate(ordered_batch):
        image = view['img'].permute(0, 2, 3, 1).cpu().numpy()[0]
        _mask = view.get('valid_mask', None)  # keep for future use

        # pick predicted pointmap
        if j == 0 and 'pts3d' in preds[j]:
            pts = preds[j]['pts3d'].detach().cpu().numpy()[0]
        else:
            key = 'pts3d_in_other_view' if 'pts3d_in_other_view' in preds[j] else 'pts3d'
            pts = preds[j][key].detach().cpu().numpy()[0]

        conf = preds[j]['conf'][0].cpu().data.numpy()

        # PnP
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        points_2d = np.stack((u, v), axis=-1)
        dist_coeffs = dist_coeffs_orig.astype(np.float32)

        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
            pts.reshape(-1, 3).astype(np.float32), 
            points_2d.reshape(-1, 2).astype(np.float32), 
            intrinsic.astype(np.float32),
            dist_coeffs,  # <--- 使用我们上面定义的标定畸变系数
            iterationsCount=100, reprojectionError=2)

        if not success:
            # fall back to identity pose if PnP fails (avoid crash)
            rotation_matrix = np.eye(3, dtype=np.float32)
            translation_vector = np.zeros((3,), dtype=np.float32)
        else:
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        extrinsic_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))
        extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1])).astype(np.float32)

        if j != 0:
            pts3d_selfview = geotrf(extrinsic_matrix, pts[None, ...])
        else:
            pts3d_selfview = pts[None, ...]

        self_depth = pts3d_selfview[..., 2]

        depths_all.append(self_depth)
        poses_all.append(inv(extrinsic_matrix))
        images_all.append((image[None, ...] + 1.0) / 2.0)
        pts_all.append(pts[None, ...])
        conf_all.append(conf[None, ...])

        if args.save_result:
            img_name = view['instance'][0].split('.')[0]
            color_depth = visualize_depth(self_depth[0])
            cv2.imwrite(os.path.join(save_depth_dir, f"{img_name}.png"), color_depth)
            np.save(os.path.join(save_depth_dir2, f"{img_name}.npy"), self_depth[0])

    # Guard against empty lists
    if len(images_all) == 0:
        raise RuntimeError("Postprocess produced empty outputs. Check skip_post/profile_only logic.")

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
        np.save(os.path.join(save_demo_path, "result.npy"), save_params)
        print("Finish reconstruction, results saved to", save_demo_path)

    t5 = time.perf_counter()
    print(f"[PROFILE] post+save     : {(t5 - t4) * 1000:.2f} ms")
    print(f"[PROFILE] total (approx): {(t5 - t0) * 1000:.2f} ms")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
