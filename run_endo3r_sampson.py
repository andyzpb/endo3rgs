import argparse
import os
import torch
import numpy as np
import json
import cv2
import trimesh
import glob
from torch.utils.data import DataLoader

from dust3r.model import Endo3R
from dust3r.utils.geometry import inv
from dust3r.post_process import estimate_focal_knowing_depth

# -----------------------------------------------------------------------------
# Data Loader: supports frame skipping sampling (kf_every)
# -----------------------------------------------------------------------------
class OptimizedDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, target_size=512, kf_every=1):
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG'}
        all_files = sorted([f for f in glob.glob(os.path.join(folder_path, '*')) 
                            if os.path.splitext(f)[1] in exts])
        # [Optimization] Frame skipping to reduce cumulative drift
        self.files = all_files[::kf_every]
        self.target_size = target_size
        print(f">> [Loader] Total frames: {len(all_files)}, Using: {len(self.files)} (Step: {kf_every})")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        img_bgr = cv2.imread(file_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
        origin_img = torch.from_numpy(img_resized).float()
        img_tensor = (origin_img.permute(2, 0, 1) / 127.5) - 1.0
        return {
            'img': img_tensor,
            'origin_img': origin_img,
            'true_shape': torch.tensor([self.target_size, self.target_size]),
            'instance': [file_path]
        }

@torch.no_grad()
def main(args):
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    # A. Load model
    print(f"[Endo3R] Loading model from {args.ckpt_path}...")
    model = Endo3R(use_feat=False).to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # B. Prepare data (use frame skipping)
    dataset = OptimizedDataset(args.image_dir, kf_every=args.kf_every)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    batch = []
    for item in loader:
        batch.append({
            'img': item['img'].to(device),
            'origin_img': item['origin_img'].to(device),
            'true_shape': item['true_shape'].to(device),
            'instance': item['instance']
        })

    # C. Inference
    print(f"[Endo3R] Running Native Inference (KF every {args.kf_every})...")
    preds, _ = model.forward(batch, eval=True, uncertainty_check=True)

    # D. Robust PnP and filtering
    print(f"[Process] Extracting points with threshold {args.conf_thresh}...")
    _, H, W, _ = preds[0]['pts3d'].shape
    pp = torch.tensor((W/2, H/2))
    focal_est = estimate_focal_knowing_depth(preds[0]['pts3d'].cpu(), pp, focal_mode='weiszfeld')
    
    intrinsic = np.eye(3)
    intrinsic[0, 0] = intrinsic[1, 1] = focal_est
    intrinsic[:2, 2] = pp.numpy()

    cameras_meta = {}
    all_pts, all_cols = [], []
    
    for j, view in enumerate(batch):
        pts_key = 'pts3d' if j == 0 else 'pts3d_in_other_view'
        pts = preds[j][pts_key].detach().cpu().numpy()[0]
        conf = preds[j]['conf'].detach().cpu().numpy()[0]
        
        # [Optimization] PnP solving uses higher iterations and stricter RANSAC
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        p2d = np.stack((u, v), axis=-1).reshape(-1, 2).astype(np.float32)
        p3d = pts.reshape(-1, 3).astype(np.float32)
        
        # Filter invalid points for PnP
        pnp_mask = (conf.flatten() > 1.0) & (np.isfinite(p3d).all(axis=1))
        if pnp_mask.sum() > 50:
            success, rvec, tvec, _ = cv2.solvePnPRansac(p3d[pnp_mask], p2d[pnp_mask], 
                                                       intrinsic.astype(np.float32), 
                                                       np.zeros(4).astype(np.float32),
                                                       iterationsCount=200, reprojectionError=1.5)
            c2w = inv(np.vstack((np.hstack((cv2.Rodrigues(rvec)[0], tvec)), [0, 0, 0, 1]))) if success else np.eye(4)
        else:
            c2w = np.eye(4)

        # [Core optimization] Increase confidence threshold to reduce artifacts
        valid_mask = (conf > args.conf_thresh)
        curr_pts = pts[valid_mask]
        curr_img = ((view['img'].permute(0,2,3,1)[0].cpu().numpy() + 1.0)/2.0)[valid_mask]
        
        all_pts.append(curr_pts)
        all_cols.append(curr_img)

        img_name = os.path.basename(view['instance'][0][0])
        cameras_meta[j] = {"img_name": img_name, "c2w": c2w.tolist()}

    # E. Save
    full_pts = np.concatenate(all_pts)
    full_cols = np.concatenate(all_cols)
    pcd = trimesh.points.PointCloud(full_pts, colors=(full_cols*255).astype(np.uint8))
    pcd.export(os.path.join(args.output_dir, 'endo_native_optimized.ply'))
    
    with open(os.path.join(args.output_dir, 'cameras.json'), 'w') as f:
        json.dump({"intrinsic": intrinsic.tolist(), "frames": cameras_meta}, f, indent=4)
    print(f">> [Success] Optimized results saved to {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='data/endo_native_best')
    parser.add_argument('--ckpt_path', type=str, default='checkpoints/endo3r.pth')
    # [New parameters] Recommended values explanation
    parser.add_argument('--kf_every', type=int, default=1, help="Frame skip step, recommended 5-10 to reduce ghosting")
    parser.add_argument('--conf_thresh', type=float, default=2, help="Confidence threshold, recommended 1.5-2.0 to filter noise")
    parser.add_argument('--device', type=str, default='cuda')
    main(parser.parse_args())