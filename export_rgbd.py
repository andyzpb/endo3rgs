import torch
import os
import argparse
import numpy as np
import cv2
import torch.nn as nn
from tqdm import tqdm
from simple_gs.gaussian_model import GaussianModel
from simple_gs.data_loader import load_scene_data
from simple_gs.renderer import render

def render_rgb_and_depth(viewpoint_cam, gaussians, bg_color):
    """
    Renders RGB, Alpha, and Depth maps from the Gaussian model.
    Performs alpha-weighted depth correction.
    """
    # 1. Render standard RGB
    render_pkg = render(viewpoint_cam, gaussians, bg_color)
    rgb_image = render_pkg["render"]

    # 2. Render Alpha (Opacity)
    # We temporarily set all Gaussian colors to pure white (1.0) to render an alpha mask
    with torch.no_grad():
        # Backup original attributes
        orig_dc = gaussians._features_dc.clone()
        orig_rest = gaussians._features_rest.clone()
        
        # Set color to white (SH coeff for 1.0 is ~1.772)
        SH_ONE = (1.0 - 0.5) / 0.28209479177387814
        gaussians._features_dc = nn.Parameter(torch.ones_like(gaussians._features_dc) * SH_ONE)
        gaussians._features_rest = nn.Parameter(torch.zeros_like(gaussians._features_rest))
        
        # Render alpha
        alpha_pkg = render(viewpoint_cam, gaussians, bg_color)
        alpha_map = alpha_pkg["render"][0:1, :, :] # [1, H, W]

        # 3. Render Depth
        # Project Gaussian centers to camera space to get Z-depth
        w2c = viewpoint_cam.world_view_transform
        pts = torch.cat([gaussians.get_xyz, torch.ones_like(gaussians.get_xyz[:, :1])], dim=1)
        pts_cam = torch.matmul(pts, w2c)
        depth_values = pts_cam[:, 2:3] # [N, 1]

        # Normalize depth for rendering (Linear projection)
        # Adjust these bounds based on your surgical scene scale (in mm)
        min_depth = 0.1
        max_depth = 300.0 
        norm_depth = (depth_values - min_depth) / (max_depth - min_depth)
        norm_depth = torch.clamp(norm_depth, 0.0, 1.0)

        # Encode depth into SH coefficients (treat depth as grayscale color)
        C0 = 0.28209479177387814
        depth_sh = (norm_depth - 0.5) / C0
        gaussians._features_dc = nn.Parameter(depth_sh.repeat(1, 3).unsqueeze(1))
        
        # Render raw depth map
        depth_pkg = render(viewpoint_cam, gaussians, bg_color)
        raw_depth_map = depth_pkg["render"][0:1, :, :]

        # 4. Correct Depth using Alpha
        # Un-premultiply alpha: Real_Depth = Rendered_Depth / Alpha
        valid_mask = alpha_map > 0.5 # Filter noisy edges
        corrected_norm_depth = torch.zeros_like(raw_depth_map)
        corrected_norm_depth[valid_mask] = raw_depth_map[valid_mask] / alpha_map[valid_mask]
        
        # Denormalize to real metric units (mm)
        final_depth_map = corrected_norm_depth * (max_depth - min_depth) + min_depth
        final_depth_map[~valid_mask] = 0.0

        # Restore original model state
        gaussians._features_dc = nn.Parameter(orig_dc)
        gaussians._features_rest = nn.Parameter(orig_rest)

    return rgb_image, final_depth_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to preprocessed data (endo_gs_processed)")
    parser.add_argument("--ply_path", type=str, default="output/final_refined_model.ply", help="Path to trained PLY")
    parser.add_argument("--output_path", type=str, default="output/rgbd_data", help="Where to save images")
    args = parser.parse_args()
    
    # Setup paths
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "color"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "depth"), exist_ok=True)
    
    device = torch.device("cuda")
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    # 1. Load Cameras (to get viewpoints)
    print(f"[Loader] Loading cameras from {args.dataset_path}...")
    cameras = load_scene_data(args.dataset_path, device=device)
    
    # 2. Load Gaussian Model
    print(f"[Loader] Loading model from {args.ply_path}...")
    gaussians = GaussianModel(sh_degree=3)
    if not os.path.exists(args.ply_path):
        print(f"Error: PLY file not found at {args.ply_path}")
        exit(1)
    gaussians.create_from_pcd(args.ply_path, spatial_lr_scale=1.0)
    
    # Freeze model
    gaussians._features_dc.requires_grad = False
    gaussians._features_rest.requires_grad = False
    
    print("[Export] Rendering RGB-D Sequence...")
    
    for idx, cam in enumerate(tqdm(cameras)):
        rgb, depth = render_rgb_and_depth(cam, gaussians, bg_color)
        
        # Save RGB
        rgb_np = rgb.permute(1, 2, 0).detach().cpu().numpy()
        rgb_np = np.clip(rgb_np, 0, 1) * 255
        cv2.imwrite(os.path.join(args.output_path, "color", f"{idx:05d}.jpg"), cv2.cvtColor(rgb_np.astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        # Save Depth (as uint16 in micrometers)
        depth_np = depth[0].detach().cpu().numpy()
        depth_u16 = (depth_np * 1000).astype(np.uint16)
        cv2.imwrite(os.path.join(args.output_path, "depth", f"{idx:05d}.png"), depth_u16)
        
        # Save Pose (World-to-Camera Matrix)
        # 3DGS stores W2C in transposed form. We transpose it back to [4,4].
        w2c = cam.world_view_transform.transpose(0, 1).cpu().numpy()
        np.savetxt(os.path.join(args.output_path, f"{idx:05d}.pose"), w2c)

    print(f"[Export] Done. Data saved to {args.output_path}")