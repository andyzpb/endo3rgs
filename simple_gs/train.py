import torch
import os
import argparse
import torchvision
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from data_loader import load_scene_data
from gaussian_model import GaussianModel
from renderer import render

def train(args):
    dataset_path = args.dataset_path
    output_path = "output"
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "debug"), exist_ok=True)
    
    device = torch.device("cuda")
    
    cameras = load_scene_data(dataset_path, device=device)
    if not cameras: return

    gaussians = GaussianModel(sh_degree=3)
    gaussians.create_from_pcd(os.path.join(dataset_path, "init_points.ply"))
    gaussians.setup_training()

    densify_start_iter = 500
    densify_end_iter = 15000 
    densification_interval = 500
    
    # [Core modification] Disable opacity reset
    # For sparse views, reset is suicidal
    opacity_reset_interval = 30000 
    
    # [Core modification] Moderately lower threshold
    # 0.001 is too hard, 0.0006 is suitable
    max_grad = 0.0006 
    min_opacity = 0.005
    DEPTH_DOWNSAMPLE = 2 

    iter_bar = tqdm(range(1, args.iterations + 1), desc="Training")
    
    for iteration in iter_bar:
        gaussians.optimizer.zero_grad(set_to_none=True)

        viewpoint_cam = cameras[torch.randint(0, len(cameras), (1,)).item()]
        
        if torch.rand(1) > 0.5:
            bg_color = torch.rand((3), device=device)
        else:
            bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

        render_pkg = render(viewpoint_cam, gaussians, bg_color, depth_downsample_factor=DEPTH_DOWNSAMPLE)
        image = render_pkg["render"]
        pred_depth = render_pkg["render_depth"] 
        
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image = viewpoint_cam.image
        gt_depth = viewpoint_cam.depth
        conf = viewpoint_cam.confidence
        
        if gt_depth.shape[-1] != pred_depth.shape[-1]:
            gt_depth_resized = F.interpolate(gt_depth.unsqueeze(0), size=pred_depth.shape[-2:], mode='nearest').squeeze(0)
        else:
            gt_depth_resized = gt_depth

        l1_loss = (torch.abs(image - gt_image) * (conf + 0.1)).mean()
        
        valid_depth_mask = gt_depth_resized > 0
        if valid_depth_mask.sum() > 10:
            p_min, p_max = pred_depth.min(), pred_depth.max()
            p_norm = (pred_depth - p_min) / (p_max - p_min + 1e-6)
            g_valid = gt_depth_resized[valid_depth_mask]
            g_min, g_max = g_valid.min(), g_valid.max()
            g_norm = torch.zeros_like(gt_depth_resized)
            g_norm[valid_depth_mask] = (gt_depth_resized[valid_depth_mask] - g_min) / (g_max - g_min + 1e-6)
            depth_loss = torch.abs(p_norm - g_norm) * valid_depth_mask
            depth_loss = depth_loss.mean()
        else:
            depth_loss = torch.tensor(0.0, device=device)
        
        opacities = gaussians.get_opacity
        ent_loss = - (opacities * torch.log(opacities + 1e-6) + (1-opacities) * torch.log(1-opacities + 1e-6)).mean()
        
        scales = gaussians.get_scaling
        max_s = torch.max(scales, dim=1).values
        min_s = torch.min(scales, dim=1).values
        scale_reg = torch.mean(torch.relu(max_s / (min_s + 1e-6) - 10.0))

        loss = l1_loss + 0.2 * depth_loss + 0.005 * ent_loss + 0.05 * scale_reg
        loss.backward()

        with torch.no_grad():
            if iteration < densify_end_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration > densify_start_iter and iteration % densification_interval == 0:
                n_pts = gaussians.get_xyz.shape[0]
                
                # [Temperature control] 
                # Here we adjust back to normal range, no longer using extreme 0.001
                if n_pts < 100000:
                    cur_grad = 0.0004
                elif n_pts < 200000:
                    cur_grad = 0.0006
                else:
                    cur_grad = 0.0008 
                
                size_threshold = 20 if iteration > 3000 else None
                gaussians.densify_and_prune(cur_grad, min_opacity, 5.0, size_threshold)
            
            if iteration % opacity_reset_interval == 0:
                gaussians.reset_opacity()
            
            if iteration % 100 == 0:
                torch.cuda.empty_cache()

        if iteration % 100 == 0:
            iter_bar.set_description(f"L1: {l1_loss:.4f} | Dpt: {depth_loss:.4f} | Pts: {gaussians.get_xyz.shape[0]}")
            
        if iteration % 1000 == 0:
            vis = torch.cat([gt_image, image], dim=2)
            torchvision.utils.save_image(vis, os.path.join(output_path, "debug", f"iter_{iteration:05d}.png"))

    gaussians.save_ply(os.path.join(output_path, "final_refined_model.ply"))
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=7000)
    args = parser.parse_args()
    train(args)