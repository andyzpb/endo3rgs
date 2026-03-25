import argparse, os, torch, json, cv2, trimesh, numpy as np
from dust3r.inference import inference
from dust3r.model import Endo3R 
from dust3r.image_pairs import make_pairs
from third_party.sam2.checkpoints import global_aligner, GlobalAlignerMode

def run_hybrid_pipeline(args):
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Endo3R model (use its memory module weights)
    model = Endo3R(use_feat=False).to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict({k.replace('module.', ''): v for k, v in (ckpt['model'] if 'model' in ckpt else ckpt).items()}, strict=False)
    model.eval()

    # 2. Load images and GT calibration
    from run_endo3r_preprocess import load_images_force_square, load_calibration
    images = load_images_force_square(args.image_dir, target_size=512)
    K_gt, W_gt, H_gt = load_calibration("calibration_result.json")

    # 3. Fast inference: use sliding window (swin-5) instead of fully connected
    # For consecutive videos, 5-frame context ensures global coherence with 4x efficiency gain
    print(">> [Step 1] Running Sliding Window Inference...")
    pairs = make_pairs(images, scene_graph='swin-5', symmetrize=True)
    output = inference(pairs, model.dust3r, device, batch_size=1, use_amp=True)

    # 4. Global Optimization: enable instrument rejection (Self-Mask)
    print(">> [Step 2] Global Alignment with Instrument Rejection...")
    scene = global_aligner(output, device=device, 
                           mode=GlobalAlignerMode.PointCloudOptimizer,
                           use_self_mask=True,       # Core: trigger Sampson/projection error filtering
                           flow_loss_weight=1.0,     # Increase geometric consistency
                           motion_mask_thre=0.2)     # Sensitivity adjustment

    # [Critical] Lock GT intrinsics
    scale_factor = 512 / W_gt
    scene.preset_focal([K_gt[0, 0] * scale_factor] * len(images), requires_grad=False)
    
    # Run optimization (200 iterations is enough for convergence)
    scene.compute_global_alignment(init='mst', niter=200, schedule='linear', lr=0.01)

    # 5. Export PLY
    print(">> [Step 3] Exporting Results...")
    pts3d = [p.detach().cpu().numpy() for p in scene.get_pts3d()]
    masks = [m.detach().cpu().numpy() for m in scene.get_masks()]
    all_pts, all_cols = [], []
    for i, img_data in enumerate(images):
        valid = masks[i] & (pts3d[i][..., 2] > 0)
        all_pts.append(pts3d[i][valid])
        all_cols.append(((img_data['img'][0].permute(1,2,0).cpu().numpy()+1.0)/2.0)[valid])
    
    pcd = trimesh.points.PointCloud(np.concatenate(all_pts), colors=(np.concatenate(all_cols)*255).astype(np.uint8))
    pcd.export(os.path.join(args.output_dir, 'hybrid_final.ply'))
    print(f"Success! Result: {args.output_dir}/hybrid_final.ply")

if __name__ == '__main__':
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='data/endo_hybrid')
    parser.add_argument('--ckpt_path', type=str, default='checkpoints/endo3r.pth')
    parser.add_argument('--device', type=str, default='cuda')
    run_hybrid_pipeline(parser.parse_args())