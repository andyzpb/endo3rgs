import argparse
import os
import torch
import torch.nn as nn
import json
import numpy as np
import trimesh
import cv2
import glob
import time
from concurrent.futures import ThreadPoolExecutor
from dust3r.inference import inference
from dust3r.model import Endo3R 
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# =========================================================
# [Configuration] Box Prompt Parameters
# =========================================================
ORIG_W, ORIG_H = 924, 924
# Your Box coordinates (Original image bottom-left coordinate system)
RAW_COORDS_X = [924, 924, 500, 500]
RAW_COORDS_Y = [185, 578, 185, 578]

# Known average real depth of this area (mm)
FIXED_REAL_DEPTH = 3.5 
# =========================================================

# -----------------------------------------------------------------------------
# 1. Load Calibration
# -----------------------------------------------------------------------------
CALIBRATION_FILE = "calibration_result.json"

def load_calibration(calib_path):
    print(f"[Calibration] Loading from {calib_path}...")
    with open(calib_path, 'r') as f:
        data = json.load(f)
    K = np.array(data["K"], dtype=np.float32)
    width = data["image_size_after_crop"]["width"]
    height = data["image_size_after_crop"]["height"]
    return K, width, height

# -----------------------------------------------------------------------------
# 2. Helper Functions & Parallel Processing Core
# -----------------------------------------------------------------------------
def get_instrument_box(target_size):
    """ Calculate Box coordinates based on target size """
    scale = target_size / ORIG_W
    x_min_raw = min(RAW_COORDS_X)
    x_max_raw = max(RAW_COORDS_X)
    y_min_raw = ORIG_H - max(RAW_COORDS_Y) 
    y_max_raw = ORIG_H - min(RAW_COORDS_Y) 
    
    box = np.array([
        x_min_raw * scale, y_min_raw * scale, 
        x_max_raw * scale, y_max_raw * scale
    ])
    return box.astype(np.int32)

def get_instrument_mask_simple(img_uint8, box):
    """ 
    Color Segmenter: Extract low saturation areas (instruments) 
    Replaces heavy SAM2 model with simple OpenCV logic.
    """
    H, W, _ = img_uint8.shape
    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])
    
    roi = img_uint8[y1:y2, x1:x2]
    if roi.size == 0: return np.zeros((H, W), dtype=bool)

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_roi)

    # Thresholds: Saturation < 80 (greyish) and Value > 20 (not too dark/shadow)
    inst_mask_roi = (s < 80) & (v > 20)

    full_mask = np.zeros((H, W), dtype=bool)
    full_mask[y1:y2, x1:x2] = inst_mask_roi
    return full_mask

def remove_highlights(img_bgr, threshold=220):
    """ 
    Remove specular highlights (CPU-intensive operation).
    Uses inpainting to fill saturated white areas.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, threshold]), np.array([180, 60, 255]))
    if cv2.countNonZero(mask) == 0: return img_bgr
    mask_dilated = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
    
    # INPAINT_TELEA is relatively slow, but acceptable under multithreading
    return cv2.inpaint(img_bgr, mask_dilated, 3, cv2.INPAINT_TELEA)

def process_single_image_worker(args):
    """ 
    Worker function for single image processing (used in thread pool).
    args: (index, file_path, target_size)
    """
    idx, file_path, target_size = args
    
    # Read image
    img_bgr = cv2.imread(file_path)
    if img_bgr is None:
        return None
        
    # Preprocessing: Remove highlights -> Resize -> Convert to RGB -> Convert to Tensor
    # This step runs in parallel threads, saving significant time compared to serial processing
    img_bgr = remove_highlights(img_bgr)
    
    img_resized = cv2.resize(img_bgr, (target_size, target_size), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [-1, 1]
    img_tensor = torch.from_numpy(img_rgb).float() / 127.5 - 1.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    
    return {
        'img': img_tensor,
        'true_shape': np.array([[target_size, target_size]]),
        'idx': idx, 
        'instance': file_path
    }

def load_images_parallel(folder_path, target_size=512, max_workers=8):
    """ Load and preprocess images in parallel """
    print(f">> [Loader] Scanning images in {folder_path}...")
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    files.sort()
    
    print(f">> [Loader] Found {len(files)} images. Starting parallel processing with {max_workers} threads...")
    start_time = time.time()
    
    # Prepare argument list for the worker
    tasks = [(i, f, target_size) for i, f in enumerate(files)]
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # executor.map returns results in order, ensuring image sequence is preserved
        out_iter = executor.map(process_single_image_worker, tasks)
        for res in out_iter:
            if res is not None:
                results.append(res)
                
    print(f">> [Loader] Loaded {len(results)} images in {time.time() - start_time:.2f}s")
    return results

# -----------------------------------------------------------------------------
# 3. Main Pipeline
# -----------------------------------------------------------------------------
def run_endo3r_pipeline(args):
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare directories
    dirs = {k: os.path.join(args.output_dir, k) for k in ["confidence_maps", "depth", "instrument_masks"]}
    for d in dirs.values(): os.makedirs(d, exist_ok=True)

    # 1. Load Model (Note: torch.compile removed to fix graph break errors)
    print(f"[Endo3R] Loading model...")
    model = Endo3R(use_feat=False).to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)
    model.eval()
    
    # 2. Load Data in Parallel
    # 320x320 is a good balance between speed and detail for endoscopy
    TARGET_SIZE = 320 
    # Use CPU core count as thread number to avoid excessive context switching
    num_workers = min(os.cpu_count(), 16)
    images = load_images_parallel(args.image_dir, target_size=TARGET_SIZE, max_workers=num_workers)
    
    K_gt, W_gt, H_gt = load_calibration(CALIBRATION_FILE)
    inst_box = get_instrument_box(TARGET_SIZE)

    # 3. Inference (Core Optimization: Swin-3)
    print(f"[Endo3R] Inference...")
    # Optimization 1: scene_graph='swin-3' (was 'swin-5') -> Reduced pairs, significant speedup
    # Optimization 2: batch_size=1 is safe; try 2 if VRAM allows
    pairs = make_pairs(images, scene_graph='swin-3', symmetrize=True)
    print(f">> [Info] Generated {len(pairs)} pairs (Swin-3 strategy)")
    
    start_infer = time.time()
    # Note: using torch.amp.autocast for mixed precision
    output = inference(pairs, model.dust3r, device, batch_size=1, use_amp=True)
    print(f">> [Time] Inference took {time.time() - start_infer:.2f}s")

    # 4. Global Optimization (Core Optimization: Reduced iterations)
    print(f"[Endo3R] Global Optimization...")
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    
    scale_factor = TARGET_SIZE / W_gt
    focal_gt_px = K_gt[0, 0] * scale_factor
    scene.preset_focal([focal_gt_px] * len(images), requires_grad=False)
    
    # Optimization 3: niter=100 (was 150), lr=0.03 (was 0.02) -> Faster convergence
    scene.compute_global_alignment(init='mst', niter=100, schedule='cosine', lr=0.03)

    # 5. Export & Scale Recovery
    print("[Scale] Recovering Absolute Scale & Exporting...")
    poses = scene.get_im_poses().detach().cpu().numpy()
    pts3d = [p.detach().cpu().numpy() for p in scene.get_pts3d()]
    masks = [m.detach().cpu().numpy() for m in scene.get_masks()]
    
    cameras_meta = {}
    all_pts, all_cols = [], []
    calculated_scales = []

    for i, img_data in enumerate(images):
        base_name = os.path.splitext(os.path.basename(img_data['instance']))[0]
        c2w = poses[i]
        pts = pts3d[i]
        msk = masks[i]
        
        # Image processing (inverse operation of tensor conversion for saving)
        img_np = (img_data['img'].squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0
        img_uint8 = (img_np * 255).astype(np.uint8)

        # Generate depth map
        w2c = np.linalg.inv(c2w)
        # Simple projection calculation
        pts_cam = (np.dot(w2c[:3, :3], pts.reshape(-1, 3).T).T) + w2c[:3, 3]
        depth_map = pts_cam[:, 2].reshape(TARGET_SIZE, TARGET_SIZE)
        
        # Get Mask
        inst_mask = get_instrument_mask_simple(img_uint8, inst_box)

        # Save Mask visualization
        mask_viz = (inst_mask * 255).astype(np.uint8)
        cv2.rectangle(mask_viz, (inst_box[0], inst_box[1]), (inst_box[2], inst_box[3]), (127), 2)
        cv2.imwrite(os.path.join(dirs["instrument_masks"], f"{base_name}_mask.png"), mask_viz)

        # Scale calculation
        if np.any(inst_mask):
            masked_depths = depth_map[inst_mask]
            valid_depths = masked_depths[masked_depths > 0]
            if len(valid_depths) > 10:
                est_depth = np.median(valid_depths)
                if est_depth > 0.1:
                    scale = FIXED_REAL_DEPTH / est_depth
                    calculated_scales.append(scale)
        
        # Collect data for export
        cameras_meta[i] = {"id": i, "img_name": base_name, "c2w": c2w.tolist()}
        valid = msk > min(args.conf_thresh, max(1e-3, msk.max() * 0.1))
        all_pts.append(pts[valid])
        all_cols.append(img_np[valid])
        
        # Save depth map visualization
        depth_viz = (depth_map / (depth_map.max()+1e-6) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(dirs["depth"], f"{base_name}.png"), depth_viz)

    # Apply scale
    final_scale = np.median(calculated_scales) if calculated_scales else 1.0
    print(f"\n>> [Result] Final Absolute Scale Factor: {final_scale:.4f}")

    full_pts = np.concatenate(all_pts) * final_scale
    full_cols = np.concatenate(all_cols)
    
    for i in cameras_meta:
        c2w = np.array(cameras_meta[i]['c2w'])
        c2w[:3, 3] *= final_scale
        cameras_meta[i]['c2w'] = c2w.tolist()

    with open(os.path.join(args.output_dir, 'cameras_metric.json'), 'w') as f:
        json.dump(cameras_meta, f, indent=4)

    if len(full_pts) > 0:
        # Downsample point cloud to speed up saving
        idx = np.random.choice(len(full_pts), min(len(full_pts), 200000), replace=False)
        pcd = trimesh.points.PointCloud(full_pts[idx], colors=(full_cols[idx]*255).astype(np.uint8))
        pcd.export(os.path.join(args.output_dir, 'metric_points.ply'))
        print(f"[Success] Saved metric reconstruction to {args.output_dir}")

if __name__ == '__main__':
    # Enable cuDNN benchmark/autotuner
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='data/endo_gs_processed')
    parser.add_argument('--ckpt_path', type=str, default='checkpoints/endo3r.pth')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--conf_thresh', type=float, default=2) 
    args = parser.parse_args()
    
    run_endo3r_pipeline(args)