import os
import torch
import numpy as np
import argparse
import json
import trimesh
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner
from dust3r.utils.device import to_numpy

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to folder containing surgery images")
    parser.add_argument("--model_path", type=str, default="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", help="Path to pretrained model")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save GS input data")
    parser.add_argument("--device", type=str, default='cuda')
    return parser.parse_args()

def save_transforms_json(imgs, poses, intrinsics, output_dir):
    """
    保存为 NeRF/3DGS 常用的 transforms.json 格式
    """
    out = {"camera_angle_x": 0.0, "frames": []} 
    
    # 此时 imgs 已经是裁剪后的 912x912
    # intrinsics 也是我们修正后的 (cx-6, cy-6)
    H, W = imgs[0]['true_shape'][0][:2] # 应该是 912, 912
    focal = intrinsics[0][0, 0]
    out["camera_angle_x"] = 2 * np.arctan(W / (2 * focal))
    
    # 保存详细参数
    out["fl_x"] = float(intrinsics[0][0, 0])
    out["fl_y"] = float(intrinsics[0][1, 1])
    out["cx"] = float(intrinsics[0][0, 2])
    out["cy"] = float(intrinsics[0][1, 2])
    out["w"] = int(W)
    out["h"] = int(H)

    for i, (img, pose) in enumerate(zip(imgs, poses)):
        c2w = pose 
        frame_data = {
            "file_path": os.path.basename(img['path']), # 注意：这里引用原文件名，但3DGS读取时需要注意尺寸差异，通常没问题
            "transform_matrix": c2w.tolist(),
            "colmap_id": i
        }
        out["frames"].append(frame_data)

    with open(os.path.join(output_dir, "transforms.json"), "w") as f:
        json.dump(out, f, indent=4)
    print(f"[Info] Saved transforms.json to {output_dir}")

def main():
    args = get_args()
    device = torch.device(args.device)

    # 1. 加载模型
    print(f"[Step 1] Loading model from {args.model_path}...")
    try:
        model = AsymmetricCroCo3DStereo.from_pretrained(args.model_path).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. 加载图像 
    # FIX: size=924 (触发 crop 到 912), square_ok=True (保持正方形)
    print(f"[Step 2] Loading images from {args.image_path}...")
    images = load_images(args.image_path, size=924, square_ok=True) 
    
    if len(images) == 0:
        raise ValueError("No images found!")
    
    # 验证一下裁剪后的尺寸
    actual_h, actual_w = images[0]['true_shape'][0][:2]
    print(f"Loaded {len(images)} images. Resized/Cropped to: {actual_w}x{actual_h}")
    
    # 计算裁剪偏移量 (用于修正 GT 内参)
    # 原图 924 -> Crop 912. Offset = (924 - 912) / 2 = 6
    orig_w = 924
    offset_x = (orig_w - actual_w) / 2
    offset_y = (orig_w - actual_h) / 2
    print(f"Crop offset detected: x={offset_x}, y={offset_y}")

    # 3. 全局对齐与 GT 内参注入
    print(f"[Step 3] Running DUSt3R inference and Global Alignment...")
    
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    scene = global_aligner(pairs, model, device, verbose=True, optimize_pp=True)

    # --- GT 内参注入 ---
    print("[Info] Injecting Ground Truth Intrinsics (Adjusted for Crop)...")
    
    # 原始 GT (924x924)
    fx = 870.7642141449198
    fy = 876.787018860571
    cx = 386.98873045756136
    cy = 491.08567174134566
    
    # 修正后的 GT (912x912)
    adjusted_cx = cx - offset_x
    adjusted_cy = cy - offset_y
    
    gt_K_adjusted = np.array([
        [fx, 0.0, adjusted_cx],
        [0.0, fy, adjusted_cy],
        [0.0, 0.0, 1.0]
    ])
    
    print(f"Original Principal Point: ({cx:.2f}, {cy:.2f})")
    print(f"Adjusted Principal Point: ({adjusted_cx:.2f}, {adjusted_cy:.2f})")

    n_imgs = len(images)
    K_list = [gt_K_adjusted for _ in range(n_imgs)]
    
    scene.preset_intrinsics(K_list)
    # --- GT 内参注入结束 ---
    
    # 运行优化
    scene.compute_global_alignment(init='mst', niter=300, schedule='linear', lr=0.01)

    # 4. 提取数据
    print(f"[Step 4] Extracting Point Cloud and Poses...")
    
    poses = to_numpy(scene.get_im_poses()) 
    intrinsics = to_numpy(scene.get_intrinsics())
    
    pts3d_list = []
    colors_list = []
    imgs = scene.imgs
    
    for i in range(len(imgs)):
        pts = scene.get_pts3d(i).detach().cpu().numpy()
        rgb = imgs[i]['img'].detach().cpu().numpy()
        rgb = np.transpose(rgb, (1, 2, 0)) 
        
        valid_pts = pts.reshape(-1, 3)
        valid_rgb = rgb.reshape(-1, 3)
        
        pts3d_list.append(valid_pts)
        colors_list.append(valid_rgb)

    all_pts = np.concatenate(pts3d_list, axis=0)
    all_rgb = np.concatenate(colors_list, axis=0)

    # 5. 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    pcd = trimesh.PointCloud(vertices=all_pts, colors=all_rgb)
    ply_path = os.path.join(args.output_dir, "points3d_init.ply")
    pcd.export(ply_path)
    print(f"[Info] Saved combined point cloud to {ply_path} ({len(all_pts)} points)")

    save_transforms_json(images, poses, intrinsics, args.output_dir)

if __name__ == '__main__':
    main()