import torch
import numpy as np
import json
import os
import cv2
from collections import namedtuple

# Define camera structure, added depth field
Camera = namedtuple("Camera", ["w2c", "camera_center", "image", "depth", "confidence", "fx", "fy", "cx", "cy", "width", "height"])

def load_scene_data(dataset_path, device):
    print(f"[Loader] Loading data from {dataset_path}...")
    
    json_path = os.path.join(dataset_path, "cameras.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"cameras.json not found in {dataset_path}")
        
    with open(json_path, 'r') as f:
        cam_data = json.load(f)
    
    cameras = []
    # Sort by numeric order of filenames
    keys = sorted(cam_data.keys(), key=lambda x: int(x))
    
    for k in keys:
        v = cam_data[k]
        
        # 1. Load RGB image
        img_path = os.path.join(dataset_path, "images", v['img_name'])
        if not os.path.exists(img_path):
            print(f"[Warning] Image not found: {img_path}, skipping.")
            continue
            
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        img_tensor = torch.tensor(img / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        
        # 2. Load Depth image (critical step)
        # Assuming depth images are in dataset_path/depth/, and filenames match images
        base_name = os.path.splitext(os.path.basename(v['img_name']))[0]
        
        # Try to match png or jpg extensions
        depth_path = None
        for ext in ['.png', '.jpg', '.tif', '.tiff']:
            temp_path = os.path.join(dataset_path, "depth", base_name + ext)
            if os.path.exists(temp_path):
                depth_path = temp_path
                break
        
        if depth_path is None:
            # If no depth image found, this is a serious warning, but to avoid errors we can use all zeros
            print(f"[Warning] Depth map not found for {base_name}. Using Zero depth.")
            depth_tensor = torch.zeros((1, img_tensor.shape[1], img_tensor.shape[2]), dtype=torch.float32, device=device)
        else:
            # Load 16-bit depth image (unit: millimeters)
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_img is None:
                raise RuntimeError(f"Failed to read depth image: {depth_path}")
            
            # Convert to tensor and divide by 1000 to convert to Meters
            # Note: we keep absolute values here, no [0,1] normalization, leave it for Loss function to handle
            depth_tensor = torch.tensor(depth_img, dtype=torch.float32, device=device) / 1000.0
            depth_tensor = depth_tensor.unsqueeze(0) # [H, W] -> [1, H, W]

        # 3. Load Confidence maps (if available)
        conf_path = os.path.join(dataset_path, "confidence_maps", base_name + ".png")
        if os.path.exists(conf_path):
            conf = cv2.imread(conf_path, cv2.IMREAD_GRAYSCALE)
            # Resize in case dimensions don't match
            if conf.shape != (img.shape[0], img.shape[1]):
                conf = cv2.resize(conf, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            conf_tensor = torch.tensor(conf / 255.0, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            conf_tensor = torch.ones_like(depth_tensor)

        # 4. Camera intrinsics and extrinsics
        fx, fy = v['fx'], v['fy']
        cx, cy = v['width'] / 2, v['height'] / 2 # Simplified assumption of center
        if 'cx' in v: cx = v['cx']
        if 'cy' in v: cy = v['cy']

        # W2C matrix processing
        # json stores c2w (Camera to World)
        c2w = np.array(v['c2w'])
        w2c = np.linalg.inv(c2w)
        w2c_tensor = torch.tensor(w2c, dtype=torch.float32, device=device)
        cam_center = torch.tensor(c2w[:3, 3], dtype=torch.float32, device=device)

        cameras.append(Camera(
            w2c=w2c_tensor,
            camera_center=cam_center,
            image=img_tensor,
            depth=depth_tensor,  # contains real depth information
            confidence=conf_tensor,
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=v['width'], height=v['height']
        ))
        
    print(f"[Loader] Successfully loaded {len(cameras)} images (with depth).")
    return cameras