import os
import json
import torch
import numpy as np
from PIL import Image
from typing import NamedTuple

class Camera(NamedTuple):
    image: torch.Tensor
    R: torch.Tensor
    T: torch.Tensor
    FoVx: float
    FoVy: float
    image_path: str
    uid: int

def load_scene_data(dataset_path, device='cuda'):
    json_path = os.path.join(dataset_path, "transforms.json")
    with open(json_path, 'r') as f:
        meta = json.load(f)

    cameras = []
    # 假设所有图片宽高一致，取第一张计算 FoV
    # 注意：Stage 1 我们计算的是 camera_angle_x
    fov_x = meta["camera_angle_x"]
    
    frames = meta["frames"]
    print(f"[Loader] Found {len(frames)} frames in transforms.json")

    for idx, frame in enumerate(frames):
        # 1. 读取图像
        img_path = os.path.join(dataset_path, "images", os.path.basename(frame["file_path"]))
        # 兼容一下路径，如果 json 里存的是相对路径
        if not os.path.exists(img_path):
             img_path = os.path.join(dataset_path, frame["file_path"])
        
        pil_image = Image.open(img_path)
        image = torch.from_numpy(np.array(pil_image)).float() / 255.0
        if image.shape[2] == 3: # H, W, 3 -> 3, H, W
            image = image.permute(2, 0, 1)
        image = image.to(device)
        
        height, width = image.shape[1], image.shape[2]
        fov_y = 2 * np.arctan(np.tan(fov_x / 2) * height / width)

        # 2. 读取位姿 (c2w -> w2c)
        c2w = np.array(frame["transform_matrix"])
        # 3DGS 的光栅化器通常需要 World-to-Camera 矩阵，且坐标系定义略有不同
        # 这里先直接求逆，后续如果渲染是反的，需要翻转 Y/Z 轴
        w2c = np.linalg.inv(c2w)
        
        R = torch.tensor(w2c[:3, :3]).float().to(device)
        T = torch.tensor(w2c[:3, 3]).float().to(device)

        cameras.append(Camera(
            image=image, R=R, T=T, 
            FoVx=fov_x, FoVy=fov_y, 
            image_path=img_path, uid=idx
        ))
        
    return cameras