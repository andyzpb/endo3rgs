import torch
import torch.nn as nn
import numpy as np
from simple_knn._C import distCUDA2
import trimesh

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

class GaussianModel:
    def __init__(self, device='cuda'):
        self.device = device
        # 核心参数
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0) # 基础颜色
        self._features_rest = torch.empty(0) # 视角相关颜色 (SH)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.optimizer = None

    def create_from_pcd(self, pcd_path):
        """
        核心结合点：从 Endo3R (Stage 1) 的 .ply 文件加载初始化
        """
        print(f"[Model] Loading point cloud from {pcd_path}...")
        pcd = trimesh.load(pcd_path)
        
        # 1. 坐标 XYZ
        xyz = torch.tensor(pcd.vertices, dtype=torch.float32, device=self.device)
        
        # 2. 颜色 RGB (转换为球谐系数 SH 的 DC 分量)
        # Endo3R 输出的是 0-255 的 RGB，需要归一化并转换
        colors = torch.tensor(pcd.colors[:, :3], dtype=torch.float32, device=self.device) / 255.0
        # RGB -> SH (简单的近似: (RGB - 0.5) / 0.28209)
        C0 = 0.28209479177387814
        features_dc = (colors - 0.5) / C0
        
        # 3. 缩放 Scale
        # 初始缩放设为最近邻距离的平均值，防止太大或太小
        dist2 = torch.clamp_min(distCUDA2(xyz), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        
        # 4. 旋转 Rotation (四元数，默认为单位旋转)
        rots = torch.zeros((xyz.shape[0], 4), device=self.device)
        rots[:, 0] = 1 # w=1, x=y=z=0

        # 5. 不透明度 Opacity (初始设为 0.1，经过 sigmoid 后)
        opacities = inverse_sigmoid(0.1 * torch.ones((xyz.shape[0], 1), device=self.device))

        # 设置为可训练参数
        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.unsqueeze(1).requires_grad_(True))
        self._features_rest = nn.Parameter(torch.zeros((xyz.shape[0], 15, 3), device=self.device).requires_grad_(True)) # 3阶SH
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        print(f"[Model] Initialized {self._xyz.shape[0]} Gaussians from Endo3R.")

    def setup_training(self, lr_xyz=0.00016, lr_rgb=0.0025):
        # 定义优化器
        l = [
            {'params': [self._xyz], 'lr': lr_xyz, "name": "xyz"},
            {'params': [self._features_dc], 'lr': lr_rgb, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': lr_rgb / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': 0.05, "name": "opacity"},
            {'params': [self._scaling], 'lr': 0.005, "name": "scaling"},
            {'params': [self._rotation], 'lr': 0.001, "name": "rotation"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    @property
    def get_xyz(self): return self._xyz
    @property
    def get_rotation(self): return torch.nn.functional.normalize(self._rotation)
    @property
    def get_scaling(self): return torch.exp(self._scaling)
    @property
    def get_opacity(self): return torch.sigmoid(self._opacity)
    @property
    def get_features(self): return torch.cat((self._features_dc, self._features_rest), dim=1)