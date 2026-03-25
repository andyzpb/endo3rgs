import os
import torch
import torch.nn as nn
import numpy as np
from simple_knn._C import distCUDA2
import trimesh
from plyfile import PlyData, PlyElement

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device='cuda')
    r = q[:, 0]; x = q[:, 1]; y = q[:, 2]; z = q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y*y + z*z); R[:, 0, 1] = 2 * (x*y - r*z); R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z); R[:, 1, 1] = 1 - 2 * (x*x + z*z); R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y); R[:, 2, 1] = 2 * (y*z + r*x); R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

class GaussianModel:
    def __init__(self, device='cuda', sh_degree=3):
        self.device = device
        self.sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0.01
        self.spatial_lr_scale = 0

    def create_from_pcd(self, pcd_path):
        print(f"[Model] Loading NORMALIZED Point Cloud: {pcd_path}")
        try:
            pcd = trimesh.load(pcd_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load PLY: {e}")
        
        if isinstance(pcd, trimesh.Scene):
            verts = []
            cols = []
            for g in pcd.geometry.values():
                if hasattr(g, 'vertices'):
                    verts.append(g.vertices)
                    if hasattr(g.visual, 'vertex_colors'):
                        cols.append(g.visual.vertex_colors[:, :3])
                    else:
                        cols.append(np.ones_like(g.vertices) * 128)
            if not verts: raise ValueError("Scene has no vertices")
            vertices = np.vstack(verts)
            colors = np.vstack(cols) / 255.0
        else:
            if not hasattr(pcd, 'vertices'): raise ValueError("PCD has no vertices")
            vertices = pcd.vertices
            if hasattr(pcd.visual, 'vertex_colors'):
                colors = pcd.visual.vertex_colors[:, :3] / 255.0
            else:
                colors = np.ones_like(vertices) * 0.5

        num_pts = vertices.shape[0]
        print(f"[Model] Initialized with {num_pts} points.")

        xyz = torch.tensor(vertices, dtype=torch.float32, device=self.device)
        colors = torch.tensor(colors, dtype=torch.float32, device=self.device)
        
        C0 = 0.28209479177387814
        features_dc = (colors - 0.5) / C0
        
        dist2 = torch.clamp_min(distCUDA2(xyz), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((num_pts, 4), device=self.device)
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.5 * torch.ones((num_pts, 1), device=self.device))

        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.unsqueeze(1).requires_grad_(True))
        self._features_rest = nn.Parameter(torch.zeros((num_pts, 15, 3), device=self.device).requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        self.spatial_lr_scale = 5.0 
        
        self.xyz_gradient_accum = torch.zeros((num_pts, 1), device=self.device)
        self.denom = torch.zeros((num_pts, 1), device=self.device)
        self.max_radii2D = torch.zeros((num_pts), device=self.device)

    def setup_training(self, lr_xyz=0.00016, lr_rgb=0.0025):
        l = [
            {'params': [self._xyz], 'lr': lr_xyz * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': lr_rgb, "name": "features_dc"},
            {'params': [self._features_rest], 'lr': lr_rgb / 20.0, "name": "features_rest"},
            {'params': [self._opacity], 'lr': 0.05, "name": "opacity"},
            {'params': [self._scaling], 'lr': 0.005, "name": "scaling"},
            {'params': [self._rotation], 'lr': 0.001, "name": "rotation"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self._replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def _replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[name] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    optimizable_tensors[name] = group["params"][0]
        return optimizable_tensors

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        grads = grads.squeeze()
        
        # === [VRAM Emergency Brake Mechanism] ===
        # Hard limit: 500K points. Exceeding this, VRAM must blow up, must force stop loss.
        MAX_POINTS = 2000000 
        current_points = self.get_xyz.shape[0]
        
        if current_points > MAX_POINTS:
            # 1. Emergency pruning mode: disable splitting, only aggressive pruning
            # Strategy: remove gaussians with opacity < 0.05 directly (10x higher than normal threshold)
            print(f"[Model] EMERGENCY: Pts {current_points} > {MAX_POINTS}. Force pruning weak points.")
            prune_mask = (self.get_opacity < 0.05).squeeze()
            self._prune_points(prune_mask)
            
            # Clear gradients and return directly, skip splitting step
            self.xyz_gradient_accum.zero_()
            self.denom.zero_()
            self.max_radii2D.zero_()
            return

        # Normal splitting logic (only execute when point count is safe)
        selected_pts_mask = grads >= max_grad
        big_points_mask = torch.max(self.get_scaling, dim=1).values > extent * 0.01
        selected_pts_mask = torch.logical_or(selected_pts_mask, big_points_mask)

        max_scales = torch.max(self.get_scaling, dim=1).values
        new_data = {"xyz": [], "features_dc": [], "features_rest": [], "opacity": [], "scaling": [], "rotation": []}

        clone_mask = torch.logical_and(selected_pts_mask, max_scales <= self.percent_dense * extent)
        if clone_mask.any():
            new_data["xyz"].append(self._xyz[clone_mask])
            new_data["features_dc"].append(self._features_dc[clone_mask])
            new_data["features_rest"].append(self._features_rest[clone_mask])
            new_data["opacity"].append(self._opacity[clone_mask])
            new_data["scaling"].append(self._scaling[clone_mask])
            new_data["rotation"].append(self._rotation[clone_mask])

        split_mask = torch.logical_and(selected_pts_mask, max_scales > self.percent_dense * extent)
        if split_mask.any():
            stds = self.get_scaling[split_mask].repeat(2, 1)
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[split_mask]).repeat(2, 1, 1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[split_mask].repeat(2, 1)
            new_scaling = torch.log(self.get_scaling[split_mask].repeat(2, 1) / 1.6)
            new_data["xyz"].append(new_xyz)
            new_data["features_dc"].append(self._features_dc[split_mask].repeat(2, 1, 1))
            new_data["features_rest"].append(self._features_rest[split_mask].repeat(2, 1, 1))
            new_data["opacity"].append(self._opacity[split_mask].repeat(2, 1))
            new_data["scaling"].append(new_scaling)
            new_data["rotation"].append(self._rotation[split_mask].repeat(2, 1))

        if len(new_data["xyz"]) > 0:
            self._densification_postfix(
                torch.cat(new_data["xyz"]), torch.cat(new_data["features_dc"]),
                torch.cat(new_data["features_rest"]), torch.cat(new_data["opacity"]),
                torch.cat(new_data["scaling"]), torch.cat(new_data["rotation"])
            )

        # Normal pruning
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
        too_big = torch.max(self.get_scaling, dim=1).values > extent * 0.1
        prune_mask = torch.logical_or(prune_mask, too_big)
        self._prune_points(prune_mask)
        
        self.xyz_gradient_accum.zero_()
        self.denom.zero_()
        self.max_radii2D.zero_()

    def _densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation):
        d = {"xyz": new_xyz, "features_dc": new_features_dc, "features_rest": new_features_rest, "opacity": new_opacity, "scaling": new_scaling, "rotation": new_rotation}
        for group in self.optimizer.param_groups:
            if group["name"] not in d: continue
            extension = d[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension)), dim=0)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                setattr(self, "_"+group["name"], group["params"][0])
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension), dim=0).requires_grad_(True))
                setattr(self, "_"+group["name"], group["params"][0])
        
        zeros = torch.zeros((new_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum = torch.cat((self.xyz_gradient_accum, zeros), 0)
        self.denom = torch.cat((self.denom, zeros), 0)
        self.max_radii2D = torch.cat((self.max_radii2D, zeros.squeeze()), 0)

    def _prune_points(self, mask):
        valid = ~mask
        if torch.sum(valid) < 1000: return 
        
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][valid]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][valid]
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][valid].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                setattr(self, "_"+group["name"], group["params"][0])
            else:
                group["params"][0] = nn.Parameter((group["params"][0][valid].requires_grad_(True)))
                setattr(self, "_"+group["name"], group["params"][0])

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid]
        self.denom = self.denom[valid]
        self.max_radii2D = self.max_radii2D[valid]

    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().flatten(start_dim=1).cpu().numpy()
        f_rest = self._features_rest.detach().flatten(start_dim=1).cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        SH_C0 = 0.28209479177387814
        rgb = np.clip((f_dc * SH_C0 + 0.5), 0.0, 1.0) * 255
        rgb = rgb.astype(np.uint8)
        dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                      ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        dtype_full.append(('f_dc_0', 'f4')); dtype_full.append(('f_dc_1', 'f4')); dtype_full.append(('f_dc_2', 'f4'))
        for i in range(f_rest.shape[1]): dtype_full.append((f'f_rest_{i}', 'f4'))
        dtype_full.append(('opacity', 'f4'))
        for i in range(scale.shape[1]): dtype_full.append((f'scale_{i}', 'f4'))
        for i in range(rotation.shape[1]): dtype_full.append((f'rot_{i}', 'f4'))
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements['x'] = xyz[:, 0]; elements['y'] = xyz[:, 1]; elements['z'] = xyz[:, 2]
        elements['nx'] = normals[:, 0]; elements['ny'] = normals[:, 1]; elements['nz'] = normals[:, 2]
        elements['red'] = rgb[:, 0]; elements['green'] = rgb[:, 1]; elements['blue'] = rgb[:, 2]
        elements['f_dc_0'] = f_dc[:, 0]; elements['f_dc_1'] = f_dc[:, 1]; elements['f_dc_2'] = f_dc[:, 2]
        for i in range(f_rest.shape[1]): elements[f'f_rest_{i}'] = f_rest[:, i]
        elements['opacity'] = opacities[:, 0]
        for i in range(scale.shape[1]): elements[f'scale_{i}'] = scale[:, i]
        for i in range(rotation.shape[1]): elements[f'rot_{i}'] = rotation[:, i]
        PlyData([PlyElement.describe(elements, 'vertex')]).write(path)
        print(f"[Model] Saved {path} with {xyz.shape[0]} points.")

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