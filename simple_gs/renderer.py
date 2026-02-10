import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def render(viewpoint_camera, pc, bg_color):
    """
    viewpoint_camera: data_loader.Camera 对象
    pc: gaussian_model.GaussianModel 对象
    """
    
    # 1. 准备屏幕空间坐标
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # 2. 设置光栅化参数
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image.shape[1]),
        image_width=int(viewpoint_camera.image.shape[2]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=torch.transpose(torch.zeros((4,4)), 0, 1).cuda(), # 暂时占位，实际应该用 camera.world_view_transform
        projmatrix=torch.transpose(torch.zeros((4,4)), 0, 1).cuda(), # 暂时占位
        sh_degree=3,
        campos=viewpoint_camera.T,
        prefiltered=False,
        debug=False
    )
    
    # *注意*：这里简化了 ViewMatrix 和 ProjMatrix 的计算。
    # 在正式代码中，需要根据 R 和 T 构建 4x4 矩阵。
    # 为了跑通，这里需要一个简单的 helper function 构建矩阵，我这里略过以保持简洁，
    # 实际运行时 diff-gaussian-rasterization 需要正确的 viewmatrix (W2C) 和 projmatrix (W2C * Projection)。
    
    # 修正：简单构建一下矩阵
    w2c = torch.eye(4, device="cuda")
    w2c[:3, :3] = viewpoint_camera.R
    w2c[:3, 3] = viewpoint_camera.T
    # 需要转置因为 rasterizer 期望行优先或列优先的具体格式（通常是转置的）
    viewmatrix = w2c.transpose(0, 1) 
    
    # 简单的投影矩阵构建 (假设无 skew)
    zfar = 100.0
    znear = 0.01
    proj = torch.zeros((4, 4), device="cuda")
    proj[0, 0] = 1.0 / tanfovx
    proj[1, 1] = 1.0 / tanfovy
    proj[2, 2] = zfar / (zfar - znear)
    proj[2, 3] = -(zfar * znear) / (zfar - znear)
    proj[3, 2] = 1.0
    
    full_proj = (w2c.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0).transpose(0, 1)
    
    raster_settings.viewmatrix = viewmatrix
    raster_settings.projmatrix = full_proj

    # 3. 实例化光栅化器并渲染
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, radii = rasterizer(
        means3D = pc.get_xyz,
        means2D = screenspace_points,
        shs = pc.get_features,
        opacities = pc.get_opacity,
        scales = pc.get_scaling,
        rotations = pc.get_rotation,
        cov3D_precomp = None
    )

    return {"render": rendered_image, "viewspace_points": screenspace_points}