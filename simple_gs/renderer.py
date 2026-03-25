import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def render(viewpoint_camera, pc, bg_color, scaling_modifier=1.0, depth_downsample_factor=2):
    """
    Render the scene. 
    depth_downsample_factor: Downsampling factor when rendering depth (2 means 1/2 resolution, saves VRAM)
    """
    
    # 1. Set rasterization parameters (RGB Pass - Full Resolution)
    tanfovx = math.tan(viewpoint_camera.fx / viewpoint_camera.width * 2 * math.atan(0.5 * viewpoint_camera.width / viewpoint_camera.fx))
    tanfovy = math.tan(viewpoint_camera.fy / viewpoint_camera.height * 2 * math.atan(0.5 * viewpoint_camera.height / viewpoint_camera.fy))

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.height),
        image_width=int(viewpoint_camera.width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.w2c.transpose(0, 1),
        projmatrix=viewpoint_camera.w2c.transpose(0, 1) @ torch.tensor([
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
        ], device="cuda").float(),
        sh_degree=3,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        means2D.retain_grad()
    except:
        pass

    # ===============================================
    # Pass 1: Render RGB image (Full Res)
    # ===============================================
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=pc._features_dc,
        colors_precomp=None, # Use SH for RGB Pass
        opacities=pc.get_opacity,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
        cov3D_precomp=None
    )

    # ===============================================
    # Pass 2: Render depth map (Low Res - VRAM Saver)
    # ===============================================
    raster_settings_depth = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.height // depth_downsample_factor), # Downsample
        image_width=int(viewpoint_camera.width // depth_downsample_factor),   # Downsample
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.zeros(3, device="cuda"), 
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.w2c.transpose(0, 1),
        projmatrix=viewpoint_camera.w2c.transpose(0, 1) @ torch.tensor([
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
        ], device="cuda").float(),
        sh_degree=3,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )
    
    rasterizer_depth = GaussianRasterizer(raster_settings=raster_settings_depth)

    # 1. Coordinate transformation
    w2c = viewpoint_camera.w2c
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    cam_points = R @ means3D.transpose(0, 1) + t.unsqueeze(1)
    
    # 2. Extract depth
    depths = cam_points[2, :].unsqueeze(1)
    
    # 3. Render (input to color)
    depth_as_color = depths.repeat(1, 3)

    # [Critical Fix] 
    # 1. Create new means2D_depth to prevent gradient conflict (previous RuntimeError was largely due to reusing means2D)
    means2D_depth = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try: means2D_depth.retain_grad()
    except: pass
    
    # 2. Strictly adhere to mutual exclusivity: shs=None, colors_precomp=depth_as_color
    rendered_depth, _ = rasterizer_depth(
        means3D=means3D,
        means2D=means2D_depth, # Use new means2D variable
        shs=None,              # Must be None!
        colors_precomp=depth_as_color, # Use depth color
        opacities=pc.get_opacity,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
        cov3D_precomp=None
    )
    
    # 4. Extract single channel [1, H/2, W/2]
    rendered_depth = rendered_depth[0:1, :, :]

    return {
        "render": rendered_image,
        "render_depth": rendered_depth,
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        "radii": radii
    }