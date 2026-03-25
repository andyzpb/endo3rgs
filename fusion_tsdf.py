import open3d as o3d
import numpy as np
import glob
import os
import json
import cv2

# Configuration
DATA_PATH = "output/rgbd_data"
CALIB_JSON = "calibration_result.json" 

def run_fusion():
    print("[TSDF] Initializing Fusion...")
    
    # 1. Validate Data
    if not os.path.exists(CALIB_JSON):
        print(f"Error: {CALIB_JSON} not found.")
        return
        
    color_files = sorted(glob.glob(os.path.join(DATA_PATH, "color", "*.jpg")))
    depth_files = sorted(glob.glob(os.path.join(DATA_PATH, "depth", "*.png")))
    pose_files = sorted(glob.glob(os.path.join(DATA_PATH, "*.pose")))
    
    if len(depth_files) == 0:
        print(f"Error: No depth files found in {DATA_PATH}/depth. Run export_rgbd.py first.")
        return

    # 2. Auto-detect Resolution from first depth map
    first_depth = cv2.imread(depth_files[0], cv2.IMREAD_UNCHANGED)
    real_h, real_w = first_depth.shape
    print(f"[TSDF] Detected resolution: {real_w}x{real_h}")

    # 3. Setup Intrinsics
    with open(CALIB_JSON, 'r') as f:
        calib = json.load(f)
    
    # Calculate scale factor relative to original calibration
    json_w = calib["image_size_after_crop"]["width"]
    scale = real_w / float(json_w)
    
    fx = calib["K"][0][0] * scale
    fy = calib["K"][1][1] * scale
    cx = calib["K"][0][2] * scale
    cy = calib["K"][1][2] * scale
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(real_w, real_h, fx, fy, cx, cy)
    
    # 4. Initialize TSDF Volume
    # Voxel size in meters (0.001 = 1mm). Smaller = higher detail but more RAM.
    voxel_size = 0.001 
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=voxel_size * 5, # Truncation margin (surface thickness)
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    n_frames = len(depth_files)
    print(f"[TSDF] Integrating {n_frames} frames...")
    
    for i in range(n_frames):
        # Optimization: Skip frames if you have too many (e.g., every 5th frame)
        if i % 1 != 0: continue 
        
        print(f"\rProcessing frame {i+1}/{n_frames}", end="")
        
        # Read Data
        color = o3d.io.read_image(color_files[i])
        depth = o3d.io.read_image(depth_files[i])
        w2c = np.loadtxt(pose_files[i])
        
        # Create RGBD Image
        # depth_scale=1000.0 converts our micrometers back to millimeters for Open3D? 
        # Actually Open3D standard unit is usually meters. 
        # If depth is in micrometers (uint16), dividing by 1000.0 gives millimeters.
        # If we want meters, we should divide by 1000000.0?
        # Standard convention: input uint16 is mm, scale=1000.0 -> meters.
        # OUR input is um (mm*1000). So input 50000 = 50mm. 
        # To get meters (0.05), we need to divide 50000 by 1,000,000.
        # However, typically simple mm works best for surgery scale.
        # Let's stick to standard Open3D: Depth uint16 usually represents mm.
        # In export_rgbd.py, we did: depth_mm * 1000. So we stored MICROMETERS.
        # So 1mm = 1000 value.
        # To get Meters: 1000 / X = 0.001 -> X = 1,000,000.
        # But Open3D ScalableTSDF often works better if units are consistent.
        # Let's try standard scale 1000.0. This treats our input as if 1000 units = 1 meter.
        # So our "1mm" (value 1000) becomes 1 meter in Open3D space.
        # This is fine, just remember your output mesh will be scaled up by 1000x relative to meters 
        # (or 1.0 unit = 1mm).
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, 
            depth_scale=1000.0, # Value 1000 -> 1.0 unit (mm)
            depth_trunc=300000.0, # Truncate after 300 units (30cm)
            convert_rgb_to_intensity=False
        )
        
        # Integrate
        # Open3D integrate expects extrinsic (World-to-Camera)
        volume.integrate(rgbd, intrinsic, w2c)
        
    print("\n[TSDF] Extracting Mesh...")
    mesh = volume.extract_triangle_mesh()
    
    # Post-processing
    mesh.compute_vertex_normals()
    mesh.remove_unreferenced_vertices()
    
    # Save
    output_mesh_path = "output/reconstructed_mesh.ply"
    o3d.io.write_triangle_mesh(output_mesh_path, mesh)
    print(f"[Success] Mesh saved to {output_mesh_path}")

if __name__ == "__main__":
    run_fusion()