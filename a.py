import numpy as np
import json
import os
import shutil
from plyfile import PlyData, PlyElement

# Configuration
INPUT_DIR = "data/endo_gs_processed"
OUTPUT_DIR = "data/endo_gs_normalized"

def normalize_scene():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    print(f"[1/3] Loading original data from {INPUT_DIR}...")
    
    # 1. Load cameras
    with open(os.path.join(INPUT_DIR, "cameras.json"), 'r') as f:
        cameras = json.load(f)
    
    # 2. Load point cloud
    plydata = PlyData.read(os.path.join(INPUT_DIR, "init_points.ply"))
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    points = np.stack([x, y, z], axis=1)
    colors = np.stack([plydata['vertex']['red'], plydata['vertex']['green'], plydata['vertex']['blue']], axis=1)
    
    # 3. Calculate bounding box and scale center
    # Strategy: center on camera trajectory, not on point cloud (which may have floating points)
    cam_centers = []
    for k, v in cameras.items():
        c2w = np.array(v['c2w'])
        cam_centers.append(c2w[:3, 3])
    cam_centers = np.array(cam_centers)
    
    center = np.mean(cam_centers, axis=0)
    # Calculate radius: keep all cameras within a sphere of radius 4.0 (leave room for Gaussian growth)
    dist = np.linalg.norm(cam_centers - center, axis=1)
    max_dist = np.max(dist)
    scale_factor = 4.0 / (max_dist + 1e-6)
    
    print(f"   - Original Center: {center}")
    print(f"   - Scale Factor: {scale_factor:.4f}")
    
    # 4. Apply transformations and save
    print("[2/3] Transforming and Saving...")
    
    # Transform point cloud
    points_norm = (points - center) * scale_factor
    
    # Save new PLY
    vertex = np.array([(p[0], p[1], p[2], c[0], c[1], c[2]) for p, c in zip(points_norm, colors)],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(os.path.join(OUTPUT_DIR, "init_points.ply"))
    
    # Transform cameras
    new_cameras = {}
    for k, v in cameras.items():
        c2w = np.array(v['c2w'])
        # Translation transformation: (t - center) * scale
        c2w[:3, 3] = (c2w[:3, 3] - center) * scale_factor
        v['c2w'] = c2w.tolist()
        new_cameras[k] = v
        
    with open(os.path.join(OUTPUT_DIR, "cameras.json"), 'w') as f:
        json.dump(new_cameras, f, indent=4)
        
    # Copy images and confidence maps
    shutil.copytree(os.path.join(INPUT_DIR, "images"), os.path.join(OUTPUT_DIR, "images"))
    if os.path.exists(os.path.join(INPUT_DIR, "confidence_maps")):
        shutil.copytree(os.path.join(INPUT_DIR, "confidence_maps"), os.path.join(OUTPUT_DIR, "confidence_maps"))
        
    # Save transform parameters for later mesh reversal if needed
    transform_info = {"center": center.tolist(), "scale": scale_factor}
    with open(os.path.join(OUTPUT_DIR, "transform.json"), 'w') as f:
        json.dump(transform_info, f)

    print(f"[3/3] Done! Use '{OUTPUT_DIR}' for training.")

if __name__ == "__main__":
    normalize_scene()