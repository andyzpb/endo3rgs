import open3d as o3d
import numpy as np
import json
import os
from plyfile import PlyData
from sklearn.neighbors import NearestNeighbors

# ---------------- configs ----------------
# Input data paths
DATA_DIR = "data/endo_stable_sp" 
PLY_FILE = os.path.join(DATA_DIR, "init_points.ply")
CAM_FILE = os.path.join(DATA_DIR, "cameras.json")

# output paths
NPY_FILE = "output/init_data.npy"
OUTPUT_MESH = "output/mesh_from_init_points.ply"
# -------------------------------------

def run_reconstruction():
    print(f"[1/5] Loading Data...")
    
    # 1. read cameras 
    if not os.path.exists(CAM_FILE):
        print(f"Error: {CAM_FILE} not found!")
        return

    with open(CAM_FILE, 'r') as f:
        cam_data = json.load(f)
    
    camera_centers = []
    # make sure to sort keys by numeric order to maintain consistency with point cloud ordering (if needed)
    sorted_keys = sorted(cam_data.keys(), key=lambda x: int(x))
    for k in sorted_keys:
        v = cam_data[k]
        c2w = np.array(v['c2w'])
        center = c2w[:3, 3]
        camera_centers.append(center)
    
    camera_centers = np.array(camera_centers)
    print(f"   - Loaded {len(camera_centers)} camera positions.")

    # 2. read point cloud
    if not os.path.exists(PLY_FILE):
        print(f"Error: {PLY_FILE} not found!")
        return

    plydata = PlyData.read(PLY_FILE)
    vertex = plydata['vertex']
    
    pts = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
    
    # try to load colors, if not present, use default gray
    if 'red' in vertex:
        colors = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=1) / 255.0
    else:
        colors = np.ones_like(pts) * 0.5
        
    print(f"   - Loaded {len(pts)} points from init_points.ply.")

    # 3. data backup (optional, but useful for debugging and future use)
    print(f"[2/5] Saving to {NPY_FILE}...")
    save_dict = {
        "points": pts,
        "colors": colors,
        "camera_centers": camera_centers
    }
    np.save(NPY_FILE, save_dict)
    
    # -------------------------------------------------
    # core reconstruction pipeline
    # -------------------------------------------------
    print(f"[3/5] Estimating & Orienting Normals...")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # A. basic normal estimation (KNN)
    # radius=5.0 or 10.0
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30))
    
    # B. use camera centers to orient normals consistently
    # 1. find nearest camera center for each point
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(camera_centers)
    distances, indices = nbrs.kneighbors(pts)
    nearest_cam_centers = camera_centers[indices.flatten()]
    
    # 2. compute directions from points to their nearest camera centers
    directions = nearest_cam_centers - pts
    
    # 3. use numpy to compute dot product and flip normals that are facing away from cameras
    normals = np.asarray(pcd.normals)
    
    # dot product between normals and directions
    # dot product > 0 means normal is facing towards the camera, < 0 means facing away
    dot_products = np.sum(normals * directions, axis=1)
    
    flip_mask = dot_products < 0
    normals[flip_mask] *= -1
    
    # update normals in point cloud
    pcd.normals = o3d.utility.Vector3dVector(normals)
    print(f"   - Flipped {np.sum(flip_mask)} normals to face cameras.")

    print(f"[4/5] Poisson Reconstruction...")
    # depth=9 
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=7, width=0, scale=1.1, linear_fit=False
    )
    
    print(f"[5/5] Post-processing & Saving...")
    # C. remove low-density vertices (these are likely to be noise or outliers)
    densities = np.asarray(densities)
    # cutoff the lowest 2% density vertices (this is a common heuristic, can be tuned)
    density_threshold = np.quantile(densities, 0.02) 
    mesh.remove_vertices_by_mask(densities < density_threshold)
    
    # final cleanup: remove degenerate triangles and compute vertex normals for better visualization
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    
    o3d.io.write_triangle_mesh(OUTPUT_MESH, mesh)
    print(f"[Success] Mesh saved to {OUTPUT_MESH}")

if __name__ == "__main__":
    run_reconstruction()