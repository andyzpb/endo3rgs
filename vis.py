import os
import argparse
import numpy as np
from dust3r.utils.viser_utils import PointCloudViewer

def get_args_parser():
    parser = argparse.ArgumentParser('Endo3R visualizer', add_help=False)
    parser.add_argument('--recon_path', type=str, default='./output/demo/', help='Path to experiment folder')
    parser.add_argument('--vis_threshold', type=float, default=1.5, help='visualize')
    return parser

parser = get_args_parser()
args = parser.parse_args()
data = np.load(os.path.join(args.recon_path, "result.npy"), allow_pickle=True).item()

pts3ds_to_vis  = data['pts_all']
colors_to_vis = data['images_all']
depth = data['depths_all']

conf = data['conf_all']
poses = data['poses_all']
edge_colors = [None] * len(pts3ds_to_vis)
intrinsic = data['intrinsic']

cam_dict = {"pose": poses,
            "intrinsic": intrinsic}

viewer = PointCloudViewer(
    pts3ds_to_vis,
    colors_to_vis,
    conf,
    cam_dict,
    device="cuda",
    edge_color_list=edge_colors,
    show_camera=True,
    vis_threshold=args.vis_threshold,
    size = 512
)
viewer.run()

