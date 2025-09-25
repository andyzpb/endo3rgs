import os
import glob
import numpy as np
import cv2
import tqdm
import sys
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import argparse

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def depth_evaluation(gt_depths, pred_depths, savedir=None, pred_masks=None, min_depth=0.0001, max_depth=100):
    assert gt_depths.shape[0] == pred_depths.shape[0]

    gt_depths_valid = []
    pred_depths_valid = []
    errors = []
    num = gt_depths.shape[0]
    for i in range(num):
        gt_depth = gt_depths[i]
        mask = (gt_depth > min_depth) * (gt_depth < max_depth)
        gt_height, gt_width = gt_depth.shape[:2]

        pred_depth = pred_depths[i]
        if pred_masks is not None:
            pred_mask = pred_masks[i]
            pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt_width, gt_height)) > 0.5
            mask = mask * pred_mask

        if mask.sum() == 0:
            continue

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        
        pred_depths_valid.append(pred_depth)
        gt_depths_valid.append(gt_depth)

    ratio = np.median(np.concatenate(gt_depths_valid)) / \
                np.median(np.concatenate(pred_depths_valid))

    for i in range(len(pred_depths_valid)):
        gt_depth = gt_depths_valid[i]
        pred_depth = pred_depths_valid[i]

        pred_depth *= ratio
        pred_depth[pred_depth < min_depth] = min_depth
        pred_depth[pred_depth > max_depth] = max_depth

        errors.append(compute_errors(gt_depth, pred_depth))

    mean_errors = np.array(errors).mean(0)
    
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")
    return mean_errors


def abs_rel(gt, pred):
    abs_rel = (np.abs(gt - pred) / gt).mean()
    return abs_rel

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def __getattr__(self, attr):
        return getattr(self.terminal, attr)
    def write(self, message):
        self.log.write(message)     
    def flush(self):
        self.log.flush()
        
        
def eval_single_scene(gt_dir, pred_dir, test_tae=False):
    gt_depth_seq = sorted(glob.glob(gt_dir + "*"))
    pred_depth_seq = sorted(glob.glob(pred_dir + "*.npy"))
    gt_depths = []
    pred_depths = []
    for i in tqdm.tqdm(range(len(gt_depth_seq))):
        pred_depth = np.load(pred_depth_seq[i])
        h, w = pred_depth.shape
        gt_depth = cv2.imread(gt_depth_seq[i], cv2.IMREAD_UNCHANGED)
        gt_depth = gt_depth.astype(np.float32) / 1000
        gt_depth = cv2.resize(gt_depth, (320, 256), interpolation=cv2.INTER_NEAREST)
        pred_depth = cv2.resize(pred_depth, (320, 256), interpolation=cv2.INTER_NEAREST)
        gt_depths.append(gt_depth)
        pred_depths.append(pred_depth)
        
    gt_depths = np.stack(gt_depths)
    pred_depths = np.stack(pred_depths)
    return depth_evaluation(gt_depths, pred_depths, savedir = pred_dir), len(gt_depth_seq)

def get_args_parser():
    parser = argparse.ArgumentParser('Endo3R Depth Evaluation', add_help=False)
    parser.add_argument('--data_root', type=str, help='Path to experiment folder')
    parser.add_argument('--data_type', type=str, help='Path to experiment folder')
    parser.add_argument('--output_path', type=str, help='ckpt path')
    return parser

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    print("Saving evaluation results to: ", f"{args.output_path}/evaluation.txt")
    sys.stdout = Logger(f"{args.output_path}/evaluation.txt")

    root = args.data_root
    datasets_dir = sorted(glob.glob(root + "*"))
    all_depth_errors = []
    all_length = 0
    eval_single = False
    if eval_single:
        key = root + 'dataset_8/keyframe_3/'
        gt_depth_path = os.path.join(key, "data/depthmap_rectified/")
        pred_dir = f"{args.output_path}/dataset_8_keyframe_3/depth/"
        print(f"eval pred dir: {pred_dir}, gt dir: {gt_depth_path}")
        depth_errors, length = eval_single_scene(gt_depth_path, pred_dir)
    else:
        if args.data_type == "scared":
            for dir in datasets_dir:
                keyframes_dir = sorted(glob.glob(dir + "/*"))
                for key in keyframes_dir:
                    gt_depth_path = os.path.join(key, "data/depthmap_rectified/")
                    pred_dir = os.path.join(args.output_path, key.split('/')[-2] + '_' + key.split('/')[-1] + '/depth/')
                    depth_errors, length = eval_single_scene(gt_depth_path, pred_dir)
                    all_depth_errors.append(depth_errors * length)
                    all_length += length
        else:
            for dir in datasets_dir:
                gt_depth_path = os.path.join(dir, "depth_cropped01/")
                pred_dir = os.path.join(args.output_path, dir.split('/')[-1] + '/depth/')
                depth_errors, length = eval_single_scene(gt_depth_path, pred_dir)
                all_depth_errors.append(depth_errors * length)
                all_length += length
                

        all_depth_errors = np.stack(all_depth_errors).sum(axis=0) / all_length
        result_depth = 'abs_rel: {0}, sq_rel: {1}, rmse: {2}, rmse_log: {3}, a1: {4}, a2: {5}, a3: {6}'.format(
                    all_depth_errors[0], all_depth_errors[1], all_depth_errors[2],
                    all_depth_errors[3], all_depth_errors[4], all_depth_errors[5],
                    all_depth_errors[6])
        print(result_depth)
