import torch
import os
import argparse
from tqdm import tqdm
from data_loader import load_scene_data
from gaussian_model import GaussianModel
from renderer import render
import torchvision

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def train(args):
    dataset_path = args.dataset_path
    
    # 1. 准备高斯模型
    gaussians = GaussianModel()
    # 关键：加载 Stage 1 的点云
    ply_path = os.path.join(dataset_path, "points3d_init.ply")
    gaussians.create_from_pcd(ply_path)
    gaussians.setup_training()

    # 2. 准备数据
    cameras = load_scene_data(dataset_path)
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    # 3. 训练循环
    iter_bar = tqdm(range(args.iterations))
    
    for iteration in iter_bar:
        # 随机选一张图
        cam_idx = torch.randint(0, len(cameras), (1,)).item()
        viewpoint_cam = cameras[cam_idx]

        # 渲染
        render_pkg = render(viewpoint_cam, gaussians, bg_color)
        image = render_pkg["render"]

        # 计算 Loss
        gt_image = viewpoint_cam.image
        loss = l1_loss(image, gt_image)
        
        # 简单打印
        if iteration % 100 == 0:
            iter_bar.set_description(f"Loss: {loss.item():.4f}")
            
        # 简单的保存中间结果
        if iteration % 1000 == 0:
            torchvision.utils.save_image(image, f"debug_render_{iteration}.png")

        # 反向传播
        loss.backward()

        # 优化器 Step
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)
        
        # 注意：这里省略了 Densification (Clone/Split) 逻辑
        # 如果是 Endo3R 初始化，因为点云已经很密了，暂时不需要复杂的 Split 也能跑出不错的效果

    print("Training done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Stage 1 output dir")
    parser.add_argument("--iterations", type=int, default=5000)
    args = parser.parse_args()
    
    train(args)