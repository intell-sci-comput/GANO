"""
2D 机翼任务：Stable-SDF 训练脚本
说明：本版本为彻底重写版，遵循 GANO 仓库规范。
直接加载预处理好的 SDF Tensor，在 GPU 上进行极速训练，同时优化网络权重和 Latent Codes。
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 将 GANO 仓库根目录加入系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.airfoil.model import DeepSDFWithPE

# ==========================================
# ======= [全局参数配置字典] ================
# ==========================================
CONFIG = {
    # --- 路径配置 ---
    "DATA_PATH": os.path.join(project_root, "data", "airfoil", "airfoil_sdf_train.pt"),
    "SAVE_DIR": os.path.join(project_root, "checkpoints", "airfoil_stablesdf"),

    # --- 模型超参数 ---
    "LATENT_SIZE": 64,       # 隐向量维度 (须与后续物理场要求对齐)
    "HIDDEN_DIM": 256,
    "NUM_LAYERS": 4,
    "NUM_FREQS": 6,

    # --- 训练超参数 ---
    "BATCH_SIZE": 128,       # 这里的 Batch 是指 "同时训练的机翼数量"
    "NUM_EPOCHS": 1000,
    "LR": 5e-4,
    "LATENT_REG": 1e-4,      # 对 Latent Code 的 L2 正则化权重

    # --- 日志与保存频率 ---
    "LOG_EVERY": 10,
    "SAVE_EVERY": 100,
}

os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载数据
    print(f"Loading data from {CONFIG['DATA_PATH']}...")
    if not os.path.exists(CONFIG['DATA_PATH']):
        print(f"Error: 找不到数据文件 {CONFIG['DATA_PATH']}。请先运行预处理脚本。")
        return

    data = torch.load(CONFIG['DATA_PATH'], map_location='cpu')
    coords_all = data['coords']  # (N, 4096, 2)
    sdfs_all = data['sdfs']      # (N, 4096, 1)
    source_indices = data.get('indices', torch.arange(coords_all.shape[0], dtype=torch.long))

    num_shapes = coords_all.shape[0]
    num_points = coords_all.shape[1]
    print(f"Loaded {num_shapes} shapes, each with {num_points} points.")

    # 为了极速训练，直接将所有数据推入 GPU
    coords_all = coords_all.to(device)
    sdfs_all = sdfs_all.to(device)
    shape_indices = torch.arange(num_shapes, device=device)

    dataset = TensorDataset(shape_indices, coords_all, sdfs_all)
    dataloader = DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)

    # 2. 初始化模型与 Latent Codes
    model = DeepSDFWithPE(
        latent_dim=CONFIG["LATENT_SIZE"],
        hidden_dim=CONFIG["HIDDEN_DIM"],
        num_layers=CONFIG["NUM_LAYERS"],
        num_freqs=CONFIG["NUM_FREQS"]
    ).to(device)

    # DeepSDF 特色：Latent Code 作为可学习的 Embedding 矩阵
    latents = nn.Embedding(num_shapes, CONFIG["LATENT_SIZE"]).to(device)
    torch.nn.init.normal_(latents.weight, mean=0.0, std=0.01)

    # 优化器同时更新网络权重和 Latent 权重
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': CONFIG["LR"]},
        {'params': latents.parameters(), 'lr': CONFIG["LR"]}
    ])

    criterion = nn.L1Loss()

    # 3. 训练循环
    print(f"Starting training for {CONFIG['NUM_EPOCHS']} epochs...")
    start_time = time.time()
    pbar = tqdm(range(1, CONFIG["NUM_EPOCHS"] + 1), desc="Training")

    for epoch in pbar:
        model.train()
        total_loss = 0.0

        for batch_indices, batch_coords, batch_sdfs in dataloader:
            optimizer.zero_grad()

            # 获取当前 batch 的 latent codes
            batch_latents = latents(batch_indices) # (B, latent_size)
            # 扩展 latent 到每个点上: (B, 1, latent_size) -> (B, 4096, latent_size)
            batch_latents_expanded = batch_latents.unsqueeze(1).expand(-1, num_points, -1)

            # 前向传播
            pred_sdfs = model(batch_coords, batch_latents_expanded) # (B, 4096, 1)

            # 计算 Loss (SDF L1 Loss + Latent 正则化)
            loss_sdf = criterion(pred_sdfs, batch_sdfs)
            loss_reg = CONFIG["LATENT_REG"] * torch.mean(batch_latents ** 2)
            loss = loss_sdf + loss_reg

            # 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        pbar.set_postfix({"Loss": f"{avg_loss:.6f}"})

        # 日志与保存
        if epoch % CONFIG["LOG_EVERY"] == 0:
            elapsed = time.time() - start_time
            tqdm.write(f"Epoch {epoch}/{CONFIG['NUM_EPOCHS']} | Loss: {avg_loss:.6f} | Time: {elapsed:.1f}s")
            
            torch.save(model.state_dict(), os.path.join(CONFIG["SAVE_DIR"], 'model_latest.pth'))
            # 注意：保存格式必须有 'weight' 键，以便后续的 dataset.py 能无缝读取
            torch.save({
                'weight': latents.weight.detach().cpu(),
                'indices': source_indices.detach().cpu()
            }, os.path.join(CONFIG["SAVE_DIR"], 'latents_latest.pth'))

        if epoch % CONFIG["SAVE_EVERY"] == 0:
            torch.save(model.state_dict(), os.path.join(CONFIG["SAVE_DIR"], f'model_{epoch}.pth'))

    print("Training complete!")

if __name__ == "__main__":
    main()
