"""
代码说明：
用于训练 HelmHoltz (HH) 任务的 Stable-SDF 模型。
采用 Clamped L1 Loss 稳定边界训练，并同步优化形状的 Latent Codes。
通过 CONFIG 字典管理超参数，输入输出路径自动对接 GANO 仓库结构。
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 将项目的根目录加入 sys.path，以便跨文件夹导入 src 下的模型
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.stablesdf.deepsdf import DeepSDFWithPE

# ==========================================
# 统一参数配置中心
# ==========================================
CONFIG = {
    # --- 路径配置 (相对当前脚本) ---
    "data_path": "../../data/hh/scattering_sdf_dataset_mixed.npz",
    "ckpt_dir": "../../checkpoints/hh/stablesdf",
    "output_dir": "../../output/hh/stablesdf_vis",
    
    # --- 模型结构配置 ---
    "latent_dim": 64,
    "hidden_dim": 256,
    "num_layers": 4,
    "num_freqs": 6,
    
    # --- 训练超参数 ---
    "batch_size": 128,
    "epochs": 1000,
    "lr": 1e-4,
    "clamp_dist": 0.05,       # SDF 截断阈值
    "reg_weight": 1e-4,       # Latent 正则化权重
    
    # --- 运行控制 ---
    "save_interval": 20,      # 保存与可视化的间隔 Epoch
    "num_workers": 4,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ==========================================
# 数据集定义
# ==========================================
class SDFDataset(Dataset):
    def __init__(self, npz_path):
        print(f"[*] 加载 SDF 数据集: {npz_path}...")
        data = np.load(npz_path)
        self.points = torch.from_numpy(data['points']).float()  
        self.sdfs = torch.from_numpy(data['sdfs']).float()      
        self.num_samples = self.points.shape[0]
        print(f"[*] 数据加载完成。共 {self.num_samples} 个形状样本。")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return idx, self.points[idx], self.sdfs[idx].unsqueeze(-1)

# ==========================================
# 可视化评估
# ==========================================
def visualize_reconstruction(model, latents, idx, epoch, device, output_dir):
    """评估模型：可视化重建网格"""
    model.eval()
    resolution = 256
    
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    grid_x, grid_y = np.meshgrid(x, y)
    grid_points = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
    grid_points = torch.from_numpy(grid_points).float().to(device)
    
    with torch.no_grad():
        z = latents(torch.tensor([idx]).to(device))
        z_expanded = z.unsqueeze(1).expand(-1, grid_points.shape[0], -1)
        points_input = grid_points.unsqueeze(0)
        
        pred_sdf = model(points_input, z_expanded)
        pred_sdf = pred_sdf.view(resolution, resolution).cpu().numpy()
    
    plt.figure(figsize=(6, 5))
    plt.imshow(pred_sdf, origin='lower', extent=[-1, 1, -1, 1], cmap='seismic', vmin=-0.1, vmax=0.1)
    plt.colorbar(label='Predicted SDF')
    plt.contour(grid_x, grid_y, pred_sdf, levels=[0], colors='black', linewidths=2)
    plt.title(f"Recon Epoch {epoch} (Clamped)")
    plt.tight_layout()
    
    vis_name = f"recon_epoch_{epoch:04d}.png"
    plt.savefig(os.path.join(output_dir, vis_name))
    plt.close()

# ==========================================
# 训练主循环
# ==========================================
def main():
    # 1. 目录准备
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.normpath(os.path.join(current_dir, CONFIG["data_path"]))
    ckpt_dir = os.path.normpath(os.path.join(current_dir, CONFIG["ckpt_dir"]))
    output_dir = os.path.normpath(os.path.join(current_dir, CONFIG["output_dir"]))
    
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"[!] 错误: 找不到数据集: {data_path}")
        return

    # 2. 数据与设备
    device = torch.device(CONFIG["device"])
    print(f"[*] 训练设备: {device}")
    
    dataset = SDFDataset(data_path)
    dataloader = DataLoader(
        dataset, batch_size=CONFIG["batch_size"], 
        shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True
    )
    
    # 3. 初始化模型与 Latent Codes
    model = DeepSDFWithPE(
        latent_dim=CONFIG["latent_dim"], 
        hidden_dim=CONFIG["hidden_dim"], 
        num_layers=CONFIG["num_layers"], 
        num_freqs=CONFIG["num_freqs"]
    ).to(device)
    
    latents = nn.Embedding(len(dataset), CONFIG["latent_dim"]).to(device)
    torch.nn.init.normal_(latents.weight.data, 0.0, 0.01)
    
    # 4. 优化器与损失函数
    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": CONFIG["lr"]},
        {"params": latents.parameters(), "lr": CONFIG["lr"]},
    ])
    loss_fn = nn.L1Loss() 
    clamp_dist = CONFIG["clamp_dist"]
    
    print("[*] 开始训练 Stable-SDF...")
    history = []
    
    # 5. 迭代训练
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}", leave=False)
        for indices, points, gt_sdfs in pbar:
            indices = indices.to(device)
            points = points.to(device)
            gt_sdfs = gt_sdfs.to(device)
            
            z_batch = latents(indices)
            z_expanded = z_batch.unsqueeze(1).expand(-1, points.shape[1], -1)
            
            pred_sdfs = model(points, z_expanded)
            
            # 截断 SDF 以平滑边界训练
            pred_sdfs_clamped = torch.clamp(pred_sdfs, -clamp_dist, clamp_dist)
            gt_sdfs_clamped = torch.clamp(gt_sdfs, -clamp_dist, clamp_dist)
            
            sdf_loss = loss_fn(pred_sdfs_clamped, gt_sdfs_clamped)
            reg_loss = torch.mean(torch.norm(z_batch, dim=1)) * CONFIG["reg_weight"]
            
            loss = sdf_loss + reg_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(dataloader)
        history.append(avg_loss)
        
        # 6. 保存与可视化验证
        if (epoch + 1) % CONFIG["save_interval"] == 0:
            print(f"Epoch {epoch+1:04d} | Avg Loss: {avg_loss:.6f}")
            
            save_name = os.path.join(ckpt_dir, f"deepsdf_epoch_{epoch+1:04d}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'latents': latents.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss
            }, save_name)
            
            visualize_reconstruction(model, latents, 0, epoch+1, device, output_dir)

    # 7. 收尾：保存最终模型与损失曲线
    final_path = os.path.join(ckpt_dir, "deepsdf_final.pth")
    torch.save({
        'model': model.state_dict(),
        'latents': latents.state_dict(),
        'config': CONFIG
    }, final_path)
    
    plt.figure()
    plt.plot(history)
    plt.yscale('log')
    plt.title("DeepSDF Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(output_dir, "training_loss_curve.png"))
    print(f"[*] 训练结束。所有检查点和图表已存入 {ckpt_dir} 和 {output_dir}")

if __name__ == "__main__":
    main()