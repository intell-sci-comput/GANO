"""
代码说明：
用于训练 HelmHoltz (HH) 任务的 GI-Transolver 模型 (前向散射场预测)。
基于 OneCycleLR 策略，支持从 Stable-SDF 加载预训练的 Latent Codes 作为条件输入。
"""

import os
import sys
import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# 将项目根目录加入 sys.path，导入 GANO 统一组件
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.gi_transolver.model import Model as TransolverModel

# A100 TF32 加速
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ==========================================
# 统一参数配置中心
# ==========================================
CONFIG = {
    # --- 路径配置 (相对当前脚本) ---
    "norm_data": "../../data/hh/scattering_dataset_normalized.npz", 
    "deepsdf_ckpt": "../../checkpoints/hh/stablesdf/deepsdf_final.pth",
    "ckpt_dir": "../../checkpoints/hh/transolver",
    "output_dir": "../../output/hh/transolver_vis",
    
    # --- 训练超参数 ---
    "batch_size": 32,
    "epochs": 200,
    "max_lr": 5e-4,
    "pct_start": 0.01,
    "div_factor": 2,
    "final_div_factor": 10,
    "weight_decay": 1e-5,
    "clip_grad": 1.0,
    "samples_per_item": 4096,
    
    # --- GI-Transolver 模型配置 ---
    "z_dim": 64,
    "space_dim": 2,
    "out_dim": 2,
    "n_hidden": 256,
    "n_layers": 4,
    "n_head": 8,
    "dropout": 0.0,
    "mlp_ratio": 1,
    "slice_num": 32,
    "unified_pos": False,
    "ref": 8,
    "use_theta_in_coord": True,
    "theta_feat_dim": 2,
    "use_slice_z_inject": True,
    "z_inject_dropout": 0.0,
    "z_inject_layers": -1,
    
    # --- 运行控制 ---
    "vis_interval": 5,
    "num_workers": 4,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ==========================================
# 辅助类与函数 (Logger, Saver, Dataset) 保持原样逻辑
# ==========================================
class SaveBestModel:
    def __init__(self, path, verbose=True):
        self.path = path
        self.verbose = verbose
        self.best_metric = float("inf")
    def __call__(self, current_metric, model, epoch):
        if current_metric < self.best_metric:
            if self.verbose:
                print(f" >> [Epoch {epoch}] Best Rel_L2 ({self.best_metric:.6f} -> {current_metric:.6f}). Saving...")
            self.best_metric = current_metric
            torch.save(model.state_dict(), self.path)

class FastPhysicsDataset(Dataset):
    def __init__(self, normalized_data_path, deepsdf_ckpt_path, num_points=2048):
        print(f"[*] 加载归一化场数据: {normalized_data_path}...")
        self.data = np.load(normalized_data_path, mmap_mode="r")
        self.fields_norm = self.data["fields_norm"]  
        self.mean = torch.from_numpy(self.data["mean"]).float()
        self.std = torch.from_numpy(self.data["std"]).float()
        self.n_shapes = self.fields_norm.shape[0]
        self.n_angles = self.fields_norm.shape[1]
        self.resolution = self.fields_norm.shape[2]
        self.total_samples = self.n_shapes * self.n_angles

        print(f"[*] 加载 Stable-SDF Latent Codes: {deepsdf_ckpt_path}...")
        ckpt = torch.load(deepsdf_ckpt_path, map_location="cpu")
        self.latents = ckpt["latents"]["weight"].detach()
        self.num_points = num_points
        x = np.linspace(-1, 1, self.resolution)
        y = np.linspace(-1, 1, self.resolution)
        xv, yv = np.meshgrid(x, y)
        self.grid_coords = np.stack([xv.flatten(), yv.flatten()], axis=1).astype(np.float32)
        print("[*] 数据集准备完毕。")

    def __len__(self): return self.total_samples

    def __getitem__(self, idx):
        shape_idx = idx // self.n_angles
        angle_idx = idx % self.n_angles
        z = self.latents[shape_idx]  
        theta_val = angle_idx * (2 * np.pi / self.n_angles)
        theta = torch.tensor([theta_val], dtype=torch.float32)  
        field_map = self.fields_norm[shape_idx, angle_idx]      
        flat_field = field_map.reshape(-1, 2)                   
        sample_indices = np.random.choice(len(self.grid_coords), self.num_points, replace=False)
        coords = torch.from_numpy(self.grid_coords[sample_indices])          
        targets = torch.from_numpy(flat_field[sample_indices]).float()       
        return z, coords, theta, targets

# 验证与可视化函数省略具体细节，逻辑不变，仅更新参数
@torch.no_grad()
def validate_epoch(model, dataloader, device, dataset):
    model.eval()
    total_rel_l1, total_rel_l2, count = 0.0, 0.0, 0
    mean, std = dataset.mean.to(device).view(1, 1, 2), dataset.std.to(device).view(1, 1, 2)
    use_amp, amp_dtype = (device.type == "cuda"), torch.bfloat16
    
    it = tqdm(dataloader, desc="Val", leave=False)
    for z_b, coords_b, theta_b, targets_norm in it:
        z_b, coords_b = z_b.to(device), coords_b.to(device)
        theta_b, targets_norm = theta_b.to(device), targets_norm.to(device)
        
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            pred_norm = model((coords_b, coords_b, (z_b, theta_b))) 
        
        pred_real = pred_norm * std + mean
        targets_real = targets_norm * std + mean
        B = z_b.size(0)
        diff, ref = (pred_real - targets_real).reshape(B, -1), targets_real.reshape(B, -1)
        rel_l1 = (torch.norm(diff, p=1, dim=1) / (torch.norm(ref, p=1, dim=1) + 1e-8)).mean()
        rel_l2 = (torch.norm(diff, p=2, dim=1) / (torch.norm(ref, p=2, dim=1) + 1e-8)).mean()
        
        total_rel_l1 += rel_l1.item() * B; total_rel_l2 += rel_l2.item() * B; count += B
    return {"rel_l1": total_rel_l1 / count, "rel_l2": total_rel_l2 / count}

def visualize_results(model, dataset, device, epoch, save_dir):
    model.eval()
    idx = random.randint(0, len(dataset) - 1)
    shape_idx, angle_idx = idx // dataset.n_angles, idx % dataset.n_angles
    grid_coords = torch.from_numpy(dataset.grid_coords).to(device).unsqueeze(0)  
    z = dataset.latents[shape_idx].to(device).unsqueeze(0)                       
    theta = torch.tensor([[angle_idx * (2 * np.pi / dataset.n_angles)]], dtype=torch.float32).to(device)  
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.bfloat16):
        pred_norm = model((grid_coords, grid_coords, (z, theta)))  

    mean, std = dataset.mean.to(device).view(1, 1, 2), dataset.std.to(device).view(1, 1, 2)
    pred_real = pred_norm * std + mean  
    gt_norm = torch.from_numpy(dataset.fields_norm[shape_idx, angle_idx]).to(device).unsqueeze(0)  
    gt_real = gt_norm * std.view(1, 1, 1, 2) + mean.view(1, 1, 1, 2)
    
    R = dataset.resolution
    pred_amp = torch.norm(pred_real, dim=-1).cpu().numpy().reshape(R, R)
    gt_amp = torch.norm(gt_real, dim=-1).cpu().numpy().reshape(R, R)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(gt_amp, cmap="inferno", origin="lower"); plt.title("GT (Real)"); plt.colorbar()
    plt.subplot(1, 3, 2); plt.imshow(pred_amp, cmap="inferno", origin="lower"); plt.title(f"Pred (Epoch {epoch})"); plt.colorbar()
    plt.subplot(1, 3, 3); plt.imshow(np.abs(pred_amp - gt_amp), cmap="viridis", origin="lower"); plt.title("Error"); plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch:03d}.png"))
    plt.close()

# ==========================================
# 训练主循环
# ==========================================
def main():
    # 路径解析
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    norm_data = os.path.normpath(os.path.join(cur_dir, CONFIG["norm_data"]))
    deepsdf_ckpt = os.path.normpath(os.path.join(cur_dir, CONFIG["deepsdf_ckpt"]))
    ckpt_dir = os.path.normpath(os.path.join(cur_dir, CONFIG["ckpt_dir"]))
    output_dir = os.path.normpath(os.path.join(cur_dir, CONFIG["output_dir"]))
    
    os.makedirs(ckpt_dir, exist_ok=True); os.makedirs(output_dir, exist_ok=True)
    
    # 随机种子设定
    torch.manual_seed(CONFIG["seed"]); np.random.seed(CONFIG["seed"]); random.seed(CONFIG["seed"])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(CONFIG["seed"])
    torch.backends.cudnn.benchmark = True

    device = torch.device(CONFIG["device"])
    full_dataset = FastPhysicsDataset(norm_data, deepsdf_ckpt, num_points=CONFIG["samples_per_item"])

    train_size = int(0.9 * len(full_dataset))
    train_ds, test_ds = random_split(full_dataset, [train_size, len(full_dataset) - train_size])

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)

    model = TransolverModel(
        space_dim=CONFIG["space_dim"], fun_dim=0, out_dim=CONFIG["out_dim"],
        n_hidden=CONFIG["n_hidden"], n_layers=CONFIG["n_layers"], n_head=CONFIG["n_head"],
        dropout=CONFIG["dropout"], mlp_ratio=CONFIG["mlp_ratio"], slice_num=CONFIG["slice_num"],
        unified_pos=CONFIG["unified_pos"], ref=CONFIG["ref"],
        use_theta_in_coord=CONFIG["use_theta_in_coord"], theta_feat_dim=CONFIG["theta_feat_dim"],
        use_slice_z_inject=CONFIG["use_slice_z_inject"], z_dim=CONFIG["z_dim"],
        z_inject_dropout=CONFIG["z_inject_dropout"], z_inject_layers=CONFIG["z_inject_layers"]
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["max_lr"], weight_decay=CONFIG["weight_decay"], fused=(device.type == "cuda"))
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CONFIG["max_lr"], steps_per_epoch=len(train_loader), epochs=CONFIG["epochs"], pct_start=CONFIG["pct_start"], div_factor=CONFIG["div_factor"], final_div_factor=CONFIG["final_div_factor"], anneal_strategy="cos")
    
    criterion = nn.L1Loss()
    saver = SaveBestModel(path=os.path.join(ckpt_dir, "best_transolver.pth"))

    print(f"[*] 启动 Transolver 前向训练 (A100 bfloat16 + OneCycleLR)...")
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_l1_loss = 0.0
        
        train_it = tqdm(train_loader, desc=f"Train {epoch+1}/{CONFIG['epochs']}", leave=False)
        for z_b, coords_b, theta_b, targets_b in train_it:
            z_b, coords_b = z_b.to(device), coords_b.to(device)
            theta_b, targets_b = theta_b.to(device), targets_b.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                pred = model((coords_b, coords_b, (z_b, theta_b)))
                loss = criterion(pred, targets_b)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip_grad"])
            optimizer.step()
            scheduler.step()
            total_l1_loss += loss.item()
            train_it.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_train_loss = total_l1_loss / max(1, len(train_loader))
        metrics = validate_epoch(model, test_loader, device, full_dataset)
        
        print(f"Epoch {epoch+1:03d} | Tr_L1: {avg_train_loss:.5f} | Val_Rel_L2: \033[92m{metrics['rel_l2']:.5f}\033[0m")
        
        if (epoch + 1) % CONFIG["vis_interval"] == 0:
            visualize_results(model, full_dataset, device, epoch + 1, output_dir)
        saver(metrics["rel_l2"], model, epoch + 1)

    print("[*] 训练结束。所有权重已保存。")

if __name__ == "__main__":
    main()