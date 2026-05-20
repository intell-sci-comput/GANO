"""
2D 机翼任务：GI-Transolver 训练脚本
说明：本版本已按 GANO 规范重构，采用全局 CONFIG 字典。
自动读取预处理好的包含 Latents(z) 和原始物理场的混合数据文件。
"""

import os
import sys
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 将 GANO 仓库根目录加入系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.airfoil.gi_transolver import GITransolver

# ==========================================
# ======= [全局参数配置字典] ================
# ==========================================
CONFIG = {
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "SEED": 42,
    
    # --- 路径配置 ---
    "PHYSICS_DATA": os.path.join(project_root, "data", "airfoil", "airfoil_physics_train.pt"),
    "SAVE_DIR": os.path.join(project_root, "checkpoints", "airfoil_transolver"),
    "VIS_DIR": os.path.join(project_root, "output", "airfoil_transolver_vis"),
    "LOG_FILE": os.path.join(project_root, "output", "training_log_transolver.csv"),
    
    # --- 训练超参数 ---
    "BATCH_SIZE": 64,
    "EPOCHS": 200,
    "MAX_LR": 5e-4,
    "WEIGHT_DECAY": 0.0,
    "CLIP_GRAD": 2.0,
    "SAMPLES_PER_ITEM": 4096,
    "NUM_WORKERS": 4,

    # --- Transolver 模型架构超参数 ---
    "Z_DIM": 64,         # 必须与 DeepSDF Latent Dim 对齐
    "SPACE_DIM": 2,
    "OUT_DIM": 3,        # U, V, P
    "N_HIDDEN": 256,
    "N_LAYERS": 5,
    "N_HEAD": 8,
    "DROPOUT": 0.0,
    "MLP_RATIO": 1,
    "SLICE_NUM": 32,
    "UNIFIED_POS": False,
    "REF": 8,

    # --- 特征注入策略 ---
    "USE_SLICE_Z_ADD": True,
    "Z_ADD_DROPOUT": 0.0,
    "Z_INJECT_LAYERS": -1, # -1: 所有层注入
    
    "VIS_INTERVAL": 5,
}

os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)
os.makedirs(CONFIG["VIS_DIR"], exist_ok=True)
os.makedirs(os.path.dirname(CONFIG["LOG_FILE"]), exist_ok=True)

class SaveBestModel:
    def __init__(self, path, verbose=True):
        self.path = path
        self.verbose = verbose
        self.best_metric = float("inf")

    def __call__(self, current_metric, model, epoch):
        if current_metric < self.best_metric:
            if self.verbose:
                print(f" >> [Epoch {epoch}] Best Rel_L2 found ({self.best_metric:.5f} -> {current_metric:.5f}). Saving...")
            self.best_metric = current_metric
            torch.save(model.state_dict(), self.path)

class Logger:
    def __init__(self, filename):
        self.filename = filename
        self.headers = ["Epoch", "Train_L1", "Val_L1_Norm", "Val_Rel_L1", "Val_Rel_L2", "LR", "Time"]
        with open(self.filename, "w") as f:
            f.write(",".join(self.headers) + "\n")

    def log(self, m):
        msg = (f"Epoch {m['Epoch']:03d} | Tr_L1: {m['Train_L1']:.5f} | "
               f"Val_Rel_L2: \033[92m{m['Val_Rel_L2']:.5f}\033[0m | "
               f"Val_Rel_L1: {m['Val_Rel_L1']:.5f} | LR: {m['LR']:.2e} | T: {m['Time']:.1f}s")
        print(msg)
        row = [str(m['Epoch']), f"{m['Train_L1']:.6f}", f"{m['Val_L1_Norm']:.6f}", 
               f"{m['Val_Rel_L1']:.6f}", f"{m['Val_Rel_L2']:.6f}", f"{m['LR']:.2e}", f"{m['Time']:.2f}"]
        with open(self.filename, "a") as f:
            f.write(",".join(row) + "\n")

class AirfoilPhysicsDataset(Dataset):
    def __init__(self, z_tensor, coords_list, targets_list, samples_per_item=2048):
        self.z = z_tensor
        self.coords_list = coords_list
        self.targets_list = targets_list
        self.samples_per_item = samples_per_item

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        z_i = self.z[idx]
        coords_i = self.coords_list[idx]
        targets_i = self.targets_list[idx]

        if coords_i.shape[-1] == 3:
            coords_i = coords_i[..., :2]

        num_points = coords_i.shape[0]
        if num_points >= self.samples_per_item:
            indices = torch.randperm(num_points)[: self.samples_per_item]
        else:
            indices = torch.randint(0, num_points, (self.samples_per_item,))

        return z_i, coords_i[indices], targets_i[indices]

@torch.no_grad()
def validate_epoch(model, dataloader, device, stats):
    model.eval()
    total_l1_norm = 0.0
    total_rel_l1 = 0.0
    total_rel_l2 = 0.0
    count = 0

    mean = stats["mean"].to(device).view(1, 1, -1)
    std = stats["std"].to(device).view(1, 1, -1)
    criterion_l1 = nn.L1Loss()

    for z_b, coords_b, targets_b in dataloader:
        z_b = z_b.to(device)
        coords_b = coords_b.to(device)
        targets_b = targets_b.to(device)

        pred_norm = model((coords_b, coords_b, z_b))
        loss_norm = criterion_l1(pred_norm, targets_b)
        total_l1_norm += loss_norm.item() * z_b.size(0)

        pred_real = pred_norm * (std + 1e-8) + mean
        target_real = targets_b * (std + 1e-8) + mean

        B = z_b.size(0)
        diff = (pred_real - target_real).view(B, -1)
        ref = target_real.view(B, -1)

        rel_l1_batch = ((torch.norm(diff, p=1, dim=1)) / (torch.norm(ref, p=1, dim=1) + 1e-8)).mean()
        rel_l2_batch = ((torch.norm(diff, p=2, dim=1)) / (torch.norm(ref, p=2, dim=1) + 1e-8)).mean()

        total_rel_l1 += rel_l1_batch.item() * B
        total_rel_l2 += rel_l2_batch.item() * B
        count += B

    model.train()
    return {"l1_norm": total_l1_norm / count, "rel_l1": total_rel_l1 / count, "rel_l2": total_rel_l2 / count}

def visualize_epoch(model, dataset, stats, device, epoch):
    model.eval()
    idx = random.randint(0, len(dataset) - 1)
    z_i, _, _ = dataset[idx]

    res = 200
    x = np.linspace(-0.5, 1.5, res)
    y = np.linspace(-0.5, 0.5, res)
    xv, yv = np.meshgrid(x, y)
    grid_coords = torch.tensor(np.stack([xv, yv], axis=-1), dtype=torch.float32).reshape(1, -1, 2).to(device)
    z_batch = z_i.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_norm = model((grid_coords, grid_coords, z_batch))
        mean = stats["mean"].to(device).view(1, 1, -1)
        std = stats["std"].to(device).view(1, 1, -1)
        pred_phys = pred_norm * (std + 1e-8) + mean

        pred_arr = pred_phys[0].cpu().numpy()
        u_pred = pred_arr[:, 0].reshape(res, res)
        p_pred = pred_arr[:, 2].reshape(res, res)

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(u_pred, origin="lower", cmap="turbo", extent=[-0.5, 1.5, -0.5, 0.5])
    plt.colorbar(label="U Velocity")
    plt.title(f"Transolver Pred U (Epoch {epoch})")

    plt.subplot(2, 1, 2)
    plt.imshow(p_pred, origin="lower", cmap="magma", extent=[-0.5, 1.5, -0.5, 0.5])
    plt.colorbar(label="Pressure")
    plt.title(f"Transolver Pred P (Epoch {epoch})")

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["VIS_DIR"], f"epoch_{epoch:03d}.png"))
    plt.close()
    model.train()

def main():
    torch.manual_seed(CONFIG["SEED"])
    np.random.seed(CONFIG["SEED"])
    random.seed(CONFIG["SEED"])

    if not os.path.exists(CONFIG["PHYSICS_DATA"]):
        print(f"Error: 找不到数据文件 {CONFIG['PHYSICS_DATA']}。请先运行预处理脚本。")
        return

    print("Loading preprocessed physical data and latents...")
    phys_data = torch.load(CONFIG["PHYSICS_DATA"])
    z_all = phys_data["latents"]       # 预处理时已经将 latents 对齐并打包，直接用！
    coords_list = phys_data["coords"]
    targets_list = phys_data["targets"]
    stats = phys_data["stats"]

    full_dataset = AirfoilPhysicsDataset(z_all, coords_list, targets_list, samples_per_item=CONFIG["SAMPLES_PER_ITEM"])
    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_ds, test_ds = random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(CONFIG["SEED"]))

    train_loader = DataLoader(train_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)

    model = GITransolver(
        space_dim=CONFIG["SPACE_DIM"], fun_dim=0, out_dim=CONFIG["OUT_DIM"], n_hidden=CONFIG["N_HIDDEN"],
        n_layers=CONFIG["N_LAYERS"], n_head=CONFIG["N_HEAD"], dropout=CONFIG["DROPOUT"], mlp_ratio=CONFIG["MLP_RATIO"],
        slice_num=CONFIG["SLICE_NUM"], unified_pos=CONFIG["UNIFIED_POS"], ref=CONFIG["REF"],
        use_slice_z_add=CONFIG["USE_SLICE_Z_ADD"], z_dim=CONFIG["Z_DIM"], z_add_dropout=CONFIG["Z_ADD_DROPOUT"], z_inject_layers=CONFIG["Z_INJECT_LAYERS"]
    ).to(CONFIG["DEVICE"])

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["MAX_LR"], weight_decay=CONFIG["WEIGHT_DECAY"])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CONFIG["MAX_LR"], steps_per_epoch=len(train_loader), epochs=CONFIG["EPOCHS"], pct_start=0.01, div_factor=10)

    logger = Logger(CONFIG["LOG_FILE"])
    saver = SaveBestModel(path=os.path.join(CONFIG["SAVE_DIR"], "airfoil_transolver_best.pth"))

    print(f"Start Transolver Training... Logs: {CONFIG['LOG_FILE']}")

    for epoch in range(CONFIG["EPOCHS"]):
        start_time = time.time()
        model.train()
        total_l1_loss = 0.0

        for z_b, coords_b, targets_b in train_loader:
            z_b, coords_b, targets_b = z_b.to(CONFIG["DEVICE"]), coords_b.to(CONFIG["DEVICE"]), targets_b.to(CONFIG["DEVICE"])
            optimizer.zero_grad(set_to_none=True)
            
            pred = model((coords_b, coords_b, z_b))
            loss = torch.mean(torch.abs(pred - targets_b))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["CLIP_GRAD"])
            optimizer.step()
            scheduler.step()
            total_l1_loss += loss.item()

        duration = time.time() - start_time
        avg_train_loss = total_l1_loss / max(1, len(train_loader))
        
        val_metrics = validate_epoch(model, test_loader, CONFIG["DEVICE"], stats)
        logger.log({"Epoch": epoch + 1, "Train_L1": avg_train_loss, "Val_L1_Norm": val_metrics["l1_norm"], "Val_Rel_L1": val_metrics["rel_l1"], "Val_Rel_L2": val_metrics["rel_l2"], "LR": optimizer.param_groups[0]["lr"], "Time": duration})

        if (epoch + 1) % CONFIG["VIS_INTERVAL"] == 0:
            visualize_epoch(model, test_ds, stats, CONFIG["DEVICE"], epoch + 1)
        saver(val_metrics["rel_l2"], model, epoch + 1)

    print("Training finished.")

if __name__ == "__main__":
    main()