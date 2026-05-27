"""
2D 机翼任务：物理场数据预处理脚本
说明：本版本已针对开源仓库规范进行重构。
从 56GB 的原始 H5 文件中提取保守变量，解算为原始量 (u, v, p)，
并结合 DeepSDF 预训练权重中的隐向量 (z)，进行空间过滤和标准化，最终输出为 PyTorch Tensor。
"""

import h5py
import numpy as np
import torch
import os
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 获取仓库根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

# ==========================================
# ======= [全局参数配置字典] ================
# ==========================================
CONFIG = {
    # --- 输入文件路径 (指向 A100 上的原位置，避免复制 56GB 巨型文件) ---
    'H5_PATH': '/home/sunguoze/airfoil/airfoil_9k_data.h5',          
    'LATENTS_PATH': os.path.join(project_root, 'checkpoints', 'airfoil_stablesdf', 'latents_latest.pth'),
    'FALLBACK_LATENTS_PATHS': [
        '/home/sunguoze/airfoil/deepsdf_airfoil.pth',
    ],
    
    # --- 输出文件路径 ---
    'SAVE_PATH': os.path.join(project_root, 'data', 'airfoil', 'airfoil_physics_train.pt'),
    
    # --- 物理参数 ---
    'ALPHA_GROUP': 'alpha+04', # 选择攻角 (例如 4 度)
    'GAMMA': 1.4,              # 空气绝热指数
    
    # --- 空间过滤范围 (只保留机翼附近的点) ---
    'X_MIN': -1.0, 'X_MAX': 2.0,
    'Y_MIN': -1.0, 'Y_MAX': 1.0,
    
    # --- 调试选项 ---
    # 设为整数(如 10)可只处理前 N 个样本用于快速测试，None 处理全部
    'PROCESS_LIMIT': None 
}

# 确保输出目录存在
os.makedirs(os.path.dirname(CONFIG['SAVE_PATH']), exist_ok=True)

def compute_uvp(rho, rho_u, rho_v, e, gamma=1.4):
    """
    从守恒量 (Conservative Variables) 解算原始量 (Primitive Variables)
    """
    # 加上极小值防止除以0
    rho = np.maximum(rho, 1e-8)
    
    # 1. 速度 u, v
    u = rho_u / rho
    v = rho_v / rho
    
    # 2. 压力 p
    kinetic_energy = 0.5 * rho * (u**2 + v**2)
    internal_energy = e - kinetic_energy
    p = (gamma - 1.0) * internal_energy
    
    return u, v, p

def resolve_latents_path():
    candidates = [CONFIG['LATENTS_PATH'], *CONFIG.get('FALLBACK_LATENTS_PATHS', [])]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No latent checkpoint found in candidates: {candidates}")

def load_latents(ckpt_path):
    """
    从 DeepSDF Checkpoint 加载隐向量 z
    """
    print(f"Loading latents from {ckpt_path}...")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    if isinstance(ckpt, torch.Tensor):
        z_all = ckpt
        indices = None
    elif isinstance(ckpt, dict) and 'weight' in ckpt:
        z_all = ckpt['weight']
        indices = ckpt.get('indices')
    elif isinstance(ckpt, dict) and 'latents' in ckpt:
        latents_data = ckpt['latents']
        indices = ckpt.get('indices')
        if isinstance(latents_data, dict) and 'weight' in latents_data:
            z_all = latents_data['weight']
            indices = latents_data.get('indices', indices)
        elif isinstance(latents_data, torch.Tensor):
            z_all = latents_data
        else:
            z_all = latents_data.get('weight', None)
            indices = latents_data.get('indices', indices)
    elif isinstance(ckpt, dict):
        print("Warning: 'latents' key not found directly. Searching state_dict...")
        z_all = None
        indices = ckpt.get('indices')
        for key in ckpt.keys():
            if 'latents' in key and 'weight' in key:
                z_all = ckpt[key]
                break
    else:
        z_all = None
        indices = None
                
    if z_all is None:
        raise ValueError("Could not extract latent vectors from checkpoint.")

    if indices is None:
        indices = torch.arange(z_all.shape[0], dtype=torch.long)
    else:
        indices = torch.as_tensor(indices, dtype=torch.long)

    if len(indices) != z_all.shape[0]:
        raise ValueError(f"Latent/index length mismatch: {z_all.shape[0]} latents vs {len(indices)} indices.")

    print(f"Successfully loaded {z_all.shape[0]} latent codes with dimension {z_all.shape[1]}.")
    return z_all, indices

def build_dataset():
    # 1. 加载隐向量 z
    try:
        latents_path = resolve_latents_path()
        z_all, latent_indices = load_latents(latents_path)
    except Exception as e:
        print(f"Error loading latents: {e}")
        return

    # 2. 准备处理 H5 数据
    if not os.path.exists(CONFIG['H5_PATH']):
        print(f"Error: {CONFIG['H5_PATH']} not found.")
        return

    print(f"Processing physics data from {CONFIG['H5_PATH']}...")
    
    processed_data = {
        "latents": [],   # z
        "coords": [],    # (x, y)
        "targets": [],   # (u, v, p)
        "indices": []    # 记录原始样本索引，方便追溯
    }
    
    with h5py.File(CONFIG['H5_PATH'], 'r') as hf:
        group = hf[CONFIG['ALPHA_GROUP']]
        flow_group = group['flow_field']
        
        sample_ids = sorted(flow_group.keys(), key=lambda x: int(x))
        total_samples = len(sample_ids)
        limit = CONFIG['PROCESS_LIMIT'] if CONFIG['PROCESS_LIMIT'] else len(latent_indices)
        limit = min(limit, len(latent_indices))
        
        print(f"Target processing count: {limit}")
        print("Phase 1: Extracting...")
        
        for i in tqdm(range(limit), desc="Extracting Fields"):
            source_idx = int(latent_indices[i].item())
            if source_idx >= total_samples:
                print(f"Warning: source index {source_idx} out of H5 range {total_samples}, skipping.")
                continue
            sid = sample_ids[source_idx]
            z_vec = z_all[i] 
            
            sample = flow_group[sid]
            x = sample['x'][:]
            y = sample['y'][:]
            
            # 空间过滤 (Masking)
            mask = (x >= CONFIG['X_MIN']) & (x <= CONFIG['X_MAX']) & \
                   (y >= CONFIG['Y_MIN']) & (y <= CONFIG['Y_MAX'])
            
            if np.sum(mask) < 100:
                continue 
            
            x_val = x[mask]
            y_val = y[mask]
            rho = sample['rho'][:][mask]
            rho_u = sample['rho_u'][:][mask]
            rho_v = sample['rho_v'][:][mask]
            e = sample['e'][:][mask]
            
            u, v, p = compute_uvp(rho, rho_u, rho_v, e, CONFIG['GAMMA'])
            
            coords = np.stack([x_val, y_val], axis=1) 
            targets = np.stack([u, v, p], axis=1)     
            
            processed_data["latents"].append(z_vec)
            processed_data["coords"].append(torch.from_numpy(coords).float())
            processed_data["targets"].append(torch.from_numpy(targets).float())
            processed_data["indices"].append(source_idx)

    # --- 第二轮：标准化 (Standardization) ---
    if not processed_data["targets"]:
        print("Error: no valid physics samples were extracted.")
        return

    print("Phase 2: Computing Statistics on ALL data...")
    all_targets_cat = torch.cat(processed_data["targets"], dim=0)
    
    mean = torch.mean(all_targets_cat, dim=0)
    std = torch.std(all_targets_cat, dim=0)
    
    print(f"Stats (u, v, p):")
    print(f"  Mean: {mean.numpy()}")
    print(f"  Std : {std.numpy()}")
    
    del all_targets_cat
    
    print("Phase 3: Normalizing Targets...")
    normalized_targets = []
    for t in tqdm(processed_data["targets"], desc="Normalizing"):
        norm_t = (t - mean) / (std + 1e-6)
        normalized_targets.append(norm_t)
        
    # --- 保存 ---
    print("Phase 4: Saving...")
    save_dict = {
        "latents": torch.stack(processed_data["latents"]), 
        "coords": processed_data["coords"],
        "targets": normalized_targets,
        "indices": torch.tensor(processed_data["indices"]),
        "stats": {
            "mean": mean,
            "std": std
        }
    }
    
    torch.save(save_dict, CONFIG['SAVE_PATH'])
    print(f"Dataset saved successfully to {CONFIG['SAVE_PATH']}")
    print(f"Final sample count: {len(normalized_targets)}")

if __name__ == "__main__":
    build_dataset()
