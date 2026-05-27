"""
3D 汽车任务：反问题（形状优化与重建）脚本
说明：本版本已针对开源仓库规范进行重构，采用统一的 CONFIG 字典管理超参数，彻底移除 argparse 和硬编码。
"""

import os
import sys
import glob
import json
import time
import torch
import numpy as np
import trimesh
from skimage import measure
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 将 GANO 仓库根目录加入系统路径，确保能导入 src 里的模型
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# 注意：这里需要根据你模型代码的实际类名进行导入
from src.stablesdf.model import DeepSDFNet 

# ==========================================
# ======= [全局参数配置字典] ================
# ==========================================
CONFIG = {
    # --- 权重与数据路径 ---
    "CHECKPOINT_DIR": os.path.join(project_root, "checkpoints", "car_training_h800_all"),
    "MODEL_NAME": 'model_latest.pth',
    "LATENT_NAME": 'latents_latest.pth',
    
    # --- 保存路径 ---
    "SAVE_DIR": os.path.join(project_root, "output", "car_reconstruction"),
    
    # --- 重建超参数 ---
    "RESOLUTION": 512,       # Marching Cubes 的网格分辨率
    "LATENT_SIZE": 256,      # 隐向量维度
    "CHUNK_SIZE": 65536,     # H800 显存大，可以开大一点以加速推断
    
    # --- 运行模式 (替代原 argparse) ---
    "SCENE_ID": 0,           # 训练集中的车辆 ID (0 ~ N-1，仅在 USE_RANDOM_Z 为 False 时有效)
    "USE_RANDOM_Z": True,    # 是否使用随机高斯隐向量 z，而不是训练好的 latent code
    "Z_SIGMA": 1e-2,         # 随机 z 的标准差
}

# 确保输出目录存在
os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)

def reconstruct():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载模型结构
    model = DeepSDFNet(latent_size=CONFIG["LATENT_SIZE"]).to(device)
    model_path = os.path.join(CONFIG["CHECKPOINT_DIR"], CONFIG["MODEL_NAME"])
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # 加载权重 (增加容错处理)
    try:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    except Exception as e:
        print(f"Loading strict failed, trying non-strict: {e}")
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state, strict=False)
    
    model.eval()

    # 2. 准备 latent code
    if CONFIG["USE_RANDOM_Z"]:
        # 随机初始化 z ~ N(0, z_sigma^2)
        latent_code = torch.randn(1, CONFIG["LATENT_SIZE"], device=device) * CONFIG["Z_SIGMA"]
        print(f"[Random z mode] Using latent z ~ N(0, {CONFIG['Z_SIGMA']}^2), shape={latent_code.shape}")
    else:
        # 加载 Latent Codes 矩阵
        latent_path = os.path.join(CONFIG["CHECKPOINT_DIR"], CONFIG["LATENT_NAME"])
        print(f"Loading latents from {latent_path}...")
        
        if not os.path.exists(latent_path):
            print(f"Error: Latent file not found at {latent_path}")
            return
            
        latents_dict = torch.load(latent_path, map_location=device)
        
        # 处理 Embedding 权重字典
        if isinstance(latents_dict, dict) and 'weight' in latents_dict:
            all_latents = latents_dict['weight']
        else:
            all_latents = latents_dict
            
        # 检查 ID 是否越界
        if CONFIG["SCENE_ID"] >= all_latents.shape[0]:
            print(f"Error: Scene ID {CONFIG['SCENE_ID']} 超出范围 (最大 {all_latents.shape[0]-1})")
            return
            
        # 提取特定的一辆车的隐向量
        latent_code = all_latents[CONFIG["SCENE_ID"]:CONFIG["SCENE_ID"] + 1].to(device)  # (1, latent_size)
        print(f"[Trained latent mode] Reconstructing Scene ID: {CONFIG['SCENE_ID']}, latent shape={latent_code.shape}")

    # 3. 空间推断 (Marching Cubes)
    start_time = time.time()
    print("Generating query points on GPU...")
    
    coords = torch.linspace(-1.0, 1.0, CONFIG["RESOLUTION"], device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing='ij')
    points_tensor = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)
    
    sdf_values = []
    print(f"Starting inference on {points_tensor.shape[0]} points...")
    
    with torch.no_grad():
        for i in tqdm(range(0, points_tensor.shape[0], CONFIG["CHUNK_SIZE"]), desc="SDF Inference"):
            chunk_points = points_tensor[i:i + CONFIG["CHUNK_SIZE"]]
            chunk_latents = latent_code.expand(chunk_points.shape[0], -1)
            pred = model(chunk_points, chunk_latents)
            sdf_values.append(pred.cpu().numpy())
            
    sdf_grid = np.concatenate(sdf_values).reshape(CONFIG["RESOLUTION"], CONFIG["RESOLUTION"], CONFIG["RESOLUTION"])
    elapsed = time.time() - start_time
    print(f"SDF grid evaluated in {elapsed:.2f} seconds.")

    # 4. 提取 Mesh
    level = 0.0
    if sdf_grid.min() > 0:
        level = float(sdf_grid.min() + 0.001)
        print(f"Warning: No negative SDF values. Using level={level}")
        
    try:
        verts, faces, normals, values = measure.marching_cubes(sdf_grid, level=level)
        verts = verts * (2.0 / (CONFIG["RESOLUTION"] - 1)) - 1.0
        now = time.strftime("%Y%m%d_%H%M%S")
        
        if CONFIG["USE_RANDOM_Z"]:
            filename = f'train_recon_random_z_sigma_{CONFIG["Z_SIGMA"]:.0e}_{now}.obj'
        else:
            filename = f'train_recon_id_{CONFIG["SCENE_ID"]}.obj'
            
        # 组装完整的保存路径
        save_path = os.path.join(CONFIG["SAVE_DIR"], filename)
        
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        mesh.export(save_path)
        print(f"Success! Saved to {save_path}")
        
    except Exception as e:
        print(f"Failed to extract mesh: {e}")

if __name__ == "__main__":
    # 直接调用主函数，不再解析命令行参数
    reconstruct()
