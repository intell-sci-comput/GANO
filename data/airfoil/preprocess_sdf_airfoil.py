"""
2D 机翼任务：SDF 数据批量采样与预处理脚本
说明：本版本已针对开源仓库规范进行重构，采用统一的 CONFIG 字典管理超参数。
从原始 H5 文件中读取机翼的 landmarks，应用混合采样与加噪策略，生成 SDF 数据并保存为 PyTorch Tensor。
"""

import os
import sys
import torch
import numpy as np
import h5py
from shapely.geometry import Polygon, Point
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 获取仓库根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

SMOKE_TEST = os.environ.get("GANO_SMOKE_TEST", "0").lower() in {"1", "true", "yes", "on"}


def env_path(name, default):
    return os.environ.get(name, default)


def env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


# ==========================================
# ======= [全局参数配置字典] ================
# ==========================================
CONFIG = {
    # --- 路径配置 ---
    # 原始机翼数据文件；服务器真实路径可通过 GANO_AIRFOIL_H5_PATH 覆盖。
    "DATA_FILE": env_path(
        "GANO_AIRFOIL_H5_PATH",
        os.path.join(project_root, "data", "airfoil", "raw", "airfoil_9k_data.h5"),
    ),
    # 处理后数据的统一保存路径
    "SAVE_DIR": env_path("GANO_AIRFOIL_SDF_SAVE_DIR", os.path.join(project_root, "data", "airfoil")),
    "SAVE_NAME": "airfoil_sdf_train.pt",

    # --- 数据集处理控制 ---
    # 设为 None 处理所有数据，或设为整数(如 100)用于快速跑通测试
    "NUM_SHAPES_TO_PROCESS": None,

    # --- 总体采样超参数 ---
    "NUM_SAMPLES": env_int("GANO_AIRFOIL_SDF_NUM_SAMPLES", 4096),  # 每个翼型的采样点数
    "BOUNDS": [-0.2, 1.2, -0.25, 0.25],  # 采样空间边界 [x_min, x_max, y_min, y_max]

    # --- 混合采样策略分布 ---
    "GLOBAL_RATIO": 0.2,                 # 全局均匀采样点占比
    "SMALL_NOISE_RATIO": 0.8,            # 在近表面点中，小噪声点占比
    
    # --- 噪声强度 ---
    "SIGMA_SMALL": 0.005,                # 小高斯噪声标准差
    "SIGMA_LARGE": 0.05,                 # 大高斯噪声标准差
}

if SMOKE_TEST:
    CONFIG.update({
        "NUM_SHAPES_TO_PROCESS": env_int("GANO_AIRFOIL_SDF_SMOKE_SHAPES", 2),
        "NUM_SAMPLES": env_int("GANO_AIRFOIL_SDF_SMOKE_NUM_SAMPLES", 128),
    })

os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)

def generate_sdf_samples(airfoil_points):
    """
    生成单个翼型的 SDF 样本 (混合策略)
    """
    # 1. 构建几何体
    try:
        poly = Polygon(airfoil_points)
        if not poly.is_valid:
            # 尝试修复自交等问题
            poly = poly.buffer(0)
        line = poly.boundary
    except Exception as e:
        print(f"Geometry error: {e}")
        return None, None

    # 2. 采样策略
    # A. 全局均匀采样
    num_samples = CONFIG["NUM_SAMPLES"]
    n_global = int(num_samples * CONFIG["GLOBAL_RATIO"])
    x_min, x_max, y_min, y_max = CONFIG["BOUNDS"]
    
    global_points = np.random.uniform(
        low=[x_min, y_min], 
        high=[x_max, y_max], 
        size=(n_global, 2)
    )
    
    # B. 近表面采样
    n_surface = num_samples - n_global
    # 随机选择轮廓上的点
    indices = np.random.choice(len(airfoil_points), n_surface)
    base_points = airfoil_points[indices]
    
    # 添加高斯噪声 (多尺度噪声，捕捉细节)
    split = int(n_surface * CONFIG["SMALL_NOISE_RATIO"])
    
    noise_small = np.random.normal(0, CONFIG["SIGMA_SMALL"], (split, 2))
    noise_large = np.random.normal(0, CONFIG["SIGMA_LARGE"], (n_surface - split, 2))
    noise = np.vstack([noise_small, noise_large])
    
    surface_points = base_points + noise
    
    # 合并
    query_points = np.vstack([global_points, surface_points])
    
    # 3. 计算 SDF
    sdf_values = []
    for p in query_points:
        point = Point(p)
        # 边界检查
        if not (x_min <= p[0] <= x_max and y_min <= p[1] <= y_max):
            sdf_values.append(1.0) # 外部默认大值
            continue
            
        dist = line.distance(point)
        if poly.contains(point):
            sdf_values.append(-dist) # 内部负
        else:
            sdf_values.append(dist)  # 外部正
            
    return query_points, np.array(sdf_values)


def process_dataset():
    data_file = CONFIG["DATA_FILE"]
    save_path = os.path.join(CONFIG["SAVE_DIR"], CONFIG["SAVE_NAME"])

    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found.")
        return

    print(f"Reading {data_file}...")
    with h5py.File(data_file, 'r') as hf:
        # 数据结构为 hf['shape']['landmarks']，形状 (N, Points, 2)
        landmarks_dataset = hf['shape']['landmarks']
        total_shapes = landmarks_dataset.shape[0]
        
        target_count = CONFIG["NUM_SHAPES_TO_PROCESS"] if CONFIG["NUM_SHAPES_TO_PROCESS"] else total_shapes
        print(f"Processing {target_count} shapes...")
        
        all_coords = []
        all_sdfs = []
        all_indices = []
        
        # 使用 tqdm 显示进度
        for i in tqdm(range(target_count), desc="Processing Airfoils"):
            points = landmarks_dataset[i] # (N_pts, 2)
            
            # 生成 SDF 数据
            coords, sdfs = generate_sdf_samples(points)
            
            if coords is not None:
                all_coords.append(coords)
                all_sdfs.append(sdfs)
                all_indices.append(i)
            else:
                print(f"Skipping shape {i} due to error.")
                # 跳过后必须记录成功样本的原始 index，后续物理场 latent 依赖该映射。

    if not all_coords:
        raise RuntimeError("No valid airfoil SDF samples were generated.")

    # 转换为 Tensor
    print("Converting to Tensors...")
    coords_tensor = torch.tensor(np.stack(all_coords), dtype=torch.float32)
    sdfs_tensor = torch.tensor(np.stack(all_sdfs), dtype=torch.float32).unsqueeze(-1) # (N, Samples, 1)
    
    # 保存
    torch.save({
        'coords': coords_tensor,
        'sdfs': sdfs_tensor,
        'indices': torch.tensor(all_indices, dtype=torch.long)
    }, save_path)
    
    print(f"Dataset saved to {save_path}")
    print(f"Coords shape: {coords_tensor.shape}")
    print(f"SDFs shape: {sdfs_tensor.shape}")

if __name__ == "__main__":
    process_dataset()
