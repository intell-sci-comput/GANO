"""
3D 汽车任务：压力场数据预处理脚本
说明：
1. 采用全局字典 CONFIG 管理超参数和输入输出路径。
2. 从给定的根目录自动遍历所有 VTK 格式的物理场模型。
3. 保持原有的目录结构，将归一化后的数据 (.npz) 保存到统一的输出目录。
4. 几何采用 bbox 对角线归一化到 1.9，压强采用全局 Mean/Std 进行标准化。
"""

import pyvista as pv
import numpy as np
import os
import glob
import json
import multiprocessing as mp
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
SMOKE_TEST = os.environ.get("GANO_SMOKE_TEST", "0").lower() in {"1", "true", "yes", "on"}


def env_path(name, default):
    return os.environ.get(name, default)


def env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


# ==============================================================================
# ======= [全局参数配置字典] ================
# ==============================================================================
CONFIG = {
    # --- 路径配置 ---
    # 汽车物理场(压强)原始数据根目录；服务器真实路径可通过 GANO_CAR_PRESSURE_RAW_ROOT 覆盖。
    "DATA_ROOT": env_path(
        "GANO_CAR_PRESSURE_RAW_ROOT",
        os.path.join(project_root, "data", "car", "raw", "PressureVTK"),
    ),
    # 统一的输出目录
    "SAVE_DIR": env_path("GANO_CAR_PRESSURE_DIR", os.path.join(project_root, "data", "car", "pressure")),
    
    # --- 压强全局统计量 ---
    "GLOBAL_MEAN": -93.427311,
    "GLOBAL_STD": 120.596359,
    
    # --- 几何归一化 ---
    "TARGET_SCALE": 1.9,
    
    # --- 并发配置 ---
    "NUM_WORKERS": env_int("GANO_CAR_PRESSURE_NUM_WORKERS", 32),
    "FILE_EXTENSION": "*.vtk"
}

if SMOKE_TEST:
    CONFIG.update({
        "NUM_WORKERS": env_int("GANO_CAR_PRESSURE_SMOKE_NUM_WORKERS", 1),
    })

# ==========================================
# 1. 核心数学组件
# ==========================================
def normalize_geometry_bbox_diagonal(points, target_scale=1.9):
    """
    完全复刻 SDF 生成时的坐标变换逻辑
    """
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    
    centroid = (min_coords + max_coords) / 2.0
    points_centered = points - centroid
    
    diagonal = np.linalg.norm(max_coords - min_coords)
    if diagonal < 1e-6: diagonal = 1.0
        
    scale_factor = target_scale / diagonal
    points_normalized = points_centered * scale_factor
    
    return points_normalized, centroid, scale_factor

def find_pressure_array(mesh):
    if 'p' in mesh.point_data: return np.array(mesh.point_data['p'])
    keys = list(mesh.point_data.keys())
    for k in keys:
        data = mesh.point_data[k]
        if len(data.shape) == 1 or (len(data.shape) == 2 and data.shape[1] == 1):
            return np.array(data)
    return None

# ==========================================
# 2. 多进程 Worker (处理并保存)
# ==========================================

def worker_process_save(args):
    """
    负责归一化、转换和保存
    Args: (vtk_path, input_root, output_root)
    """
    vtk_path, input_root, output_root = args
    
    try:
        # 1. 读取
        mesh = pv.read(vtk_path)
        points = np.array(mesh.points)
        pressure = find_pressure_array(mesh)
        
        if pressure is None: return False
        
        # 2. 几何归一化 (SDF 对齐)
        points_norm, centroid, scale_factor = normalize_geometry_bbox_diagonal(points, target_scale=CONFIG["TARGET_SCALE"])
        
        # 3. 压强标准化 (使用预设的 Mean/Std)
        pressure_norm = (pressure - CONFIG["GLOBAL_MEAN"]) / CONFIG["GLOBAL_STD"]
        
        # 4. 数据类型转换 (Float32)
        points_f32 = points_norm.astype(np.float32)
        pressure_f32 = pressure_norm.astype(np.float32).reshape(-1, 1)
        
        # 5. 计算保存路径 (保持目录结构)
        rel_path = os.path.relpath(vtk_path, input_root)
        save_path = os.path.join(output_root, rel_path.replace('.vtk', '.npz'))
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 6. 保存
        np.savez(save_path,
                 coords=points_f32,
                 data=pressure_f32,
                 meta={
                     'centroid': centroid,
                     'scale_factor': scale_factor,
                     'normalization_method': f'bbox_diagonal_{CONFIG["TARGET_SCALE"]}',
                     'global_mean': CONFIG["GLOBAL_MEAN"],
                     'global_std': CONFIG["GLOBAL_STD"],
                     'original_path': vtk_path
                 })
        return True
    except Exception as e:
        return False

# ==========================================
# 3. 主流程
# ==========================================

def main():
    input_dir = CONFIG["DATA_ROOT"]
    output_dir = CONFIG["SAVE_DIR"]
    num_workers = CONFIG["NUM_WORKERS"]
    
    if not os.path.exists(input_dir):
        print(f"错误: 找不到数据根目录 {input_dir}")
        return

    # 1. 搜索文件
    search_pattern = os.path.join(input_dir, "**", CONFIG["FILE_EXTENSION"])
    vtk_files = glob.glob(search_pattern, recursive=True)
    
    if not vtk_files:
        print("No files found.")
        return

    total_files = len(vtk_files)
    print(f"Found {total_files} files.")
    print(f"Using fixed stats -> Mean: {CONFIG['GLOBAL_MEAN']}, Std: {CONFIG['GLOBAL_STD']}")
    print(f"Starting processing with {num_workers} workers...")
    
    # 2. 准备任务参数
    tasks = [(f, input_dir, output_dir) for f in vtk_files]
    success_count = 0
    
    # 3. 并行处理
    with mp.Pool(processes=num_workers) as pool:
        for success in tqdm(pool.imap_unordered(worker_process_save, tasks, chunksize=2), 
                            total=total_files, desc="Processing"):
            if success: success_count += 1

    # 4. 保存数据集元信息
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "dataset_stats.json"), 'w') as f:
        json.dump({
            "global_mean": float(CONFIG["GLOBAL_MEAN"]),
            "global_std": float(CONFIG["GLOBAL_STD"]),
            "total_files": total_files,
            "success_files": success_count,
            "normalization_method": f"bbox_diagonal_target_{CONFIG['TARGET_SCALE']}"
        }, f, indent=4)
        
    print(f"\nProcessing Complete!")
    print(f"Successfully processed: {success_count}/{total_files}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()
