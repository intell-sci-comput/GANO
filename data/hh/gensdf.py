"""
代码说明：
用于生成 HelmHoltz (HH) 任务中用于训练 Stable-SDF 模型的混合多尺度 SDF 采样数据。
在原有形状掩膜的基础上，采用精细 (fine)、粗略 (coarse) 和全局 (global) 三种策略混合采样，
以增强模型对物体边界的刻画能力。
输入：由 genshape.py 生成的形状及 SDF 数据 (.npz)。
输出：包含多尺度采样点及其对应 SDF 真值的数据体 (.npz)。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, map_coordinates
from skimage.measure import find_contours
from joblib import Parallel, delayed
import os
import time

SMOKE_TEST = os.environ.get("GANO_SMOKE_TEST", "0").lower() in {"1", "true", "yes", "on"}


def env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


# ==========================================
# 统一参数配置中心
# ==========================================
CONFIG = {
    # --- 采样与噪声配置 ---
    'num_points': env_int("GANO_HH_SDF_NUM_POINTS", 10000),  # 每个样本的总采样点数
    'resolution': env_int("GANO_HH_SDF_RESOLUTION", 256),    # 网格分辨率
    'ratios': {
        'fine': 0.4,         # 精细表面采样占比 40%
        'coarse': 0.4,       # 粗略表面采样占比 40%
        'global': 0.2        # 全局均匀采样占比 20%
    },
    'noise_levels': {
        'fine': 0.005,       # 精细采样的高斯噪声标准差
        'coarse': 0.025      # 粗略采样的高斯噪声标准差
    },
    
    # --- 路径配置 (相对当前脚本路径) ---
    'data_dir': "../../data/hh",               # 数据统一存放目录
    'load_name': "scattering_shapes_256.npz",  # 从 genshape.py 获取的源数据
    'save_name': "scattering_sdf_dataset_mixed.npz", # 本次生成的 SDF 数据
    'output_dir': "../../output/hh",           # 验证图片存放目录
    'vis_name': "final_dataset_check.png",
    
    # --- 运行配置 ---
    'n_jobs': env_int("GANO_HH_SDF_N_JOBS", -1),  # 并行核心数 (-1 表示跑满 CPU)
    'verbose': 5             # 并行执行的日志等级
}

if SMOKE_TEST:
    CONFIG.update({
        'num_points': env_int("GANO_HH_SDF_SMOKE_NUM_POINTS", 256),
        'resolution': env_int("GANO_HH_SDF_SMOKE_RESOLUTION", 64),
        'n_jobs': env_int("GANO_HH_SDF_SMOKE_N_JOBS", 1),
        'verbose': 0,
    })

# ==========================================
# 核心处理函数 (保持数学采样逻辑不变)
# ==========================================

def compute_sdf_grid(mask, resolution):
    """计算全场 SDF 网格 (物理距离)"""
    dist_outside = distance_transform_edt(1 - mask)
    dist_inside = distance_transform_edt(mask)
    
    sdf_pixel = dist_outside - dist_inside - 0.5
    pixel_size = 2.0 / resolution
    sdf_physical = sdf_pixel * pixel_size
    
    return sdf_physical

def resample_single_shape(mask, seed=None):
    """对单个 Mask 进行混合重采样"""
    if seed is not None:
        np.random.seed(seed)
        
    resolution = mask.shape[0]
    num_points = CONFIG['num_points']
    
    # 1. 提取高精度亚像素轮廓
    contours = find_contours(mask, level=0.5)
    if not contours:
        print("[!] Warning: Empty mask found!")
        return np.random.uniform(-1, 1, (num_points, 2)), np.ones(num_points)
        
    surface_points_pixel = np.vstack(contours)
    
    # 转换为物理坐标 (row, col) -> (y, x)
    surface_y = (surface_points_pixel[:, 0] / (resolution - 1)) * 2 - 1
    surface_x = (surface_points_pixel[:, 1] / (resolution - 1)) * 2 - 1
    surface_points_phys = np.stack([surface_x, surface_y], axis=1)
    
    # 2. 计算各部分采样数量
    n_fine = int(num_points * CONFIG['ratios']['fine'])
    n_coarse = int(num_points * CONFIG['ratios']['coarse'])
    n_global = num_points - n_fine - n_coarse
    
    # 3. 表面采样 (添加多尺度噪声)
    idx_fine = np.random.choice(len(surface_points_phys), n_fine)
    idx_coarse = np.random.choice(len(surface_points_phys), n_coarse)
    
    points_fine = surface_points_phys[idx_fine] + np.random.normal(0, CONFIG['noise_levels']['fine'], (n_fine, 2))
    points_coarse = surface_points_phys[idx_coarse] + np.random.normal(0, CONFIG['noise_levels']['coarse'], (n_coarse, 2))
    
    # 4. 全局采样
    points_global = np.random.uniform(-1, 1, (n_global, 2))
    
    # 合并并截断到物理域内
    samples_xy = np.vstack([points_fine, points_coarse, points_global])
    samples_xy = np.clip(samples_xy, -1.0, 1.0)
    
    # 5. 计算 SDF 真值 (双线性插值)
    sdf_grid = compute_sdf_grid(mask, resolution)
    grid_coords_x = (samples_xy[:, 0] + 1) / 2 * (resolution - 1)
    grid_coords_y = (samples_xy[:, 1] + 1) / 2 * (resolution - 1)
    
    samples_sdf = map_coordinates(sdf_grid, [grid_coords_y, grid_coords_x], order=1)
    
    return samples_xy.astype(np.float32), samples_sdf.astype(np.float32)

def verify_dataset(file_path, vis_path):
    """简单验证生成的第一个样本"""
    print("\n[*] 执行最终验证绘图...")
    data = np.load(file_path, allow_pickle=True)
    points = data['points'][0]
    sdfs = data['sdfs'][0]
    mask = data['masks'][0]
    
    plt.figure(figsize=(10, 5))
    
    # Mask
    plt.subplot(1, 2, 1)
    plt.imshow(mask, origin='lower', extent=[-1, 1, -1, 1], cmap='gray')
    plt.title("Original Mask (Sample 0)")
    
    # Points
    plt.subplot(1, 2, 2)
    mask_surf = np.abs(sdfs) < 0.1
    plt.scatter(points[~mask_surf, 0], points[~mask_surf, 1], s=1, c='gray', alpha=0.1, label='Far Field')
    plt.scatter(points[mask_surf, 0], points[mask_surf, 1], s=1, c=sdfs[mask_surf], cmap='seismic', vmin=-0.05, vmax=0.05, label='Near Surface')
    plt.colorbar(label='SDF')
    plt.legend()
    plt.title("Resampled Points (Colored by SDF)")
    plt.xlim(-1, 1); plt.ylim(-1, 1)
    
    plt.tight_layout()
    plt.savefig(vis_path)
    plt.close()
    print(f"[*] 验证图片已保存为 {vis_path}")

# ==========================================
# 主流程
# ==========================================
def main():
    # --- 1. 路径准备 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(current_dir, CONFIG['data_dir']))
    output_dir = os.path.normpath(os.path.join(current_dir, CONFIG['output_dir']))
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    src_path = os.path.join(data_dir, CONFIG['load_name'])
    save_path = os.path.join(data_dir, CONFIG['save_name'])
    vis_path = os.path.join(output_dir, CONFIG['vis_name'])
    
    print(f"[*] 正在加载源文件: {src_path} ...")
    if not os.path.exists(src_path):
        print(f"[!] 错误: 找不到文件: {src_path}")
        print(f"    请确保先运行过 genshape.py。")
        return

    data = np.load(src_path)
    masks = data['masks'] 
    num_samples = masks.shape[0]
    
    print(f"[*] 数据加载完毕。样本数: {num_samples}, 分辨率: {masks.shape[1]}x{masks.shape[2]}")
    
    # --- 2. 并行处理 ---
    print(f"\n[*] 开始并行重采样 (Total Points: {CONFIG['num_points']})...")
    start_time = time.time()
    
    results = Parallel(n_jobs=CONFIG['n_jobs'], verbose=CONFIG['verbose'])(
        delayed(resample_single_shape)(masks[i], seed=i) 
        for i in range(num_samples)
    )
    
    all_points = np.array([r[0] for r in results]) 
    all_sdfs = np.array([r[1] for r in results])   
    
    duration = time.time() - start_time
    print(f"[*] 处理完成！耗时: {duration:.2f} 秒")
    
    # --- 3. 保存新数据集 ---
    print(f"[*] 正在保存至: {save_path} ...")
    np.savez_compressed(
        save_path,
        masks=masks,        
        points=all_points,
        sdfs=all_sdfs,
        config=CONFIG       
    )
    print(f"[*] 保存完毕。文件大小: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")
    
    # --- 4. 最终验证 ---
    verify_dataset(save_path, vis_path)

if __name__ == "__main__":
    main()
