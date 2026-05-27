"""
代码说明：
用于生成 HelmHoltz (HH) 任务的 2D 随机形状及对应的 SDF (Signed Distance Field) 采样数据。
包含形状参数化生成、全场 SDF 计算、表面及全局均匀采样。
输出数据格式为 .npz, 存入 data/hh 目录，供后续 Stable-sdf 模型训练使用。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
from scipy.ndimage import distance_transform_edt, map_coordinates
from joblib import Parallel, delayed
from tqdm import tqdm

# ==========================================
# 统一参数配置中心
# ==========================================
CONFIG = {
    # --- 数据集规模配置 ---
    "num_samples": 1000,               # 生成样本的总数
    "resolution": 256,                 # 物理场的网格分辨率 (256x256)
    "num_points": 10000,               # 每个样本采样的点数量 (包括边界点和全局点)
    
    # --- 形状生成配置 ---
    "base_radius": 0.3,                # 基础半径
    "max_radius_limit": 0.5,           # 最大半径限制
    "surface_point_ratio": 0.8,        # 边界表面采样点占比 (默认 80%)
    "noise_std": 0.005,                # 表面采样点的高斯噪声标准差
    
    # --- 路径配置 (相对当前脚本路径) ---
    "data_dir": "../../data/hh",       # 生成数据的保存目录
    "save_name": "scattering_shapes_256.npz", 
    "output_dir": "../../output/hh",   # 可视化图片的保存目录
    "vis_save_name": "shape_sample_check.png",
    
    # --- 运行配置 ---
    "n_jobs": -1,                      # 并行核心数 (-1 表示跑满 CPU)
    "visualize_first_sample": True     # 是否在生成结束后可视化第一个样本
}


# ==========================================
# 核心业务逻辑 (不改变原有数学与采样逻辑)
# ==========================================

def generate_smooth_shape(resolution=256, num_modes=5, base_radius=0.3, perturbation=0.15, max_radius_limit=0.5):
    theta = np.linspace(0, 2*np.pi, 1000) # 增加边界点密度以用于高质量采样
    
    r = np.full_like(theta, base_radius)
    for k in range(1, num_modes + 1):
        decay = 1.0 / k 
        ak = np.random.uniform(-1, 1) * perturbation * decay
        bk = np.random.uniform(-1, 1) * perturbation * decay
        r += ak * np.cos(k * theta) + bk * np.sin(k * theta)
    
    r = np.maximum(r, 0.05) 
    current_max = np.max(r)
    if current_max > max_radius_limit:
        scale_factor = (max_radius_limit - 0.01) / current_max 
        r *= scale_factor
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # 生成 Mask
    img_coords_r = ((y + 1) / 2 * resolution).astype(int)
    img_coords_c = ((x + 1) / 2 * resolution).astype(int)
    img_coords_r = np.clip(img_coords_r, 0, resolution - 1)
    img_coords_c = np.clip(img_coords_c, 0, resolution - 1)
    
    mask = np.zeros((resolution, resolution), dtype=np.float32)
    rr, cc = polygon(img_coords_r, img_coords_c, shape=mask.shape)
    mask[rr, cc] = 1.0
    
    # 返回高精度边界点 (N, 2) 和 Mask
    return mask, np.stack([x, y], axis=1)

def compute_sdf_grid(mask, resolution=256):
    """计算整个网格的 SDF 场。内部为负，外部为正。"""
    dist_outside = distance_transform_edt(1 - mask)
    dist_inside = distance_transform_edt(mask)
    
    # 合并：内部为负，外部为正，减去0.5是为了让0值面更接近真实的连续边界
    sdf_pixel = dist_outside - dist_inside
    
    # 将像素距离转换为物理距离 (跨度为 2.0)
    pixel_size = 2.0 / resolution
    sdf_physical = sdf_pixel * pixel_size
    
    return sdf_physical

def generate_single_sample(seed):
    """生成单个样本的所有数据"""
    np.random.seed(seed)
    
    # 1. 随机化形状参数
    modes = np.random.randint(3, 8)
    pert = np.random.uniform(0.1, 0.2)
    
    # 2. 生成形状 (接入 CONFIG 参数)
    mask, boundary_points = generate_smooth_shape(
        resolution=CONFIG["resolution"], 
        num_modes=modes, 
        base_radius=CONFIG["base_radius"],
        perturbation=pert,
        max_radius_limit=CONFIG["max_radius_limit"]
    )
    
    # 3. 计算全场 SDF
    sdf_grid = compute_sdf_grid(mask, CONFIG["resolution"])
    
    # 4. 生成采样点 (X, Y)
    n_surface = int(CONFIG["num_points"] * CONFIG["surface_point_ratio"])
    n_uniform = CONFIG["num_points"] - n_surface
    
    # 4.1 表面采样 + 高斯噪声
    indices = np.random.choice(len(boundary_points), n_surface)
    surface_samples = boundary_points[indices]
    noise = np.random.normal(0, CONFIG["noise_std"], surface_samples.shape)
    surface_samples += noise
    
    # 4.2 全局均匀采样
    uniform_samples = np.random.uniform(-1, 1, (n_uniform, 2))
    
    # 合并采样点
    samples_xy = np.vstack([surface_samples, uniform_samples])
    
    # 5. 获取采样点的 SDF 值 (双线性插值)
    grid_coords_x = (samples_xy[:, 0] + 1) / 2 * (CONFIG["resolution"] - 1)
    grid_coords_y = (samples_xy[:, 1] + 1) / 2 * (CONFIG["resolution"] - 1)
    
    samples_sdf = map_coordinates(sdf_grid, [grid_coords_y, grid_coords_x], order=1)
    
    return {
        "mask": mask.astype(np.float32),          
        "samples_xy": samples_xy.astype(np.float32), 
        "samples_sdf": samples_sdf.astype(np.float32) 
    }

def visualize_check(sample_data, save_path):
    """可视化检查生成的样本分布"""
    mask = sample_data['mask']
    xy = sample_data['samples_xy']
    sdf = sample_data['samples_sdf']
    
    plt.figure(figsize=(12, 5))
    
    # 图1: Mask 和 采样点分布
    plt.subplot(1, 2, 1)
    plt.imshow(mask, extent=[-1, 1, -1, 1], origin='lower', cmap='gray', alpha=0.5)
    plt.scatter(xy[:2000, 0], xy[:2000, 1], c=sdf[:2000], cmap='seismic', s=1, vmin=-0.1, vmax=0.1)
    plt.colorbar(label='SDF Value')
    plt.title("Sample Points Distribution (First 2000)")
    plt.xlim(-1, 1); plt.ylim(-1, 1)
    
    # 图2: SDF 值直方图
    plt.subplot(1, 2, 2)
    plt.hist(sdf, bins=50, color='blue', alpha=0.7)
    plt.title("Histogram of SDF Values")
    plt.xlabel("Signed Distance")
    
    plt.savefig(save_path)
    print(f"[*] 样本检查图片已保存至: {save_path}")
    plt.close()

def create_dataset():
    num_samples = CONFIG["num_samples"]
    print(f"[*] 开始生成 {num_samples} 个 HH 形状样本...")
    
    # 并行生成
    results = Parallel(n_jobs=CONFIG["n_jobs"])(
        delayed(generate_single_sample)(seed=i) 
        for i in tqdm(range(num_samples))
    )
    
    # 整理数据
    masks = np.array([r['mask'] for r in results])
    points = np.array([r['samples_xy'] for r in results])
    sdfs = np.array([r['samples_sdf'] for r in results])
    
    print(f"[*] 数据生成完毕。")
    print(f"    Masks shape: {masks.shape}")
    print(f"    Points shape: {points.shape}")
    print(f"    SDFs shape: {sdfs.shape}")
    
    # 处理路径并保存
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.normpath(os.path.join(current_dir, CONFIG["data_dir"]))
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, CONFIG["save_name"])
    np.savez_compressed(
        save_path, 
        masks=masks, 
        points=points, 
        sdfs=sdfs
    )
    print(f"[*] 数据集已成功保存至: {save_path}")
    
    return results[0]

if __name__ == "__main__":
    # 执行数据集生成
    sample_0 = create_dataset()
    
    # 可视化检查第一个样本
    if CONFIG["visualize_first_sample"]:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.normpath(os.path.join(current_dir, CONFIG["output_dir"]))
        os.makedirs(output_dir, exist_ok=True)
        
        vis_path = os.path.join(output_dir, CONFIG["vis_save_name"])
        visualize_check(sample_0, vis_path)