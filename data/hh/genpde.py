"""
代码说明：
用于生成 HelmHoltz (HH) 任务的偏微分方程 (PDE) 数据。
基于有限差分法 (FDM) 和完全匹配层 (PML) 求解 2D 频域散射场。
输入：由 genshape.py 生成的形状及 SDF 数据 (.npz)。
输出：包含多角度散射场、掩膜、采样点及对应 SDF 的数据体 (.npz)。
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, kron, eye, csc_matrix
from scipy.sparse.linalg import splu
from joblib import Parallel, delayed
import os
import time

SMOKE_TEST = os.environ.get("GANO_SMOKE_TEST", "0").lower() in {"1", "true", "yes", "on"}


def env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


def env_float(name, default):
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else default


# ==========================================
# 统一参数配置中心
# ==========================================
CONFIG = {
    # --- 物理场与网格配置 ---
    'resolution': env_int("GANO_HH_PDE_RESOLUTION", 256),  # 网格分辨率
    'domain_size': env_float("GANO_HH_PDE_DOMAIN_SIZE", 2.0),  # 物理计算域大小
    'k': env_float("GANO_HH_PDE_K", 7.0),  # 波数 (wavenumber)
    'n_angles': env_int("GANO_HH_PDE_N_ANGLES", 10),  # 入射波的角度数量
    'pml_width': env_int("GANO_HH_PDE_PML_WIDTH", 30),  # 完全匹配层 (PML) 宽度
    'pml_sigma': env_float("GANO_HH_PDE_PML_SIGMA", 8.0),  # PML 最大吸收系数
    
    # --- 路径配置 (相对当前脚本路径) ---
    'data_dir': "../../data/hh",               # 数据统一存放目录
    'load_name': "scattering_shapes_256.npz",  # 上一步生成的形状数据
    'save_name': "scattering_dataset_scat_fields_k7.npz", # 本次生成的 PDE 数据
    'output_dir': "../../output/hh",           # 验证图片存放目录
    'vis_name': "scat_field_dataset_check.png",
    
    # --- 运行配置 ---
    'n_jobs': env_int("GANO_HH_PDE_N_JOBS", 32),  # 并行核心数
    'verbose': 5             # 并行执行的日志等级
}

if SMOKE_TEST:
    CONFIG.update({
        'resolution': env_int("GANO_HH_PDE_SMOKE_RESOLUTION", 64),
        'n_angles': env_int("GANO_HH_PDE_SMOKE_N_ANGLES", 2),
        'pml_width': env_int("GANO_HH_PDE_SMOKE_PML_WIDTH", 8),
        'n_jobs': env_int("GANO_HH_PDE_SMOKE_N_JOBS", 1),
        'verbose': 0,
    })

# ==========================================
# 核心求解逻辑 (保持原有数学逻辑不变)
# ==========================================

def build_solver_matrices(mask, k, N, h):
    absorb_width = CONFIG['pml_width']
    sigma_max = CONFIG['pml_sigma']
    
    sx = np.zeros(N)
    sy = np.zeros(N)
    for i in range(absorb_width):
        val = ((absorb_width - i) / absorb_width) ** 2 * sigma_max
        sx[i] = sx[N-1-i] = val
        sy[i] = sy[N-1-i] = val
        
    SX, SY = np.meshgrid(sx, sy)
    sigma = SX + SY
    k_sq_map = (k + 1j * sigma)**2
    
    data = np.ones((3, N))
    data[1] = -2
    diags = [-1, 0, 1]
    D2 = spdiags(data, diags, N, N) / h**2
    I = eye(N)
    L = kron(D2, I) + kron(I, D2)
    
    k_term = k_sq_map * (1 + mask)
    K_mat = spdiags(k_term.flatten(), 0, N*N, N*N)
    
    A = L + K_mat
    return csc_matrix(A)

def solve_scattered_fields_for_sample(mask, sample_idx):
    """为一个样本计算所有角度的散射场 u_scat"""
    N = CONFIG['resolution']
    k = CONFIG['k']
    domain_size = CONFIG['domain_size']
    h = domain_size / (N - 1)
    
    # 1. 构建矩阵
    A = build_solver_matrices(mask, k, N, h)
    solver = splu(A)
    
    # 2. 准备 RHS
    x = np.linspace(-domain_size/2, domain_size/2, N)
    X, Y = np.meshgrid(x, x)
    angles = np.linspace(0, 2*np.pi, CONFIG['n_angles'], endpoint=False)
    
    rhs_list = []
    for ang in angles:
        u_inc = np.exp(1j * k * (X * np.cos(ang) + Y * np.sin(ang)))
        # 方程: (Lap + k^2(1+q)) u_scat = -k^2 * q * u_inc
        rhs = -(k**2) * mask * u_inc
        rhs_list.append(rhs.flatten())
        
    B = np.column_stack(rhs_list)
    
    # 3. 求解
    U_scat_flat = solver.solve(B)
    
    # 4. 直接保存散射场
    scat_fields = np.zeros((CONFIG['n_angles'], N, N), dtype=np.complex64)
    for i in range(CONFIG['n_angles']):
        u_scat = U_scat_flat[:, i].reshape(N, N)
        scat_fields[i] = u_scat.astype(np.complex64)
        
    return scat_fields

def visualize_full_batch(sample_fields, mask, save_path):
    """检查散射场可视化"""
    n_show = 5
    fig, axes = plt.subplots(1, n_show + 1, figsize=(18, 4))
    
    axes[0].imshow(mask, origin='lower', cmap='gray')
    axes[0].set_title("Target Mask")
    axes[0].axis('off')
    
    for i in range(n_show):
        amp = np.abs(sample_fields[i])
        ax = axes[i+1]
        im = ax.imshow(amp, origin='lower', cmap='inferno')
        ax.set_title(f"Scattered Field {i}")
        ax.axis('off')
    
    plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[*] 检查图片已保存为: {save_path}")

def main():
    # --- 1. 路径准备 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(current_dir, CONFIG['data_dir']))
    output_dir = os.path.normpath(os.path.join(current_dir, CONFIG['output_dir']))
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    load_path = os.path.join(data_dir, CONFIG['load_name'])
    save_path = os.path.join(data_dir, CONFIG['save_name'])
    vis_path = os.path.join(output_dir, CONFIG['vis_name'])
    
    if not os.path.exists(load_path):
        print(f"[!] 错误: 找不到输入数据文件 {load_path}")
        print(f"    请确保先运行过 genshape.py 生成形状数据。")
        return

    print(f"[*] 正在加载形状数据: {load_path} ...")
    data = np.load(load_path)
    masks = data['masks']
    points = data['points']
    sdfs = data['sdfs']
    
    num_samples = masks.shape[0]
    
    # --- 2. 并行计算 PDE ---
    print(f"[*] 开始并行计算 {num_samples} 个样本的散射场 (使用 {CONFIG['n_jobs']} 核)...")
    start_time = time.time()
    
    results = Parallel(n_jobs=CONFIG['n_jobs'], verbose=CONFIG['verbose'])(
        delayed(solve_scattered_fields_for_sample)(masks[i], i) 
        for i in range(num_samples)
    )
    
    fields_data = np.array(results, dtype=np.complex64)
    
    duration = time.time() - start_time
    print(f"[*] 计算完成！总耗时: {duration:.2f} 秒")
    
    # --- 3. 存储结果 ---
    print(f"[*] 正在保存带散射场的数据集至 {save_path} ...")
    np.savez_compressed(
        save_path,
        fields=fields_data,  
        masks=masks,
        points=points,
        sdfs=sdfs,
        k=CONFIG['k']
    )
    print("[*] 保存完毕。")

    # --- 4. 可视化检查 ---
    visualize_full_batch(fields_data[0], masks[0], vis_path)

if __name__ == "__main__":
    main()
