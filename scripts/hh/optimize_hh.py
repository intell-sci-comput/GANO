"""
代码说明：
HelmHoltz (HH) 任务的反问题求解。
基于训练好的 GI-Transolver (正问题) 和 Stable-SDF (形状解码)，
利用稀疏传感器点的散射场数据，通过 OneCycleLR 策略反演优化隐变量 z，从而重建原始形状。
并将可视化结果保存至 GANO 规范的 output 目录。
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, kron, eye, csc_matrix
from scipy.sparse.linalg import splu
from scipy.ndimage import map_coordinates
from skimage.draw import polygon
from tqdm import tqdm

# 将项目根目录加入 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.gi_transolver.model import Model as TransolverModel
from src.stablesdf.deepsdf import DeepSDFWithPE

# ==========================================
# 统一参数配置中心
# ==========================================
CONFIG = {
    # --- 预训练模型与统计路径 (必须保证这些文件存在) ---
    "deepsdf_ckpt": "../../checkpoints/hh/stablesdf/deepsdf_final.pth",
    "transolver_ckpt": "../../checkpoints/hh/transolver/best_transolver.pth",
    "norm_stats": "../../data/hh/normalization_stats.pt",  # 注意后缀是 .pt，由 normalize_pde.py 生成
    "output_dir": "../../output/hh/inverse_vis",
    
    # --- 反演优化配置 ---
    "num_sensors": 100,        # 传感器采样点数量
    "optim_steps": 100,        # OneCycleLR 优化步数 (想快可以改成更小测试)
    "max_lr": 0.01,            # 优化最大学习率
    "reg_weight": 1e-4,        # Latent z 正则化系数
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

FIXED_ANGLES = np.linspace(0, 2*np.pi, 10, endpoint=False).astype(np.float32)

# ==========================================
# 1. 形状生成与 FDM 求解 (保持原有数学逻辑)
# ==========================================
def generate_random_shape(resolution=256):
    theta = np.linspace(0, 2*np.pi, 1000)
    base_radius = 0.3
    num_modes = np.random.randint(3, 7)
    perturbation = np.random.uniform(0.1, 0.2)
    
    r = np.full_like(theta, base_radius)
    for k in range(1, num_modes + 1):
        decay = 1.0 / k 
        ak = np.random.uniform(-1, 1) * perturbation * decay
        bk = np.random.uniform(-1, 1) * perturbation * decay
        r += ak * np.cos(k * theta) + bk * np.sin(k * theta)
    r = np.maximum(r, 0.05); r = np.minimum(r, 0.45) 
    x = r * np.cos(theta); y = r * np.sin(theta)
    img_coords_r = np.clip(((y + 1) / 2 * resolution).astype(int), 0, resolution - 1)
    img_coords_c = np.clip(((x + 1) / 2 * resolution).astype(int), 0, resolution - 1)
    mask = np.zeros((resolution, resolution), dtype=np.float32)
    rr, cc = polygon(img_coords_r, img_coords_c, shape=mask.shape)
    mask[rr, cc] = 1.0
    return mask

def solve_fdm_multiview(mask):
    N = 256; k = 7.0; domain = 2.0; h = domain/(N-1)
    pml_w = 30; pml_s = 8.0
    sx = np.zeros(N); sy = np.zeros(N)
    for i in range(pml_w):
        v = ((pml_w-i)/pml_w)**2 * pml_s
        sx[i]=sx[N-1-i]=v; sy[i]=sy[N-1-i]=v
    SX,SY = np.meshgrid(sx,sy)
    k_map = (k + 1j*(SX+SY))**2 * (1 + mask)
    data = np.ones((3, N)); data[1]=-2
    D2 = spdiags(data, [-1,0,1], N, N)/h**2
    L = kron(D2, eye(N)) + kron(eye(N), D2)
    A = csc_matrix(L + spdiags(k_map.flatten(), 0, N*N, N*N))
    solver = splu(A)
    x = np.linspace(-1, 1, N); X, Y = np.meshgrid(x, x)
    fields = []
    print(f"[*] 正在计算 {len(FIXED_ANGLES)} 个角度的 Ground Truth (FDM)...")
    for ang in tqdm(FIXED_ANGLES, desc="FDM Solving", leave=False):
        u_inc = np.exp(1j * k * (X*np.cos(ang) + Y*np.sin(ang)))
        rhs = -(k**2) * mask * u_inc
        u_scat = solver.solve(rhs.flatten()).reshape(N, N)
        fields.append(u_scat)
    return np.array(fields)

def sample_multiview_sensors(fields_list, r_min=0.5, r_max=0.51, num_sensors=100):
    N = fields_list.shape[1]
    n_angles = fields_list.shape[0]
    random_r = np.sqrt(np.random.uniform(r_min**2, r_max**2, num_sensors))
    random_theta = np.random.uniform(0, 2*np.pi, num_sensors)
    x = random_r * np.cos(random_theta); y = random_r * np.sin(random_theta)
    sensor_coords = np.stack([x, y], axis=1).astype(np.float32) 
    grid_x = (x + 1) / 2 * (N - 1); grid_y = (y + 1) / 2 * (N - 1)
    all_values = []
    for i in range(n_angles):
        u_field = fields_list[i]
        real_vals = map_coordinates(u_field.real, [grid_y, grid_x], order=1)
        imag_vals = map_coordinates(u_field.imag, [grid_y, grid_x], order=1)
        vals = np.stack([real_vals, imag_vals], axis=1) 
        all_values.append(vals)
    return sensor_coords, np.array(all_values, dtype=np.float32)

# ==========================================
# 2. 反问题优化核心
# ==========================================
def inverse_solve_onecycle(phys_model, sensor_coords, sensor_values_all, device, mean, std, num_steps):
    phys_model.eval()
    for p in phys_model.parameters(): p.requires_grad = False
    
    z = torch.randn(1, 64, device=device, requires_grad=True)
    torch.nn.init.normal_(z, 0.0, 0.02)
    optimizer = optim.Adam([z])
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CONFIG["max_lr"], total_steps=num_steps, 
        pct_start=0.1, anneal_strategy='cos', div_factor=5.0, final_div_factor=100.0
    )
    loss_fn = nn.L1Loss()
    
    n_angles = len(FIXED_ANGLES)
    coords_sensor = torch.from_numpy(sensor_coords).to(device).float().unsqueeze(0).expand(n_angles, -1, -1) 
    theta_batch = torch.from_numpy(FIXED_ANGLES).to(device).unsqueeze(1).float() 
    gt_batch = torch.from_numpy(sensor_values_all).to(device).float()            
    
    # 将 Mean/Std 转换为正确的 Tensor 形状 (1, 1, 2)
    mean_t = mean.to(device).view(1, 1, 2).float()
    std_t = std.to(device).view(1, 1, 2).float()
    gt_norm = (gt_batch - mean_t) / (std_t + 1e-8)

    Ns, Nrand = coords_sensor.shape[1], 4096
    coords_rand = (torch.rand(n_angles, Nrand, 2, device=device) * 2.0 - 1.0) 
    coords_all = torch.cat([coords_sensor, coords_rand], dim=1)               

    print(f"\n[*] 启动反向优化 (OneCycleLR, Steps={num_steps})...")
    history_loss, history_lr = [], []
    
    pbar = tqdm(range(num_steps), desc="Inversing Z")
    for step in pbar:
        optimizer.zero_grad()
        z_batch = z.expand(n_angles, -1)
        pred_all = phys_model((coords_all, coords_all, (z_batch, theta_batch))) 
        pred_norm = pred_all[:, :Ns, :] 
        
        data_loss = loss_fn(pred_norm, gt_norm)
        loss = data_loss + torch.mean(z ** 2) * CONFIG["reg_weight"]
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        history_loss.append(loss.item())
        history_lr.append(optimizer.param_groups[0]['lr'])
        pbar.set_postfix({'L': f"{loss.item():.4f}"})
        
    return z.detach(), history_loss, history_lr

# ==========================================
# 3. 主流程与画图
# ==========================================
def main():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    deepsdf_ckpt = os.path.normpath(os.path.join(cur_dir, CONFIG["deepsdf_ckpt"]))
    transolver_ckpt = os.path.normpath(os.path.join(cur_dir, CONFIG["transolver_ckpt"]))
    norm_stats_path = os.path.normpath(os.path.join(cur_dir, CONFIG["norm_stats"]))
    output_dir = os.path.normpath(os.path.join(cur_dir, CONFIG["output_dir"]))
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(CONFIG["device"])
    
    # ---- 加载模型与数据 ----
    print("[*] 正在加载 Stable-SDF 和 GI-Transolver 模型...")
    sdf_model = DeepSDFWithPE(latent_dim=64, hidden_dim=256, num_layers=4, num_freqs=6).to(device)
    sdf_model.load_state_dict(torch.load(deepsdf_ckpt, map_location=device)['model'])
    
    phys_model = TransolverModel(
        space_dim=2, fun_dim=0, out_dim=2, n_hidden=256, n_layers=4, n_head=8, 
        dropout=0.0, mlp_ratio=1, slice_num=32, unified_pos=False, ref=8,
        use_theta_in_coord=True, theta_feat_dim=2, use_slice_z_inject=True, z_dim=64,
        z_inject_dropout=0.0, z_inject_layers=-1
    ).to(device)
    phys_model.load_state_dict(torch.load(transolver_ckpt, map_location=device))
    
    print(f"[*] 加载归一化统计量: {norm_stats_path}")
    stats = torch.load(norm_stats_path)
    mean, std = stats['mean'], stats['std']

    # ---- 步骤 1: 生成真值 ----
    print("\n[*] Step 1: 生成随机形状与 FDM 真实物理场...")
    mask_gt = generate_random_shape()
    fields_gt = solve_fdm_multiview(mask_gt)
    
    # ---- 步骤 2: 采样传感器数据 ----
    print("[*] Step 2: 采集稀疏传感器数据...")
    sensor_coords, sensor_vals_all = sample_multiview_sensors(fields_gt, num_sensors=CONFIG["num_sensors"])
    
    # ---- 步骤 3: 反向优化 ----
    z_inverted, loss_hist, lr_hist = inverse_solve_onecycle(
        phys_model, sensor_coords, sensor_vals_all, device, mean, std, num_steps=CONFIG["optim_steps"]
    )
    
    # ---- 步骤 4: 重建形状 ----
    print("\n[*] Step 3: 根据优化出的 z 重建形状并绘制对比图...")
    with torch.no_grad():
        x = np.linspace(-1, 1, 256)
        y = np.linspace(-1, 1, 256)
        xv, yv = np.meshgrid(x, y)
        grid = torch.tensor(np.stack([xv.flatten(), yv.flatten()], axis=1), dtype=torch.float32).to(device).unsqueeze(0)
        
        z_exp = z_inverted.unsqueeze(1).expand(-1, grid.shape[1], -1)
        sdf_pred = sdf_model(grid, z_exp).view(256, 256).cpu().numpy()
        mask_pred = (sdf_pred < 0).astype(float)

    # ---- 步骤 5: 绘图与保存 ----
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 4)
    
    # 图1-3: 形状对比
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(mask_gt, origin='lower', cmap='gray')
    ax1.set_title("GT Shape"); ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mask_pred, origin='lower', cmap='gray')
    ax2.set_title("Reconstructed Shape"); ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    im_err = ax3.imshow(np.abs(mask_gt - mask_pred), origin='lower', cmap='Reds')
    ax3.set_title("Shape Error"); ax3.axis('off')
    plt.colorbar(im_err, ax=ax3, fraction=0.046)
    
    # 图4: 优化过程 Loss 曲线
    ax_loss = fig.add_subplot(gs[0, 3])
    ax_loss.plot(loss_hist, label='Loss', color='blue')
    ax_loss.set_title("Optimization Process")
    ax_loss.set_xlabel("Step"); ax_loss.set_ylabel("L1 Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_lr = ax_loss.twinx()
    ax_lr.plot(lr_hist, label='Learning Rate', color='orange', linestyle='--')
    ax_lr.set_ylabel("Learning Rate")
    lines1, labels1 = ax_loss.get_legend_handles_labels()
    lines2, labels2 = ax_lr.get_legend_handles_labels()
    ax_loss.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # 图5-7: 物理场对比
    ax4 = fig.add_subplot(gs[1, 0])
    u_gt = fields_gt[0]
    vmax = np.max(np.abs(u_gt.real))
    ax4.imshow(u_gt.real, origin='lower', cmap='RdBu', vmin=-vmax, vmax=vmax)
    ax4.set_title("GT Field (Real)")
    
    with torch.no_grad():
        theta_0 = torch.tensor([[FIXED_ANGLES[0]]], dtype=torch.float32).to(device)
        pred_norm = phys_model((grid, grid, (z_inverted, theta_0)))
        mean_t = mean.to(device).view(1, 1, 2)
        std_t = std.to(device).view(1, 1, 2)
        pred_real = pred_norm * std_t + mean_t
        pred_np = pred_real[0].cpu().numpy()
        u_ai = (pred_np[:,0] + 1j*pred_np[:,1]).reshape(256, 256)
        
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(u_ai.real, origin='lower', cmap='RdBu', vmin=-vmax, vmax=vmax)
    ax5.set_title("Predicted Field (Real)")
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(np.abs(u_ai - u_gt), origin='lower', cmap='viridis')
    ax6.set_title("Field Absolute Error")
    
    # 图8: 传感器分布
    ax7 = fig.add_subplot(gs[1, 3])
    ax7.imshow(np.zeros((256,256)), cmap='gray')
    ax7.scatter((sensor_coords[:,0]+1)/2*255, (sensor_coords[:,1]+1)/2*255, s=2, c='cyan')
    ax7.set_title("Sensor Distribution"); ax7.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "inverse_onecycle_result.png")
    plt.savefig(save_path)
    print(f"\n[*] 反演结果图已保存至: {save_path}")

if __name__ == "__main__":
    main()