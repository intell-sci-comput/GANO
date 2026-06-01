"""
2D 机翼任务：反问题（空气动力学形状优化）脚本
说明：本版本已针对开源仓库规范进行重构。采用全局统一的 CONFIG 字典，
调用训练好的 DeepSDF 与 GI-Transolver 代理模型，通过梯度下降优化隐向量 z，
在满足阻力(Cd)约束的前提下，最大化机翼的升力(Cl)。
"""

import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# 将 GANO 仓库根目录加入系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.airfoil.gi_transolver import GITransolver
from src.airfoil.model import DeepSDFWithPE

SMOKE_TEST = os.environ.get("GANO_SMOKE_TEST", "0").lower() in {"1", "true", "yes", "on"}


def env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


# ==========================================
# ======= [全局参数配置字典] ================
# ==========================================
CONFIG = {
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    # --- 权重与数据路径 ---
    "TRANSOLVER_CKPT": os.path.join(project_root, "checkpoints", "airfoil_transolver", "airfoil_transolver_best.pth"),
    "PHYSICS_DATA": os.path.join(project_root, "data", "airfoil", "airfoil_physics_train.pt"),
    "DEEPSDF_CKPT": os.path.join(project_root, "checkpoints", "airfoil_stablesdf", "model_latest.pth"),

    # --- 优化目标超参数 ---
    "ALPHA_DEG": 4.0,           # 攻角 (度)
    "CHORD": 1.0,               # 弦长
    "CD_MAX": 0.020,            # 最大阻力限制
    "LAMBDA_CD": 200.0,         # 阻力惩罚项权重
    "LAMBDA_REG": 1e-4,         # Latent L2 正则化权重

    # --- 优化控制 ---
    "LR": 1e-3,
    "STEPS": 100,
    "SAMPLE_IDX": 0,         # 用作优化起点的初始翼型 ID

    # --- 网格与物理场参数 ---
    "RES": 256,
    "BOX_X": (-0.4, 1.4),
    "BOX_Y": (-0.8, 0.8),
    "RHO_FIX": 1.0,
    "P_FIX": 0.7143,
    "Q_FIX": 0.0050,
    "AIRFOIL_ROI_X": (-0.1, 1.1),
    "AIRFOIL_ROI_Y": (-0.3, 0.3),

    # --- COMSOL 几何导出配置 ---
    "SDF_RES_EXPORT": 512,
    "SDF_X_EXPORT": (-0.5, 1.5),
    "SDF_Y_EXPORT": (-0.5, 0.5),
    "EXPORT_FILENAME": "optimized_airfoil_comsol.txt",

    "USE_AMP": False,
    "AMP_DTYPE": torch.bfloat16,
    "PRED_CHUNK": 65536,

    # --- 上下文点云加密配置 ---
    "CTX_MAX": 100000,
    "NEAR_BAND": 0.02,
    "NEAR_MIN_POINTS": 6000,
    "JITTER_RATIO": 1.0,
    "JITTER_SCALE": 0.01,
    "FAR_STRIDE": 6,

    # --- 输出路径 ---
    "OUT_DIR": os.path.join(project_root, "output", "airfoil_optimization"),
    "SAVE_FIG": True,
    "SAVE_NPZ": True,
}

if SMOKE_TEST:
    CONFIG.update({
        "STEPS": env_int("GANO_AIRFOIL_OPT_SMOKE_STEPS", 1),
        "RES": env_int("GANO_AIRFOIL_OPT_SMOKE_RES", 32),
        "SDF_RES_EXPORT": env_int("GANO_AIRFOIL_OPT_SMOKE_SDF_RES_EXPORT", 32),
        "PRED_CHUNK": env_int("GANO_AIRFOIL_OPT_SMOKE_PRED_CHUNK", 4096),
        "CTX_MAX": env_int("GANO_AIRFOIL_OPT_SMOKE_CTX_MAX", 2048),
        "NEAR_MIN_POINTS": env_int("GANO_AIRFOIL_OPT_SMOKE_NEAR_MIN_POINTS", 128),
        "FAR_STRIDE": env_int("GANO_AIRFOIL_OPT_SMOKE_FAR_STRIDE", 16),
    })

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_stats_and_latents(physics_data_path, deepsdf_ckpt_path, device):
    data = torch.load(physics_data_path, map_location=device)

    stats = data.get("stats", None)
    if isinstance(stats, dict) and ("mean" in stats) and ("std" in stats):
        mean = stats["mean"].view(1, 1, -1).to(device)
        std  = stats["std"].view(1, 1, -1).to(device)
    elif stats is not None and hasattr(stats, "mean") and hasattr(stats, "std"):
        mean = stats.mean.view(1, 1, -1).to(device)
        std  = stats.std.view(1, 1, -1).to(device)
    else:
        mean = torch.zeros(1, 1, 3, device=device)
        std  = torch.ones(1, 1, 3, device=device)

    z_all = None
    for k in ["latents", "z", "codes"]:
        if k in data:
            z_all = data[k].to(device)
            break

    if z_all is None:
        ckpt = torch.load(deepsdf_ckpt_path, map_location="cpu")
        if "latents" in ckpt and isinstance(ckpt["latents"], dict) and "weight" in ckpt["latents"]:
            z_all = ckpt["latents"]["weight"].to(device)
        else:
            z_all = ckpt["latents"]["weight"].to(device)

    return mean, std, z_all

def get_grid(res, x_range, y_range, device):
    x = torch.linspace(x_range[0], x_range[1], res, device=device)
    y = torch.linspace(y_range[0], y_range[1], res, device=device)
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack([gx, gy], dim=-1).reshape(1, -1, 2)  # [1, N, 2]
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
    return coords, dx, dy, extent, x, y

def build_airfoil_roi_mask_torch(x_vec: torch.Tensor, y_vec: torch.Tensor, roi_x, roi_y):
    x_ok = (x_vec >= roi_x[0]) & (x_vec <= roi_x[1])
    y_ok = (y_vec >= roi_y[0]) & (y_vec <= roi_y[1])
    return (y_ok[:, None] & x_ok[None, :])  # [H,W]

def load_deepsdf_decoder(deepsdf_ckpt_path, device):
    ckpt = torch.load(deepsdf_ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    deepsdf = DeepSDFWithPE(latent_dim=64, hidden_dim=256, num_layers=4, num_freqs=6).to(device)
    deepsdf.load_state_dict(state, strict=False)
    deepsdf.eval()
    for p in deepsdf.parameters():
        p.requires_grad_(False)
    return deepsdf

def sdf_on_coords(deepsdf, coords_1xN2, z_1xD):
    z_exp = z_1xD.unsqueeze(1).expand(-1, coords_1xN2.shape[1], -1)
    sdf = deepsdf(coords_1xN2, z_exp)
    return sdf

def export_to_comsol(sdf_field, sdf_extent, filename):
    print(f"\n>> Exporting geometry to {filename} using Matplotlib (Robust Method)...")
    rows, cols = sdf_field.shape
    x_min, x_max, y_min, y_max = sdf_extent

    x = np.linspace(x_min, x_max, cols)
    y = np.linspace(y_min, y_max, rows)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    CS = plt.contour(X, Y, sdf_field, levels=[0])
    all_segments = CS.allsegs[0]
    plt.close(fig)

    if len(all_segments) == 0:
        print("Error: No contour found (Airfoil might have disappeared).")
        return None

    main_segment = max(all_segments, key=lambda seg: len(seg))
    data = main_segment

    if np.linalg.norm(data[0] - data[-1]) > 1e-4:
        data = np.vstack((data, data[0]))

    np.savetxt(filename, data, fmt='%.6f', header='% Coordinates for COMSOL\n% x y', comments='')
    print(f"   Saved {len(data)} points. Ready for COMSOL Import.")
    return data

def compute_forces_cv(u, v, p_gauge, dx, dy, rho, alpha_rad):
    u_L, v_L, p_L = u[:, :, 0],  v[:, :, 0],  p_gauge[:, :, 0]
    u_R, v_R, p_R = u[:, :, -1], v[:, :, -1], p_gauge[:, :, -1]
    u_B, v_B, p_B = u[:, 0, :],  v[:, 0, :],  p_gauge[:, 0, :]
    u_T, v_T, p_T = u[:, -1, :], v[:, -1, :], p_gauge[:, -1, :]

    fx_in  = torch.sum(p_L + rho * u_L**2) * dy
    fx_out = torch.sum(-p_R - rho * u_R**2) * dy
    fx_bot = torch.sum(rho * u_B * v_B) * dx
    fx_top = torch.sum(-rho * u_T * v_T) * dx
    F_x = fx_in + fx_out + fx_bot + fx_top

    fy_in  = torch.sum(rho * u_L * v_L) * dy
    fy_out = torch.sum(-rho * u_R * v_R) * dy
    fy_bot = torch.sum(p_B + rho * v_B**2) * dx
    fy_top = torch.sum(-p_T - rho * v_T**2) * dx
    F_y = fy_in + fy_out + fy_bot + fy_top

    sin_a = torch.sin(alpha_rad)
    cos_a = torch.cos(alpha_rad)
    Lift = -F_x * sin_a + F_y * cos_a
    Drag =  F_x * cos_a + F_y * sin_a
    return Lift, Drag

def predict_query_chunked(model, coords_in, coords_out, z_opt, mean, std, use_amp, amp_dtype, chunk):
    N_out = coords_out.shape[1]
    outs = []
    for s in range(0, N_out, chunk):
        e = min(s + chunk, N_out)
        q = coords_out[:, s:e, :]
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            pred_norm = model((q, coords_in, z_opt))
            pred_phys = pred_norm * (std + 1e-6) + mean
        outs.append(pred_phys)
    return torch.cat(outs, dim=1)

def build_context_points(deepsdf, z_opt, grid_coords, sdf_hw, roi_hw, extent, res, near_band, near_min_points, jitter_ratio, jitter_scale, far_stride, ctx_max):
    device = grid_coords.device
    sdf_flat = sdf_hw.view(-1)
    roi_flat = roi_hw.view(-1)

    inside = (sdf_hw < 0.0) & roi_hw
    outside_flat = (~inside).view(-1)

    W = res
    top    = torch.arange(0, W, device=device)
    bottom = torch.arange((W - 1) * W, W * W, device=device)
    left   = torch.arange(0, W * W, W, device=device)
    right  = torch.arange(W - 1, W * W, W, device=device)
    boundary_idx = torch.unique(torch.cat([top, bottom, left, right], dim=0))

    band = float(near_band)
    near_idx = None
    with torch.no_grad():
        for _ in range(6):
            near_mask = outside_flat & roi_flat & (sdf_flat >= 0.0) & (sdf_flat < band)
            cand = torch.where(near_mask)[0]
            if cand.numel() >= int(near_min_points):
                near_idx = cand
                break
            band *= 1.5
        if near_idx is None:
            near_idx = torch.where(outside_flat & roi_flat & (sdf_flat >= 0.0) & (sdf_flat < band))[0]

    coords_b = grid_coords[:, boundary_idx, :]
    coords_near = torch.empty((1, 0, 2), device=device)
    if near_idx.numel() > 0:
        coords_near = grid_coords[:, near_idx, :]

    coords_jitter = torch.empty((1, 0, 2), device=device)
    if coords_near.shape[1] > 0 and float(jitter_ratio) > 0:
        K = int(coords_near.shape[1] * float(jitter_ratio))
        if K > 0:
            base = coords_near[:, torch.randint(0, coords_near.shape[1], (K,), device=device), :]
            s = float(jitter_scale) if float(jitter_scale) > 0 else (band * 0.5)
            cand = base + torch.randn_like(base) * s

            x_min, x_max, y_min, y_max = extent
            cand[..., 0] = cand[..., 0].clamp(x_min, x_max)
            cand[..., 1] = cand[..., 1].clamp(y_min, y_max)

            with torch.no_grad():
                sdf_c = sdf_on_coords(deepsdf, cand, z_opt)
                if sdf_c.ndim == 3 and sdf_c.shape[-1] == 1:
                    sdf_c = sdf_c[..., 0]

                in_roi = (
                    (cand[..., 0] >= CONFIG["AIRFOIL_ROI_X"][0]) & (cand[..., 0] <= CONFIG["AIRFOIL_ROI_X"][1]) &
                    (cand[..., 1] >= CONFIG["AIRFOIL_ROI_Y"][0]) & (cand[..., 1] <= CONFIG["AIRFOIL_ROI_Y"][1])
                )
                keep = (sdf_c >= 0.0) & in_roi
                keep_idx = torch.where(keep.view(-1))[0]
                if keep_idx.numel() > 0:
                    coords_jitter = cand[:, keep_idx, :]

    far_stride = int(max(1, far_stride))
    with torch.no_grad():
        xs = torch.arange(0, res, far_stride, device=device)
        ys = torch.arange(0, res, far_stride, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        flat = (gy * res + gx).reshape(-1)
        flat = flat[outside_flat[flat]]
        coords_far = grid_coords[:, flat, :] if flat.numel() > 0 else torch.empty((1, 0, 2), device=device)

    coords_in = torch.cat([coords_b, coords_near, coords_jitter, coords_far], dim=1)

    ctx_max = int(ctx_max)
    if ctx_max > 0 and coords_in.shape[1] > ctx_max:
        Nb = coords_b.shape[1]
        if Nb >= ctx_max:
            coords_in = coords_b[:, :ctx_max, :]
        else:
            rest = coords_in[:, Nb:, :]
            budget = ctx_max - Nb
            if rest.shape[1] > budget:
                perm = torch.randperm(rest.shape[1], device=device)[:budget]
                rest = rest[:, perm, :]
            coords_in = torch.cat([coords_b, rest], dim=1)

    return coords_in

def optimize():
    device = CONFIG["DEVICE"]
    ensure_dir(CONFIG["OUT_DIR"])

    model = GITransolver(
        space_dim=2, fun_dim=0, out_dim=3, n_hidden=256, n_layers=5, n_head=8,
        dropout=0.0, mlp_ratio=1, slice_num=32, unified_pos=False, ref=8,
        use_slice_z_add=True, z_dim=64, z_add_dropout=0, z_inject_layers=-1,
    ).to(device)

    ckpt = torch.load(CONFIG["TRANSOLVER_CKPT"], map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    mean, std, all_z = load_stats_and_latents(CONFIG["PHYSICS_DATA"], CONFIG["DEEPSDF_CKPT"], device)
    z_init = all_z[int(CONFIG["SAMPLE_IDX"])].unsqueeze(0)
    z_opt = z_init.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([z_opt], lr=CONFIG["LR"])

    grid_coords, dx, dy, phys_extent, x_vec, y_vec = get_grid(CONFIG["RES"], CONFIG["BOX_X"], CONFIG["BOX_Y"], device)
    res = CONFIG["RES"]
    N = res * res

    deepsdf = load_deepsdf_decoder(CONFIG["DEEPSDF_CKPT"], device)
    roi_hw = build_airfoil_roi_mask_torch(x_vec, y_vec, CONFIG["AIRFOIL_ROI_X"], CONFIG["AIRFOIL_ROI_Y"])
    alpha_rad = torch.tensor(CONFIG["ALPHA_DEG"] * math.pi / 180.0, device=device)

    use_amp = (device.type == "cuda") and bool(CONFIG["USE_AMP"])
    amp_dtype = CONFIG["AMP_DTYPE"]
    chunk = int(CONFIG["PRED_CHUNK"])

    rho, p_inf, q_inf, cd_max = float(CONFIG["RHO_FIX"]), float(CONFIG["P_FIX"]), float(CONFIG["Q_FIX"]), float(CONFIG["CD_MAX"])
    history = {"loss": [], "Cl": [], "Cd": [], "pen_cd": []}

    print("\n>> Start optimization: maximize Cl subject to Cd <= Cd_max (soft hinge penalty)")
    for step in tqdm(range(CONFIG["STEPS"]), desc="Optimizing Shape"):
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            sdf = sdf_on_coords(deepsdf, grid_coords, z_opt)
            if sdf.ndim == 3 and sdf.shape[-1] == 1:
                sdf = sdf[..., 0]
            sdf_hw = sdf.view(res, res)
            inside = (sdf_hw < 0.0) & roi_hw
            outside = ~inside
            outside_idx = torch.where(outside.view(-1))[0]

        coords_in = build_context_points(
            deepsdf=deepsdf, z_opt=z_opt, grid_coords=grid_coords, sdf_hw=sdf_hw, roi_hw=roi_hw,
            extent=phys_extent, res=res, near_band=CONFIG["NEAR_BAND"], near_min_points=CONFIG["NEAR_MIN_POINTS"],
            jitter_ratio=CONFIG["JITTER_RATIO"], jitter_scale=CONFIG["JITTER_SCALE"], far_stride=CONFIG["FAR_STRIDE"], ctx_max=CONFIG["CTX_MAX"],
        )

        coords_out = grid_coords[:, outside_idx, :]
        pred_phys_out = predict_query_chunked(model=model, coords_in=coords_in, coords_out=coords_out, z_opt=z_opt, mean=mean, std=std, use_amp=use_amp, amp_dtype=amp_dtype, chunk=chunk)

        if pred_phys_out.shape[1] != outside_idx.numel():
            raise RuntimeError(f"Prediction count mismatch: pred={pred_phys_out.shape[1]} vs outside_idx={outside_idx.numel()}.")

        pred_phys_full = torch.zeros((1, N, 3), device=device, dtype=pred_phys_out.dtype)
        pred_phys_full.index_copy_(1, outside_idx, pred_phys_out)
        field = pred_phys_full.view(1, res, res, 3)

        u, v, p = field[..., 0], field[..., 1], field[..., 2]
        p_gauge = p - p_inf
        Lift, Drag = compute_forces_cv(u, v, p_gauge, dx, dy, rho, alpha_rad)
        Cl = Lift / (q_inf * CONFIG["CHORD"])
        Cd = Drag / (q_inf * CONFIG["CHORD"])

        pen_cd = torch.relu(Cd - cd_max)
        loss_aero = -Cl + CONFIG["LAMBDA_CD"] * (pen_cd ** 2)
        loss_reg = CONFIG["LAMBDA_REG"] * torch.sum(z_opt ** 2)
        loss = loss_aero + loss_reg

        loss.backward()
        optimizer.step()

        history["loss"].append(float(loss.item()))
        history["Cl"].append(float(Cl.item()))
        history["Cd"].append(float(Cd.item()))
        history["pen_cd"].append(float(pen_cd.item()))

    print(f"\nOptimization Done! Final: Cl={history['Cl'][-1]:.6f}  Cd={history['Cd'][-1]:.6f}  Cd_max={cd_max:.6f}")

    with torch.no_grad():
        sdf = sdf_on_coords(deepsdf, grid_coords, z_opt)
        if sdf.ndim == 3 and sdf.shape[-1] == 1:
            sdf = sdf[..., 0]
        sdf_hw = sdf.view(res, res)
        inside = (sdf_hw < 0.0) & roi_hw
        outside = ~inside
        outside_idx = torch.where(outside.view(-1))[0]
        coords_in = build_context_points(deepsdf=deepsdf, z_opt=z_opt, grid_coords=grid_coords, sdf_hw=sdf_hw, roi_hw=roi_hw, extent=phys_extent, res=res, near_band=CONFIG["NEAR_BAND"], near_min_points=CONFIG["NEAR_MIN_POINTS"], jitter_ratio=CONFIG["JITTER_RATIO"], jitter_scale=CONFIG["JITTER_SCALE"], far_stride=CONFIG["FAR_STRIDE"], ctx_max=CONFIG["CTX_MAX"])
        coords_out = grid_coords[:, outside_idx, :]
        pred_phys_out = predict_query_chunked(model=model, coords_in=coords_in, coords_out=coords_out, z_opt=z_opt, mean=mean, std=std, use_amp=use_amp, amp_dtype=amp_dtype, chunk=chunk)

        pred_phys_full = torch.zeros((1, N, 3), device=device, dtype=pred_phys_out.dtype)
        pred_phys_full.index_copy_(1, outside_idx, pred_phys_out)
        field = pred_phys_full.view(1, res, res, 3)

        u_final = field[0, :, :, 0].float().cpu().numpy()
        p_final = field[0, :, :, 2].float().cpu().numpy()
        airfoil_mask = inside.detach().cpu().numpy().astype(np.uint8)
        sdf_flow = sdf_hw.float().cpu().numpy()

    export_coords, _, _, export_extent, _, _ = get_grid(CONFIG["SDF_RES_EXPORT"], CONFIG["SDF_X_EXPORT"], CONFIG["SDF_Y_EXPORT"], device)
    with torch.no_grad():
        sdf_export = sdf_on_coords(deepsdf, export_coords, z_opt)
        if sdf_export.ndim == 3 and sdf_export.shape[-1] == 1:
            sdf_export = sdf_export[..., 0]
        sdf_export = sdf_export.view(CONFIG["SDF_RES_EXPORT"], CONFIG["SDF_RES_EXPORT"]).float().cpu().numpy()

    H = W = CONFIG["SDF_RES_EXPORT"]
    xs = np.linspace(export_extent[0], export_extent[1], W)
    ys = np.linspace(export_extent[2], export_extent[3], H)
    X, Y = np.meshgrid(xs, ys)
    roi_export = ((X >= CONFIG["AIRFOIL_ROI_X"][0]) & (X <= CONFIG["AIRFOIL_ROI_X"][1]) & (Y >= CONFIG["AIRFOIL_ROI_Y"][0]) & (Y <= CONFIG["AIRFOIL_ROI_Y"][1]))
    sdf_export_roi = sdf_export.copy()
    sdf_export_roi[~roi_export] = 1.0

    comsol_path = os.path.join(CONFIG["OUT_DIR"], CONFIG["EXPORT_FILENAME"])
    contour_xy = export_to_comsol(sdf_export_roi, export_extent, comsol_path)

    if CONFIG["SAVE_NPZ"]:
        np.savez(os.path.join(CONFIG["OUT_DIR"], "optimized_fields.npz"), u=u_final, p=p_final, sdf_flow=sdf_flow, airfoil_mask=airfoil_mask, extent=np.array(phys_extent, dtype=np.float32), history=history)
        torch.save({"z_opt": z_opt.detach().cpu(), "z_init": z_init.detach().cpu()}, os.path.join(CONFIG["OUT_DIR"], "z_opt.pt"))

    def masked_imshow(ax, img, mask, extent, cmap, title, cbar_label):
        m = np.ma.array(img, mask=(mask > 0)) if mask is not None else np.ma.array(img)
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad(color="white")
        im = ax.imshow(m, origin="lower", extent=extent, cmap=cm, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        cb = plt.colorbar(im, ax=ax)
        cb.set_label(cbar_label)

    if CONFIG["SAVE_FIG"]:
        plt.figure(figsize=(6, 3))
        plt.imshow(airfoil_mask.astype(np.float32), origin="lower", extent=phys_extent, cmap="gray")
        plt.title("Airfoil Mask (SDF<0 within ROI, updated z)")
        plt.xlabel("x"); plt.ylabel("y")
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG["OUT_DIR"], "airfoil_mask.png"), dpi=200)
        plt.close()

        if contour_xy is not None:
            plt.figure(figsize=(6, 3))
            plt.plot(contour_xy[:, 0], contour_xy[:, 1], linewidth=2)
            plt.gca().set_aspect("equal", adjustable="box")
            plt.xlim(-0.5, 1.5); plt.ylim(-0.5, 0.5)
            plt.xlabel("x"); plt.ylabel("y")
            plt.tight_layout()
            plt.savefig(os.path.join(CONFIG["OUT_DIR"], "airfoil_contour.png"), dpi=200)
            plt.close()

        fig = plt.figure(figsize=(20, 5))
        ax1 = plt.subplot(1, 3, 1)
        masked_imshow(ax1, u_final, airfoil_mask, phys_extent, "turbo", title=f"Optimized U (Cl={history['Cl'][-1]:.3f}, Cd={history['Cd'][-1]:.4f})", cbar_label="U")
        ax2 = plt.subplot(1, 3, 2)
        masked_imshow(ax2, p_final, airfoil_mask, phys_extent, "magma", title="Optimized Pressure P (masked by SDF<0 in ROI)", cbar_label="p")
        ax3 = plt.subplot(1, 3, 3)
        l1 = ax3.plot(history["Cl"], label="Cl", linewidth=2)
        ax3.set_xlabel("step"); ax3.set_ylabel("Cl"); ax3.grid(True, alpha=0.3)
        ax3b = ax3.twinx()
        l2 = ax3b.plot(history["Cd"], label="Cd", linestyle="--", linewidth=2)
        ax3b.axhline(CONFIG["CD_MAX"], color="k", linestyle=":", label="Cd_max")
        ax3b.set_ylabel("Cd")
        lines = l1 + l2 + ax3b.get_legend_handles_labels()[0]
        labels = [ln.get_label() for ln in (l1 + l2)] + ax3b.get_legend_handles_labels()[1]
        ax3.legend(lines, labels, loc="best")
        ax3.set_title("Cl/Cd History")
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG["OUT_DIR"], "optimized_cl_cd_result.png"), dpi=200)
        plt.close()

    print(f"\nSaved outputs to: {CONFIG['OUT_DIR']}")
    return z_opt.detach()

if __name__ == "__main__":
    optimize()
