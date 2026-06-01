"""
代码说明：
3D 汽车任务的反问题脚本。
加载 Stable-SDF 形状解码器和汽车压力 GI-Transolver，优化指定车辆的 latent code，
在保持镜子/车轮等约束点几何不变的近似零空间内降低预测阻力，并导出优化过程中的 OBJ。
"""

import contextlib
import glob
import json
import os
import random
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import trimesh
from skimage import measure
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.car.gi_transolver import Transolver
from src.car.model import DeepSDFNet


SMOKE_TEST = os.environ.get("GANO_SMOKE_TEST", "0").lower() in {"1", "true", "yes", "on"}


def env_path(name, default):
    return os.environ.get(name, default)


def env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


CONFIG = {
    # --- 默认仓库路径；服务器真实路径可通过 GANO_CAR_OPT_* 环境变量覆盖 ---
    "CAR_ID": "E_S_WW_WM_395",
    "PARTS_ROOT_DIR": env_path("GANO_CAR_OPT_PARTS_ROOT", os.path.join(project_root, "data", "car", "parts")),
    "Z_PATH": env_path(
        "GANO_CAR_OPT_Z_PATH",
        os.path.join(project_root, "checkpoints", "car_training_h800_all", "latents_latest.pth"),
    ),
    "FILE_LIST_PATH": env_path(
        "GANO_CAR_OPT_FILE_LIST_PATH",
        os.path.join(project_root, "checkpoints", "car_training_h800_all", "file_list.json"),
    ),
    "SDF_CKPT": env_path(
        "GANO_CAR_OPT_SDF_CKPT",
        os.path.join(project_root, "checkpoints", "car_training_h800_all", "model_latest.pth"),
    ),
    "PHYSICS_CKPT": env_path(
        "GANO_CAR_OPT_PHYSICS_CKPT",
        os.path.join(project_root, "checkpoints", "car_transolver", "best_model.pth"),
    ),

    # --- 压力和几何归一化 ---
    "GLOBAL_MEAN": -93.427311,
    "GLOBAL_STD": 120.596359,
    "TARGET_SCALE": 1.9,

    # --- Stable-SDF 参数 ---
    "Z_DIM": 256,
    "SDF_HIDDEN_DIM": 512,
    "SDF_NUM_FREQS": 4,

    # --- GI-Transolver 参数，必须与训练一致 ---
    "TRANS_IN_DIM": 6,
    "TRANS_N_LAYERS": 5,
    "TRANS_N_HIDDEN": 256,
    "TRANS_N_HEAD": 8,
    "TRANS_DROPOUT": 0.0,
    "TRANS_MLP_RATIO": 2,
    "TRANS_SLICE_NUM": 32,
    "TRANS_OUT_DIM": 1,
    "TRANS_USE_SLICE_Z_INJECT": True,
    "TRANS_Z_DIM": 256,
    "TRANS_Z_INJECT_DROPOUT": 0.0,
    "TRANS_Z_INJECT_LAYERS": -1,

    # --- 优化超参数 ---
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "SEED": 42,
    "STEPS": 40,
    "LAMBDA_DRAG": 1.0,
    "LAMBDA_REG": 1e-4,
    "GRAD_CLIP": 1.0,

    # --- LBFGS 参数 ---
    "LBFGS_LR": 0.005,
    "LBFGS_MAX_ITER": 2,
    "LBFGS_MAX_EVAL": 4,
    "LBFGS_HISTORY_SIZE": 15,
    "LBFGS_TOLERANCE_GRAD": 1e-7,
    "LBFGS_TOLERANCE_CHANGE": 1e-9,
    "REBUILD_PROJECTOR_EVERY": 1,

    # --- 几何约束和表面采样 ---
    "N_CONSTRAINT_POINTS": 16,
    "PROJECTION_STEPS": 5,
    "TOTAL_SAMPLE_POINTS": 50000,
    "KNN_K": 6,
    "KNN_CHUNK_SIZE": 10000,
    "EXCLUDE_OPTIMIZE_PREFIXES": ["Mirrors", "Underbody", "wheels"],

    # --- SDF 法向计算 ---
    "SDF_NORMAL_CHUNK": 8192,
    "SDF_NORMAL_EPS": 1e-8,
    "SDF_NORMAL_COMPUTE_DTYPE": "fp32",

    # --- 输出 ---
    "SAVE_ROOT": os.path.join(project_root, "output", "car_optimization"),
    "EXP_NAME": "opt_drag_nullspace_transolver_lbfgs",
    "SAVE_INTERVAL": 40,

    # --- Marching Cubes 导出 ---
    "MESH_RESOLUTION": 512,
    "MESH_CHUNK_SIZE": 65536,
}

if SMOKE_TEST:
    CONFIG.update({
        "STEPS": env_int("GANO_CAR_OPT_SMOKE_STEPS", 1),
        "N_CONSTRAINT_POINTS": env_int("GANO_CAR_OPT_SMOKE_CONSTRAINT_POINTS", 4),
        "PROJECTION_STEPS": env_int("GANO_CAR_OPT_SMOKE_PROJECTION_STEPS", 1),
        "TOTAL_SAMPLE_POINTS": env_int("GANO_CAR_OPT_SMOKE_SAMPLE_POINTS", 256),
        "KNN_CHUNK_SIZE": env_int("GANO_CAR_OPT_SMOKE_KNN_CHUNK_SIZE", 256),
        "SDF_NORMAL_CHUNK": env_int("GANO_CAR_OPT_SMOKE_NORMAL_CHUNK", 128),
        "SAVE_INTERVAL": 1,
        "MESH_RESOLUTION": env_int("GANO_CAR_OPT_SMOKE_MESH_RESOLUTION", 32),
        "MESH_CHUNK_SIZE": env_int("GANO_CAR_OPT_SMOKE_MESH_CHUNK_SIZE", 4096),
    })


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def denormalize(tensor, mean, std):
    return tensor * std + mean


def load_state_dict_flexible(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    state = {key.replace("_orig_mod.", ""): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


@torch.no_grad()
def freeze_module(module):
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


class PartBasedDataManager:
    """
    从按车辆 ID 分文件夹保存的 STL parts 中采样优化表面点，并读取对应 latent code。
    """

    def __init__(self, z_path, file_list_path, parts_root_dir):
        self.parts_root_dir = parts_root_dir

        z_data = torch.load(z_path, map_location="cpu")
        if isinstance(z_data, dict) and "weight" in z_data:
            self.all_z = z_data["weight"]
        elif torch.is_tensor(z_data):
            self.all_z = z_data
        elif isinstance(z_data, dict):
            first_key = list(z_data.keys())[0]
            self.all_z = z_data[first_key]
        else:
            raise ValueError(f"Unrecognized z file format: {type(z_data)}")

        with open(file_list_path, "r") as f:
            file_list_dict = json.load(f)

        self.file_paths = []
        if isinstance(file_list_dict, dict):
            for split_name in ("train", "val", "test"):
                if split_name in file_list_dict:
                    self.file_paths.extend(file_list_dict[split_name])
            for key, value in file_list_dict.items():
                if key not in ("train", "val", "test") and isinstance(value, list):
                    self.file_paths.extend(value)
        elif isinstance(file_list_dict, list):
            self.file_paths = file_list_dict
        else:
            raise ValueError(f"Unexpected file list json format: {type(file_list_dict)}")

        if self.all_z.dim() != 2:
            raise ValueError(f"all_z must be [M, zdim], got {tuple(self.all_z.shape)}")
        if len(self.file_paths) != self.all_z.shape[0]:
            print(
                f"Warning: file_paths({len(self.file_paths)}) != latents({self.all_z.shape[0]}). "
                "Check file_list and latent ordering."
            )

    @staticmethod
    def _stem_from_path(path):
        base = os.path.basename(path)
        stem, _ = os.path.splitext(base)
        return stem.strip()

    def _is_optimizable(self, filename):
        return not any(filename.startswith(prefix) for prefix in CONFIG["EXCLUDE_OPTIMIZE_PREFIXES"])

    def find_index_by_car_id(self, car_id):
        for idx, path in enumerate(self.file_paths):
            if self._stem_from_path(path) == car_id:
                return idx
        target = f"{car_id}.npz"
        for idx, path in enumerate(self.file_paths):
            if path.endswith(target):
                return idx
        return -1

    def _get_car_id_and_dir_by_idx(self, idx):
        if idx >= len(self.file_paths):
            raise IndexError("Index out of bounds")
        json_path = self.file_paths[idx]
        car_id = self._stem_from_path(json_path)
        car_part_dir = os.path.join(self.parts_root_dir, car_id)
        if not os.path.exists(car_part_dir):
            raise FileNotFoundError(f"Part dir not found: {car_part_dir}")
        return car_id, car_part_dir, self.all_z[idx]

    def _compute_normalization_params(self, stl_files):
        all_vertices = []
        for stl_path in stl_files:
            try:
                mesh = trimesh.load(stl_path, force="mesh")
                all_vertices.append(mesh.vertices)
            except Exception:
                pass
        if not all_vertices:
            raise ValueError("No vertices loaded from STL files.")

        all_vertices = np.vstack(all_vertices)
        min_coords = np.min(all_vertices, axis=0)
        max_coords = np.max(all_vertices, axis=0)
        centroid = (min_coords + max_coords) / 2.0
        diagonal = np.linalg.norm(max_coords - min_coords)
        if diagonal < 1e-6:
            diagonal = 1.0
        scale_factor = CONFIG["TARGET_SCALE"] / diagonal
        return centroid, scale_factor

    def get_full_car_sample_by_id(self, car_id, num_points):
        idx = self.find_index_by_car_id(car_id)
        if idx < 0:
            raise KeyError(f"car_id='{car_id}' not found in {CONFIG['FILE_LIST_PATH']}")

        found_id, car_part_dir, z = self._get_car_id_and_dir_by_idx(idx)
        if found_id != car_id:
            print(f"Warning: matched idx={idx}, but stem '{found_id}' differs from requested '{car_id}'.")

        stl_files = glob.glob(os.path.join(car_part_dir, "*.stl"))
        centroid, scale_factor = self._compute_normalization_params(stl_files)

        mesh_cache = []
        total_area = 0.0
        for stl_path in stl_files:
            try:
                mesh = trimesh.load(stl_path, force="mesh")
                total_area += mesh.area
                mesh_cache.append((stl_path, mesh))
            except Exception:
                pass
        if total_area <= 0:
            raise ValueError("Total mesh area <= 0, cannot sample.")

        all_points = []
        all_masks = []
        for stl_path, mesh in mesh_cache:
            filename = os.path.basename(stl_path)
            n_points = int(num_points * (mesh.area / total_area))
            if n_points < 8:
                continue
            points, _ = trimesh.sample.sample_surface(mesh, n_points)
            points = (points - centroid) * scale_factor
            is_optimizable = 1.0 if self._is_optimizable(filename) else 0.0
            all_points.append(points.astype(np.float32))
            all_masks.append(np.full((n_points, 1), is_optimizable, dtype=np.float32))

        return {
            "idx": idx,
            "z": z.float(),
            "coords": torch.from_numpy(np.vstack(all_points).astype(np.float32)).float(),
            "masks": torch.from_numpy(np.vstack(all_masks).astype(np.float32)).float(),
            "name": car_id,
        }

    def get_constraint_points_by_id(self, car_id, n_points=64):
        idx = self.find_index_by_car_id(car_id)
        if idx < 0:
            raise KeyError(f"car_id='{car_id}' not found in {CONFIG['FILE_LIST_PATH']}")

        _, car_part_dir, _ = self._get_car_id_and_dir_by_idx(idx)

        all_stls = glob.glob(os.path.join(car_part_dir, "*.stl"))
        centroid, scale_factor = self._compute_normalization_params(all_stls)

        mirror_stls = [path for path in all_stls if "Mirrors" in os.path.basename(path)]
        if not mirror_stls:
            print("Warning: no Mirrors found. Falling back to wheels.")
            mirror_stls = [path for path in all_stls if "wheels" in os.path.basename(path)]
        if not mirror_stls:
            raise FileNotFoundError("No mirrors/wheels STL files found for constraints.")

        combined = trimesh.util.concatenate([trimesh.load(path, force="mesh") for path in mirror_stls])
        points, _ = trimesh.sample.sample_surface(combined, n_points)
        points = (points - centroid) * scale_factor
        return torch.from_numpy(points.astype(np.float32))


def sdf_unit_normals_chunked_no_zgrad(
    sdf_model,
    coords,
    z_copy,
    chunk=8192,
    eps=1e-8,
    compute_dtype=torch.float32,
):
    device = coords.device
    batch_size, num_points, _ = coords.shape

    x_all = coords.reshape(-1, 3).contiguous()
    z_all = z_copy[:, None, :].expand(batch_size, num_points, z_copy.shape[-1]).reshape(-1, z_copy.shape[-1]).contiguous()

    if device.type == "cuda":
        autocast_off = torch.amp.autocast(device_type="cuda", enabled=False)
    else:
        autocast_off = contextlib.nullcontext()

    normals_out = []
    for start in range(0, x_all.shape[0], chunk):
        x = x_all[start : start + chunk].to(device=device, dtype=compute_dtype).detach().requires_grad_(True)
        z_chunk = z_all[start : start + chunk].to(device=device, dtype=compute_dtype).detach()

        with autocast_off, torch.enable_grad():
            sdf = sdf_model(x, z_chunk)
            if sdf.dim() == 2:
                sdf = sdf.squeeze(-1)
            grad = torch.autograd.grad(
                outputs=sdf.sum(),
                inputs=x,
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )[0]

        normal = grad / (grad.norm(dim=-1, keepdim=True) + eps)
        normals_out.append(normal.detach().to(device=device, dtype=coords.dtype))

    return torch.cat(normals_out, dim=0).reshape(batch_size, num_points, 3)


def project_points_to_surface(sdf_model, coords, z_detached, steps, clamp_val=0.95, compute_dtype=torch.float32):
    device = coords.device
    batch_size, num_points, _ = coords.shape

    if device.type == "cuda":
        autocast_off = torch.amp.autocast(device_type="cuda", enabled=False)
    else:
        autocast_off = contextlib.nullcontext()

    for _ in range(steps):
        coords_var = coords.detach().clone().to(dtype=compute_dtype).requires_grad_(True)
        x_flat = coords_var.view(-1, 3)
        z_flat = z_detached[:, None, :].expand(batch_size, num_points, z_detached.shape[-1]).reshape(
            -1, z_detached.shape[-1]
        )

        with autocast_off, torch.enable_grad():
            sdf = sdf_model(x_flat, z_flat)
            if sdf.dim() == 2:
                sdf = sdf.squeeze(-1)
            grad = torch.autograd.grad(
                outputs=sdf.sum(),
                inputs=x_flat,
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )[0]

        sdf = sdf.view(batch_size, num_points, 1).to(dtype=coords.dtype)
        grad = grad.view(batch_size, num_points, 3).to(dtype=coords.dtype)
        coords = coords.detach() - sdf.detach() * grad.detach()
        coords = torch.clamp(coords, -clamp_val, clamp_val)

    return coords.detach()


@torch.no_grad()
def compute_point_areas_chunked(coords, k=6, chunk_size=2000):
    _, num_points, _ = coords.shape
    areas = []
    for start in range(0, num_points, chunk_size):
        chunk_coords = coords[:, start : start + chunk_size, :]
        dists = torch.cdist(chunk_coords, coords)
        topk_vals, _ = dists.topk(k + 1, dim=-1, largest=False)
        radius = topk_vals[..., 1:].mean(dim=-1, keepdim=True)
        d_area = torch.pi * ((radius / 2.0) ** 2)
        areas.append(d_area)
    return torch.cat(areas, dim=1)


class NullSpaceProjector:
    def __init__(self, sdf_model, z_init, constraint_coords):
        self.device = z_init.device
        self.z_dim = z_init.shape[1]
        self.P = self._build_P(sdf_model, z_init.detach(), constraint_coords.detach().to(self.device))

    def _build_P(self, model, z, coords):
        num_constraints = coords.shape[0]
        z_var = z.clone().detach().requires_grad_(True)
        z_in = z_var.expand(num_constraints, -1)
        sdf = model(coords, z_in)
        if sdf.dim() == 2:
            sdf = sdf.squeeze(-1)

        jacobian_rows = []
        for idx in range(num_constraints):
            if z_var.grad is not None:
                z_var.grad.zero_()
            sdf[idx].backward(retain_graph=True)
            jacobian_rows.append(z_var.grad.detach().clone())
        jacobian = torch.cat(jacobian_rows, dim=0)

        jacobian_pinv = torch.linalg.pinv(jacobian)
        return (torch.eye(self.z_dim, device=self.device) - (jacobian_pinv @ jacobian)).detach()

    @torch.no_grad()
    def project_grad(self, grad):
        grad_col = grad.view(self.z_dim, 1)
        projected = self.P @ grad_col
        return projected.view_as(grad)


@torch.no_grad()
def save_geometry_obj(step, sdf_model, z_detached, save_dir, res=256, chunk_size=65536):
    os.makedirs(save_dir, exist_ok=True)
    device = z_detached.device

    coords = np.linspace(-1.0, 1.0, res, dtype=np.float32)
    grid_x, grid_y, grid_z = np.meshgrid(coords, coords, coords, indexing="ij")
    points = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1).astype(np.float32)
    points_t = torch.from_numpy(points).to(device)

    z_expanded = z_detached.expand(points_t.shape[0], -1)

    sdf_vals = []
    for start in range(0, points_t.shape[0], chunk_size):
        sdf_chunk = sdf_model(points_t[start : start + chunk_size], z_expanded[start : start + chunk_size])
        sdf_vals.append(sdf_chunk.detach().float().cpu().numpy())
    sdf_grid = np.concatenate(sdf_vals, axis=0).reshape(res, res, res)

    if sdf_grid.min() > 0 or sdf_grid.max() < 0:
        level = float(np.clip(0.0, sdf_grid.min(), sdf_grid.max()))
    else:
        level = 0.0

    verts, faces, normals, _ = measure.marching_cubes(sdf_grid, level=level)
    verts = verts * (2.0 / (res - 1)) - 1.0

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)
    mesh.export(os.path.join(save_dir, f"shape_{step:04d}.obj"))


@torch.no_grad()
def eval_drag_on_current_surface(sdf_model, transolver, coords, z_detached, masks, d_area, sdf_dtype):
    normals = sdf_unit_normals_chunked_no_zgrad(
        sdf_model=sdf_model,
        coords=coords,
        z_copy=z_detached.clone(),
        chunk=CONFIG["SDF_NORMAL_CHUNK"],
        eps=CONFIG["SDF_NORMAL_EPS"],
        compute_dtype=sdf_dtype,
    ).detach()

    x6 = torch.cat([coords, normals], dim=-1)
    normal_x = normals[..., 0:1]

    pred_p_norm = transolver(x6, z=z_detached)
    pred_p_real = denormalize(pred_p_norm, CONFIG["GLOBAL_MEAN"], CONFIG["GLOBAL_STD"])
    raw_force_x = pred_p_real * normal_x * d_area
    drag = -1.0 * torch.sum(raw_force_x * masks)
    return float(drag.item())


def build_models(device):
    sdf_model = DeepSDFNet(
        latent_size=CONFIG["Z_DIM"],
        hidden_dim=CONFIG["SDF_HIDDEN_DIM"],
        num_freqs=CONFIG["SDF_NUM_FREQS"],
    ).to(device)

    transolver = Transolver(
        in_dim=CONFIG["TRANS_IN_DIM"],
        n_layers=CONFIG["TRANS_N_LAYERS"],
        n_hidden=CONFIG["TRANS_N_HIDDEN"],
        n_head=CONFIG["TRANS_N_HEAD"],
        dropout=CONFIG["TRANS_DROPOUT"],
        mlp_ratio=CONFIG["TRANS_MLP_RATIO"],
        slice_num=CONFIG["TRANS_SLICE_NUM"],
        out_dim=CONFIG["TRANS_OUT_DIM"],
        act="gelu",
        use_slice_z_inject=CONFIG["TRANS_USE_SLICE_Z_INJECT"],
        z_dim=CONFIG["TRANS_Z_DIM"],
        z_inject_dropout=CONFIG["TRANS_Z_INJECT_DROPOUT"],
        z_inject_layers=CONFIG["TRANS_Z_INJECT_LAYERS"],
    ).to(device)

    if not os.path.exists(CONFIG["SDF_CKPT"]):
        raise FileNotFoundError(CONFIG["SDF_CKPT"])
    if not os.path.exists(CONFIG["PHYSICS_CKPT"]):
        raise FileNotFoundError(CONFIG["PHYSICS_CKPT"])

    load_state_dict_flexible(sdf_model, CONFIG["SDF_CKPT"], device)
    load_state_dict_flexible(transolver, CONFIG["PHYSICS_CKPT"], device)
    freeze_module(sdf_model)
    freeze_module(transolver)
    return sdf_model, transolver


def optimize():
    set_seed(CONFIG["SEED"])
    device = torch.device(CONFIG["DEVICE"])
    sdf_dtype = torch.float64 if CONFIG["SDF_NORMAL_COMPUTE_DTYPE"].lower() == "fp64" else torch.float32

    print(f"Device: {device}")
    print(f"Optimizing car id: {CONFIG['CAR_ID']}")

    sdf_model, transolver = build_models(device)
    print("Models loaded and frozen.")

    data_manager = PartBasedDataManager(CONFIG["Z_PATH"], CONFIG["FILE_LIST_PATH"], CONFIG["PARTS_ROOT_DIR"])
    sample = data_manager.get_full_car_sample_by_id(CONFIG["CAR_ID"], CONFIG["TOTAL_SAMPLE_POINTS"])
    constraint_coords = data_manager.get_constraint_points_by_id(
        CONFIG["CAR_ID"], n_points=CONFIG["N_CONSTRAINT_POINTS"]
    ).to(device)

    z_init = sample["z"].unsqueeze(0).to(device)
    z_opt = z_init.clone().detach().requires_grad_(True)
    coords = sample["coords"].unsqueeze(0).to(device)
    masks = sample["masks"].unsqueeze(0).to(device)

    print(f"Target={sample['name']} idx={sample['idx']} coords={tuple(coords.shape)} z={tuple(z_opt.shape)}")

    optimizer = torch.optim.LBFGS(
        [z_opt],
        lr=CONFIG["LBFGS_LR"],
        max_iter=CONFIG["LBFGS_MAX_ITER"],
        max_eval=CONFIG["LBFGS_MAX_EVAL"],
        history_size=CONFIG["LBFGS_HISTORY_SIZE"],
        tolerance_grad=CONFIG["LBFGS_TOLERANCE_GRAD"],
        tolerance_change=CONFIG["LBFGS_TOLERANCE_CHANGE"],
        line_search_fn="strong_wolfe",
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(CONFIG["SAVE_ROOT"], f"{CONFIG['EXP_NAME']}_{CONFIG['CAR_ID']}_{timestamp}")
    obj_dir = os.path.join(save_dir, "objs")
    os.makedirs(obj_dir, exist_ok=True)

    print("Saving initial mesh...")
    save_geometry_obj(
        step=0,
        sdf_model=sdf_model,
        z_detached=z_init.detach(),
        save_dir=obj_dir,
        res=CONFIG["MESH_RESOLUTION"],
        chunk_size=CONFIG["MESH_CHUNK_SIZE"],
    )

    history_drag = []
    history_loss = []
    projector = None

    pbar = tqdm(range(1, CONFIG["STEPS"] + 1), desc="Optimize")
    for step in pbar:
        if projector is None or step % CONFIG["REBUILD_PROJECTOR_EVERY"] == 0:
            projector = NullSpaceProjector(
                sdf_model=sdf_model,
                z_init=z_opt.detach(),
                constraint_coords=constraint_coords,
            )

        d_area_fixed = compute_point_areas_chunked(
            coords, k=CONFIG["KNN_K"], chunk_size=CONFIG["KNN_CHUNK_SIZE"]
        ).detach()

        normals_fixed = sdf_unit_normals_chunked_no_zgrad(
            sdf_model=sdf_model,
            coords=coords,
            z_copy=z_opt.detach().clone(),
            chunk=CONFIG["SDF_NORMAL_CHUNK"],
            eps=CONFIG["SDF_NORMAL_EPS"],
            compute_dtype=sdf_dtype,
        ).detach()

        x6_fixed = torch.cat([coords, normals_fixed], dim=-1).detach()
        normal_x_fixed = normals_fixed[..., 0:1].detach()
        projector_fixed = projector

        def closure():
            optimizer.zero_grad(set_to_none=True)

            pred_p_norm = transolver(x6_fixed, z=z_opt)
            pred_p_real = denormalize(pred_p_norm, CONFIG["GLOBAL_MEAN"], CONFIG["GLOBAL_STD"])
            raw_force_x = pred_p_real * normal_x_fixed * d_area_fixed
            drag = -1.0 * torch.sum(raw_force_x * masks)

            reg = torch.mean((z_opt - z_init) ** 2)
            loss = CONFIG["LAMBDA_DRAG"] * drag + CONFIG["LAMBDA_REG"] * reg
            loss.backward()

            if z_opt.grad is not None:
                z_opt.grad.data = projector_fixed.project_grad(z_opt.grad.data)
                torch.nn.utils.clip_grad_norm_([z_opt], CONFIG["GRAD_CLIP"])

            return loss

        loss = optimizer.step(closure)

        coords = project_points_to_surface(
            sdf_model=sdf_model,
            coords=coords,
            z_detached=z_opt.detach(),
            steps=CONFIG["PROJECTION_STEPS"],
            clamp_val=0.95,
            compute_dtype=sdf_dtype,
        )

        d_area_eval = compute_point_areas_chunked(
            coords, k=CONFIG["KNN_K"], chunk_size=CONFIG["KNN_CHUNK_SIZE"]
        ).detach()

        drag_val = eval_drag_on_current_surface(
            sdf_model=sdf_model,
            transolver=transolver,
            coords=coords,
            z_detached=z_opt.detach(),
            masks=masks,
            d_area=d_area_eval,
            sdf_dtype=sdf_dtype,
        )
        loss_val = float(loss.item())
        history_drag.append(drag_val)
        history_loss.append(loss_val)

        pbar.set_postfix({"drag": f"{drag_val:.4f}", "loss": f"{loss_val:.4f}"})

        if step % CONFIG["SAVE_INTERVAL"] == 0 or step == CONFIG["STEPS"]:
            save_geometry_obj(
                step=step,
                sdf_model=sdf_model,
                z_detached=z_opt.detach(),
                save_dir=obj_dir,
                res=CONFIG["MESH_RESOLUTION"],
                chunk_size=CONFIG["MESH_CHUNK_SIZE"],
            )
            torch.save(
                {
                    "car_id": CONFIG["CAR_ID"],
                    "idx": sample["idx"],
                    "step": step,
                    "z_opt": z_opt.detach().cpu(),
                    "drag": drag_val,
                    "loss": loss_val,
                    "config": CONFIG,
                },
                os.path.join(save_dir, "latest_opt.pth"),
            )

    torch.save(z_opt.detach().cpu(), os.path.join(save_dir, "optimized_z.pt"))
    np.save(os.path.join(save_dir, "drag_history.npy"), np.asarray(history_drag, dtype=np.float32))
    np.save(os.path.join(save_dir, "loss_history.npy"), np.asarray(history_loss, dtype=np.float32))

    print(f"Done. saved to: {save_dir}")


if __name__ == "__main__":
    try:
        optimize()
    except Exception as exc:
        print(f"Fatal error: {exc}")
        traceback.print_exc()
        sys.exit(1)
