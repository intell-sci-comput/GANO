"""
代码说明：
3D 汽车任务的 GI-Transolver 压力场训练脚本。
读取汽车压力场预处理 npz、Stable-SDF latent code 与 SDF 模型权重，
用 SDF 梯度法向量和坐标组成 6 维输入训练压力场代理模型。
"""

import contextlib
import datetime
import glob
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.car.gi_transolver import Transolver
from src.car.model import DeepSDFNet


def first_existing_path(env_name, *candidates):
    """
    支持在 H800 和 zhangrui A100 间切换路径。
    设置 env_name 环境变量时优先使用环境变量；否则返回第一个存在的候选路径。
    """
    override = os.environ.get(env_name)
    if override:
        return override
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return candidates[0] if candidates else ""


def env_int(env_name, default):
    value = os.environ.get(env_name)
    return int(value) if value is not None and value != "" else default


def env_float(env_name, default):
    value = os.environ.get(env_name)
    return float(value) if value is not None and value != "" else default


CONFIG = {
    # --- 数据与权重路径 ---
    # 优先适配 H800；如果需要手动指定，可设置对应 GANO_CAR_* 环境变量。
    "PRESSURE_DIR": first_existing_path(
        "GANO_CAR_PRESSURE_DIR",
        "/mnt/sunguoze/processed_data/car_pressure_all",
        "/home/zhangrui/zhangruiC/car/pressure_all",
    ),
    "Z_PATH": first_existing_path(
        "GANO_CAR_Z_PATH",
        os.path.join(project_root, "checkpoints", "car_training_h800_all", "latents_latest.pth"),
        "/home/sunguoze/GANO/checkpoints/car_training_h800_all/latents_latest.pth",
        "/home/zhangrui/zhangruiC/car/latent/car_training_h800_all/latents_latest.pth",
        "/home/zhangrui/sunguoze233/trained_ds_all/latents_latest.pth",
    ),
    "Z_JSON": first_existing_path(
        "GANO_CAR_Z_JSON",
        os.path.join(project_root, "checkpoints", "car_training_h800_all", "file_list.json"),
        "/home/sunguoze/GANO/checkpoints/car_training_h800_all/file_list.json",
        "/home/zhangrui/zhangruiC/car/latent/car_training_h800_all/file_list.json",
        "/home/zhangrui/sunguoze233/trained_ds_all/file_list.json",
    ),
    "TRAIN_LIST": first_existing_path(
        "GANO_CAR_TRAIN_LIST",
        os.path.join(project_root, "data", "car", "split", "train.txt"),
        "/home/zhangrui/zhangruiC/car/split/train.txt",
    ),
    "VAL_LIST": first_existing_path(
        "GANO_CAR_VAL_LIST",
        os.path.join(project_root, "data", "car", "split", "test.txt"),
        "/home/zhangrui/zhangruiC/car/split/test.txt",
    ),
    "SDF_CKPT": first_existing_path(
        "GANO_CAR_SDF_CKPT",
        os.path.join(project_root, "checkpoints", "car_training_h800_all", "model_latest.pth"),
        "/home/sunguoze/GANO/checkpoints/car_training_h800_all/model_latest.pth",
        "/home/zhangrui/zhangruiC/car/latent/car_training_h800_all/model_latest.pth",
        "/home/zhangrui/sunguoze233/trained_ds_all/model_latest.pth",
    ),

    # --- 仓库规范输出路径 ---
    "SAVE_ROOT": os.path.join(project_root, "checkpoints", "car_transolver"),
    "EXP_NAME": "transolver_sdf_normals",

    # --- Stable-SDF 参数，必须与 H800 训练权重一致 ---
    "SDF_LATENT_SIZE": 256,
    "SDF_HIDDEN_DIM": 512,
    "SDF_NUM_FREQS": 4,

    # --- GI-Transolver 结构 ---
    "IN_DIM": 6,
    "N_LAYERS": 5,
    "N_HIDDEN": 256,
    "N_HEAD": 8,
    "DROPOUT": 0.0,
    "MLP_RATIO": 2,
    "SLICE_NUM": 32,
    "OUT_DIM": 1,
    "USE_SLICE_Z_INJECT": True,
    "Z_DIM": 256,
    "Z_INJECT_DROPOUT": 0.0,
    "Z_INJECT_LAYERS": -1,

    # --- 训练超参数 ---
    "NUM_POINTS": env_int("GANO_CAR_NUM_POINTS", 50000),
    "BATCH_SIZE": env_int("GANO_CAR_BATCH_SIZE", 16),
    "EPOCHS": env_int("GANO_CAR_EPOCHS", 200),
    "LR": env_float("GANO_CAR_LR", 1e-3),
    "WEIGHT_DECAY": env_float("GANO_CAR_WEIGHT_DECAY", 0.0),
    "NUM_WORKERS": env_int("GANO_CAR_NUM_WORKERS", 16),
    "SEED": 42,
    "WARMUP_EPOCHS": 10,
    "MIN_LR": 1e-5,
    "GRAD_CLIP": 1.0,

    # --- 压力归一化后的数值稳定设置 ---
    "CLAMP_MIN": -8.0,
    "CLAMP_MAX": 5.0,

    # --- AMP 与法向计算 ---
    "USE_AMP": True,
    "AMP_DTYPE": "bf16",
    "SDF_NORMAL_CHUNK": 8192,
    "SDF_NORMAL_EPS": 1e-8,
    "SDF_NORMAL_COMPUTE_DTYPE": "fp32",

    # --- 推理耗时和显存基准，可训练时关闭 ---
    "BENCH_ENABLE": False,
    "BENCH_EVERY": 1,
    "BENCH_WARMUP": 10,
    "BENCH_ITERS": 50,
}


def get_id_from_path(path_or_name):
    base = os.path.basename(path_or_name)
    stem, _ = os.path.splitext(base)
    return stem.strip()


def load_ids_from_txt(txt_path):
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"List file not found: {txt_path}")
    ids = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(get_id_from_path(line))
    return ids


class CarPressureDataset(Dataset):
    """
    对齐压力场 npz 与 Stable-SDF latent code。
    npz 需要包含 coords: [N,3] 和 data: [N,1]。
    """

    def __init__(self, id_source, pressure_dir, z_pth_path, z_json_path, num_points=2048):
        self.num_points = num_points
        self.samples = []

        if isinstance(id_source, str):
            self.target_ids = load_ids_from_txt(id_source)
        elif isinstance(id_source, list):
            self.target_ids = [get_id_from_path(x) for x in id_source]
        else:
            raise ValueError("id_source must be a list or txt path")

        print(f"Loading Z tensor from {z_pth_path}...")
        raw_data = torch.load(z_pth_path, map_location="cpu")
        if isinstance(raw_data, dict) and "weight" in raw_data:
            self.z_tensor = raw_data["weight"]
        elif isinstance(raw_data, torch.Tensor):
            self.z_tensor = raw_data
        else:
            keys = raw_data.keys() if isinstance(raw_data, dict) else "not a dict"
            raise ValueError(f"Unknown latent format in {z_pth_path}. Keys: {keys}")

        print(f"Loading Z mapping from {z_json_path}...")
        with open(z_json_path, "r") as f:
            z_paths_dict = json.load(f)

        all_z_paths = []
        if isinstance(z_paths_dict, list):
            all_z_paths = z_paths_dict
        elif isinstance(z_paths_dict, dict):
            for key in z_paths_dict:
                value = z_paths_dict[key]
                if isinstance(value, list):
                    all_z_paths.extend(value)
        else:
            raise ValueError(f"Unsupported z json format: {type(z_paths_dict)}")

        self.id_to_z_idx = {}
        for idx, path in enumerate(all_z_paths):
            self.id_to_z_idx[get_id_from_path(path)] = idx

        print(f"Scanning pressure files in {pressure_dir}...")
        self.id_to_pressure_path = {}
        pressure_files = glob.glob(os.path.join(pressure_dir, "**", "*.npz"), recursive=True)
        for path in pressure_files:
            self.id_to_pressure_path[get_id_from_path(path)] = path

        print(f"Aligning dataset for {len(self.target_ids)} requested ids...")
        for car_id in self.target_ids:
            if car_id not in self.id_to_z_idx or car_id not in self.id_to_pressure_path:
                continue
            z_idx = self.id_to_z_idx[car_id]
            if z_idx >= self.z_tensor.shape[0]:
                print(f"Warning: ID {car_id} maps to invalid z index {z_idx}. Skipped.")
                continue
            self.samples.append(
                {
                    "id": car_id,
                    "z_idx": z_idx,
                    "p_path": self.id_to_pressure_path[car_id],
                }
            )

        print(f"Dataset ready: {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        z_vec = self.z_tensor[sample["z_idx"]]

        try:
            npz_data = np.load(sample["p_path"])
            coords = npz_data["coords"]
            pressure = npz_data["data"]

            total_points = coords.shape[0]
            if total_points >= self.num_points:
                choice_idx = np.random.choice(total_points, self.num_points, replace=False)
            else:
                choice_idx = np.random.choice(total_points, self.num_points, replace=True)

            return {
                "z": z_vec.float(),
                "coords": torch.from_numpy(coords[choice_idx]).float(),
                "pressure": torch.from_numpy(pressure[choice_idx]).float(),
                "id": sample["id"],
            }
        except Exception as exc:
            print(f"Error loading {sample['id']}: {exc}")
            return {
                "z": torch.zeros_like(z_vec).float(),
                "coords": torch.zeros(self.num_points, 3),
                "pressure": torch.zeros(self.num_points, 1),
                "id": "error",
            }


class Logger:
    def __init__(self, save_dir, name="car_transolver"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers.clear()

        fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        fh = logging.FileHandler(os.path.join(save_dir, "train.log"), mode="w")
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        self.logger.addHandler(ch)

        self.info(f"Saving to: {save_dir}")

    def info(self, msg):
        self.logger.info(msg)

    def close(self):
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calc_relative_l2(pred, target, eps=1e-6):
    diff = torch.norm(pred - target, p=2, dim=1)
    denom = torch.norm(target, p=2, dim=1)
    return torch.mean(diff / (denom + eps))


def calc_relative_l1(pred, target, eps=1e-6):
    diff = torch.sum(torch.abs(pred - target), dim=1)
    denom = torch.sum(torch.abs(target), dim=1)
    return torch.mean(diff / (denom + eps))


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


def sdf_unit_normals_chunked(
    sdf_model,
    coords,
    z,
    chunk=8192,
    eps=1e-8,
    compute_dtype=torch.float32,
):
    """
    计算单位 SDF 法向量：normal = grad_x SDF / ||grad_x SDF||。
    只对临时坐标变量求导，SDF 参数和 z 不参与更新。
    """
    device = coords.device
    batch_size, num_points, _ = coords.shape

    x_all = coords.reshape(-1, 3).contiguous()
    z = z.detach()
    z_all = z[:, None, :].expand(batch_size, num_points, z.shape[-1]).reshape(-1, z.shape[-1]).contiguous()

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


def make_autocast(device, amp_dtype):
    return torch.amp.autocast(
        device_type="cuda",
        dtype=amp_dtype,
        enabled=(CONFIG["USE_AMP"] and device.type == "cuda"),
    )


def train_one_epoch(model, sdf_model, loader, optimizer, criterion, device, epoch, scaler, amp_dtype, sdf_compute_dtype):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in pbar:
        z = batch["z"].to(device, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        target = batch["pressure"].to(device, non_blocking=True)

        if target.dim() == 2:
            target = target.unsqueeze(-1)
        target = torch.clamp(target, CONFIG["CLAMP_MIN"], CONFIG["CLAMP_MAX"])

        optimizer.zero_grad(set_to_none=True)

        normals = sdf_unit_normals_chunked(
            sdf_model=sdf_model,
            coords=coords,
            z=z,
            chunk=CONFIG["SDF_NORMAL_CHUNK"],
            eps=CONFIG["SDF_NORMAL_EPS"],
            compute_dtype=sdf_compute_dtype,
        )
        x6 = torch.cat([coords, normals], dim=-1)

        with make_autocast(device, amp_dtype):
            pred = model(x6, z=z)
            pred = torch.clamp(pred, CONFIG["CLAMP_MIN"], CONFIG["CLAMP_MAX"])
            loss = criterion(pred, target)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["GRAD_CLIP"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["GRAD_CLIP"])
            optimizer.step()

        value = float(loss.item())
        total_loss += value
        pbar.set_postfix(loss=f"{value:.6f}")

        del pred, loss, normals, x6

    return total_loss / max(1, len(loader))


def validate(model, sdf_model, loader, criterion, device, epoch, amp_dtype, sdf_compute_dtype):
    model.eval()
    total_loss, total_l2, total_l1 = 0.0, 0.0, 0.0

    for batch in tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False):
        z = batch["z"].to(device, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        target = batch["pressure"].to(device, non_blocking=True)

        if target.dim() == 2:
            target = target.unsqueeze(-1)
        target = torch.clamp(target, CONFIG["CLAMP_MIN"], CONFIG["CLAMP_MAX"])

        normals = sdf_unit_normals_chunked(
            sdf_model=sdf_model,
            coords=coords,
            z=z,
            chunk=CONFIG["SDF_NORMAL_CHUNK"],
            eps=CONFIG["SDF_NORMAL_EPS"],
            compute_dtype=sdf_compute_dtype,
        )
        x6 = torch.cat([coords, normals], dim=-1)

        with torch.no_grad(), make_autocast(device, amp_dtype):
            pred = model(x6, z=z)
            pred = torch.clamp(pred, CONFIG["CLAMP_MIN"], CONFIG["CLAMP_MAX"])
            loss = criterion(pred, target)

        pred_d = pred.detach()
        target_d = target.detach()
        total_loss += float(loss.item())
        total_l2 += float(calc_relative_l2(pred_d, target_d).item())
        total_l1 += float(calc_relative_l1(pred_d, target_d).item())

        del pred, pred_d, target_d, loss, normals, x6

    count = max(1, len(loader))
    return total_loss / count, total_l2 / count, total_l1 / count


def take_one_sample(batch):
    return {
        "z": batch["z"][0:1].contiguous(),
        "coords": batch["coords"][0:1].contiguous(),
    }


@torch.no_grad()
def benchmark_infer_ms_per_sample_fullpipe(model, sdf_model, one_batch, device, amp_dtype, warmup, iters, sdf_compute_dtype):
    model.eval()
    sdf_model.eval()
    z = one_batch["z"].to(device, non_blocking=True)
    coords = one_batch["coords"].to(device, non_blocking=True)

    def forward_once():
        normals = sdf_unit_normals_chunked(
            sdf_model=sdf_model,
            coords=coords,
            z=z,
            chunk=CONFIG["SDF_NORMAL_CHUNK"],
            eps=CONFIG["SDF_NORMAL_EPS"],
            compute_dtype=sdf_compute_dtype,
        )
        x6 = torch.cat([coords, normals], dim=-1)
        with make_autocast(device, amp_dtype):
            _ = model(x6, z=z)
        del normals, x6, _

    if device.type != "cuda":
        for _ in range(max(1, warmup)):
            forward_once()
        start = time.perf_counter()
        for _ in range(max(1, iters)):
            forward_once()
        return (time.perf_counter() - start) * 1000.0 / max(1, iters), 0.0, 0.0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    for _ in range(max(1, warmup)):
        forward_once()
    torch.cuda.synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(max(1, iters)):
        forward_once()
    end_event.record()
    torch.cuda.synchronize(device)

    ms_per_sample = start_event.elapsed_time(end_event) / max(1, iters)
    peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)
    return ms_per_sample, peak_alloc_mb, peak_reserved_mb


def main():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    set_seed(CONFIG["SEED"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if CONFIG["AMP_DTYPE"].lower() == "bf16" else torch.float16
    sdf_compute_dtype = torch.float64 if CONFIG["SDF_NORMAL_COMPUTE_DTYPE"].lower() == "fp64" else torch.float32

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(CONFIG["SAVE_ROOT"], f"{CONFIG['EXP_NAME']}_{timestamp}")
    logger = Logger(save_dir)
    logger.info(f"Device: {device}")
    logger.info(f"CONFIG: {json.dumps(CONFIG, indent=2, ensure_ascii=False)}")

    train_ds = CarPressureDataset(
        id_source=CONFIG["TRAIN_LIST"],
        pressure_dir=CONFIG["PRESSURE_DIR"],
        z_pth_path=CONFIG["Z_PATH"],
        z_json_path=CONFIG["Z_JSON"],
        num_points=CONFIG["NUM_POINTS"],
    )
    val_ds = CarPressureDataset(
        id_source=CONFIG["VAL_LIST"],
        pressure_dir=CONFIG["PRESSURE_DIR"],
        z_pth_path=CONFIG["Z_PATH"],
        z_json_path=CONFIG["Z_JSON"],
        num_points=CONFIG["NUM_POINTS"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True,
        num_workers=CONFIG["NUM_WORKERS"],
        pin_memory=True,
        persistent_workers=(CONFIG["NUM_WORKERS"] > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=False,
        num_workers=CONFIG["NUM_WORKERS"],
        pin_memory=True,
        persistent_workers=(CONFIG["NUM_WORKERS"] > 0),
    )

    sdf_model = DeepSDFNet(
        latent_size=CONFIG["SDF_LATENT_SIZE"],
        hidden_dim=CONFIG["SDF_HIDDEN_DIM"],
        num_freqs=CONFIG["SDF_NUM_FREQS"],
    ).to(device)
    if not os.path.exists(CONFIG["SDF_CKPT"]):
        raise FileNotFoundError(f"SDF checkpoint not found: {CONFIG['SDF_CKPT']}")
    load_state_dict_flexible(sdf_model, CONFIG["SDF_CKPT"], device)
    freeze_module(sdf_model)
    logger.info("Stable-SDF loaded and frozen.")

    model = Transolver(
        in_dim=CONFIG["IN_DIM"],
        n_layers=CONFIG["N_LAYERS"],
        n_hidden=CONFIG["N_HIDDEN"],
        n_head=CONFIG["N_HEAD"],
        dropout=CONFIG["DROPOUT"],
        mlp_ratio=CONFIG["MLP_RATIO"],
        slice_num=CONFIG["SLICE_NUM"],
        out_dim=CONFIG["OUT_DIM"],
        act="gelu",
        use_slice_z_inject=CONFIG["USE_SLICE_Z_INJECT"],
        z_dim=CONFIG["Z_DIM"],
        z_inject_dropout=CONFIG["Z_INJECT_DROPOUT"],
        z_inject_layers=CONFIG["Z_INJECT_LAYERS"],
    ).to(device)

    num_params = sum(param.numel() for param in model.parameters())
    logger.info(f"Pressure model params: {num_params / 1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LR"], weight_decay=CONFIG["WEIGHT_DECAY"])
    criterion = nn.HuberLoss(delta=8.0)

    warmup_epochs = int(CONFIG["WARMUP_EPOCHS"])
    if warmup_epochs > 0:
        scheduler_warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, CONFIG["EPOCHS"] - warmup_epochs),
            eta_min=CONFIG["MIN_LR"],
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs]
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, CONFIG["EPOCHS"]), eta_min=CONFIG["MIN_LR"]
        )

    scaler = None
    if CONFIG["USE_AMP"] and device.type == "cuda" and amp_dtype == torch.float16:
        scaler = torch.amp.GradScaler()

    logger.info("Sanity check...")
    first_batch = next(iter(train_loader))
    z0 = first_batch["z"].to(device)
    c0 = first_batch["coords"].to(device)
    normals0 = sdf_unit_normals_chunked(
        sdf_model=sdf_model,
        coords=c0,
        z=z0,
        chunk=min(CONFIG["SDF_NORMAL_CHUNK"], c0.shape[1]),
        eps=CONFIG["SDF_NORMAL_EPS"],
        compute_dtype=sdf_compute_dtype,
    )
    x60 = torch.cat([c0, normals0], dim=-1)
    with torch.no_grad(), make_autocast(device, amp_dtype):
        y0 = model(x60, z=z0)
    logger.info(f"Sanity OK. pred shape={tuple(y0.shape)}")
    del first_batch, z0, c0, normals0, x60, y0
    if device.type == "cuda":
        torch.cuda.empty_cache()

    bench_one = None
    if CONFIG["BENCH_ENABLE"]:
        for batch in val_loader:
            bench_one = take_one_sample(batch)
            break

    best_val_l2 = float("inf")
    logger.info("Start training...")
    start_time = time.time()

    for epoch in range(1, CONFIG["EPOCHS"] + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model, sdf_model, train_loader, optimizer, criterion, device, epoch, scaler, amp_dtype, sdf_compute_dtype
        )
        val_loss, val_l2, val_l1 = validate(
            model, sdf_model, val_loader, criterion, device, epoch, amp_dtype, sdf_compute_dtype
        )

        lr_now = optimizer.param_groups[0]["lr"]
        message = (
            f"Epoch {epoch:03d} | {time.time() - epoch_start:.1f}s | lr={lr_now:.2e} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"val_relL2={val_l2:.6f} | val_relL1={val_l1:.6f}"
        )

        if bench_one is not None and epoch % int(CONFIG["BENCH_EVERY"]) == 0:
            infer_ms, peak_alloc_mb, peak_reserved_mb = benchmark_infer_ms_per_sample_fullpipe(
                model=model,
                sdf_model=sdf_model,
                one_batch=bench_one,
                device=device,
                amp_dtype=amp_dtype,
                warmup=int(CONFIG["BENCH_WARMUP"]),
                iters=int(CONFIG["BENCH_ITERS"]),
                sdf_compute_dtype=sdf_compute_dtype,
            )
            message += (
                f" | Infer(fullpipe): {infer_ms:.3f} ms/sample | "
                f"GPU peak alloc/resv: {peak_alloc_mb:.1f}/{peak_reserved_mb:.1f} MB"
            )

        logger.info(message)

        if val_l2 < best_val_l2:
            best_val_l2 = val_l2
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            logger.info(f"  >>> best saved (val_relL2={best_val_l2:.6f})")

        if epoch % 50 == 0:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_relL2": val_l2},
                os.path.join(save_dir, "latest.pth"),
            )

        scheduler.step()

    total = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f"Done. total={total}, best_val_relL2={best_val_l2:.6f}")
    logger.close()


if __name__ == "__main__":
    main()
