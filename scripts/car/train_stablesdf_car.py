"""
3D汽车任务的 Stable-sdf 训练脚本
说明：本版本已针对开源仓库规范进行重构，采用统一的 CONFIG 字典管理超参数和路径。
"""

import os
import sys
import glob
import json
import random
import time
import torch
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 【关键适配】将 GANO 仓库根目录加入系统路径，这样才能正确导入 src 下的模型
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.stablesdf.model import DeepSDFWorkspace

# ==========================================
# ======= [全局参数配置字典] ================
# ==========================================
CONFIG = {
    # --- 路径配置 ---
    "DATA_DIRS": [
        '/home/sunguoze/Getsdf/sdf_output_hybrid',
        '/home/sunguoze/Getsdf/sdf_output_hybrid2'
    ],
    # 将保存路径指向我们在 GANO 里规范的 checkpoints 目录
    "SAVE_DIR": os.path.join(project_root, "checkpoints", "car_training_h800_all"),

    # --- 训练超参数 ---
    "BATCH_SIZE": 500000,
    "NUM_EPOCHS": 800,
    "START_EPOCH": 400,
    "LATENT_SIZE": 256,
    "RESUME": True,

    # --- 噪声控制 ---
    "NOISE_STD": 0.005,      # 实验 1: 0.0 | 实验 2: 0.005 | 实验 3: 0.05
    "NOISE_OFF_EPOCH": 720,  # >= 这个 epoch 后关闭噪声微调

    # --- 日志与保存频率 ---
    "LOG_EVERY": 10,         # 每多少轮打印一次日志并保存 latest
    "SAVE_EVERY": 100,       # 每多少轮保留一个历史权重备份
}

os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)

def load_data_to_gpu(file_list):
    """
    将所有 npz 文件的数据读取并合并到一个巨大的 GPU Tensor 中
    优化版：避免在 CPU 进行 huge numpy concatenate，防止内存溢出卡死
    """
    all_points = []
    all_sdfs = []
    all_indices = []
    
    print(f"正在加载 {len(file_list)} 个文件到内存...")
    
    for idx, filepath in tqdm(enumerate(file_list), total=len(file_list)):
        try:
            data = np.load(filepath)
            pts = None
            sdf = None
            
            # 1. 优先匹配 sdf.py 生成的格式: 'coords' (N,3) 和 'sdf' (N,1)
            if 'coords' in data and 'sdf' in data:
                pts = data['coords']
                sdf = data['sdf']
            
            # 2. 兼容其他常见格式 (points, sdf)
            elif 'points' in data and 'sdf' in data:
                pts = data['points']
                sdf = data['sdf']
            
            # 3. 尝试寻找 (N, 4) 格式的数组 (xyz + sdf)
            if pts is None:
                for key in data.keys():
                    arr = data[key]
                    if isinstance(arr, np.ndarray) and len(arr.shape) == 2 and arr.shape[1] == 4:
                        pts = arr[:, :3]
                        sdf = arr[:, 3:] 
                        break
            
            # 4. DeepSDF 原始格式 (pos/neg)
            if pts is None and 'pos' in data:
                pts = np.concatenate([data['pos'], data['neg']], axis=0)
                sdf = np.concatenate([data['pos_sdf'], data['neg_sdf']], axis=0)

            if pts is None or sdf is None:
                # 打印一下 keys 方便调试
                print(f"跳过文件 {filepath}: 未找到合适的数据字段 (keys: {list(data.keys())})")
                continue

            pts = pts.astype(np.float32)
            sdf = sdf.astype(np.float32).reshape(-1, 1)

            indices = np.full((pts.shape[0], 1), idx, dtype=np.int64)
            all_points.append(pts)
            all_sdfs.append(sdf)
            all_indices.append(indices)
            
        except Exception as e:
            print(f"加载错误 {filepath}: {e}")

    if len(all_points) == 0:
        raise RuntimeError("没有任何数据被成功加载！")

    # --- 优化开始 ---
    # 计算总点数
    total_points = sum(p.shape[0] for p in all_points)
    print(f"总数据量: {total_points} 点。正在分配 GPU 显存...")

    # 1. 在 GPU 上预分配空间 (避免 CPU 内存峰值)
    gpu_points = torch.empty((total_points, 3), dtype=torch.float32, device='cuda')
    gpu_sdfs = torch.empty((total_points, 1), dtype=torch.float32, device='cuda')
    gpu_indices = torch.empty((total_points,), dtype=torch.int64, device='cuda')

    print("正在将数据流式传输到 GPU...")
    
    current_idx = 0
    # 使用 tqdm 显示搬运进度
    for pts, sdf, idxs in tqdm(zip(all_points, all_sdfs, all_indices), total=len(all_points), desc="Uploading to GPU"):
        n = pts.shape[0]
        end_idx = current_idx + n
        
        # 2. 直接将小块数据搬运到 GPU 对应的位置
        gpu_points[current_idx:end_idx] = torch.from_numpy(pts).to('cuda', non_blocking=True)
        gpu_sdfs[current_idx:end_idx] = torch.from_numpy(sdf).to('cuda', non_blocking=True)
        gpu_indices[current_idx:end_idx] = torch.from_numpy(idxs.reshape(-1)).to('cuda', non_blocking=True)
        
        current_idx = end_idx

    # 3. 手动释放 CPU 内存
    del all_points, all_sdfs, all_indices
    import gc
    gc.collect()
    # --- 优化结束 ---
    
    print(f"数据加载完成！总点数: {gpu_points.shape[0]}")
    return gpu_points, gpu_sdfs, gpu_indices

def main():
    # 1. 确定训练文件列表
    # 逻辑修改：遍历所有 DATA_DIRS，收集所有 npz 文件，不进行分割
    all_files = []
    print("正在搜索数据文件...")
    for data_dir in CONFIG["DATA_DIRS"]:
        if os.path.exists(data_dir):
            files = glob.glob(os.path.join(data_dir, "*.npz"))
            print(f"  - {data_dir}: 找到 {len(files)} 个文件")
            all_files.extend(files)
        else:
            print(f"  - {data_dir}: 路径不存在，跳过")

    if not all_files:
        print(f"错误：未找到任何 .npz 文件")
        return

    # 排序并打乱
    all_files = sorted(all_files)
    random.seed(42)
    random.shuffle(all_files)
    
    train_files = all_files # 所有文件都用于训练
    
    print(f"总共使用 {len(train_files)} 个文件进行训练 (全量数据，无测试集)")
    
    # 保存列表以便记录
    file_list_path = os.path.join(CONFIG["SAVE_DIR"], "file_list.json")
    with open(file_list_path, "w") as f:
        json.dump({"train": train_files, "test": []}, f, indent=4)

    # 2. 加载数据到 GPU
    points_gpu, sdfs_gpu, indices_gpu = load_data_to_gpu(train_files)
    num_samples = points_gpu.shape[0]
    
    # 3. 初始化 Workspace
    workspace = DeepSDFWorkspace(num_scenes=len(train_files), latent_size=CONFIG["LATENT_SIZE"])

    # 4. 加载断点 (Checkpoint)
    if CONFIG["RESUME"]:
        model_path = os.path.join(CONFIG["SAVE_DIR"], 'model_latest.pth')
        latent_path = os.path.join(CONFIG["SAVE_DIR"], 'latents_latest.pth')
        
        if os.path.exists(model_path) and os.path.exists(latent_path):
            print(f"正在加载断点: {model_path}")
            workspace.model.load_state_dict(torch.load(model_path))
            workspace.latents.load_state_dict(torch.load(latent_path))
            print(">>> 模型与 Latent Codes 加载成功！继续训练。")
        else:
            print("警告：未找到断点文件，将从头开始训练！")

    # 5. 训练循环
    print(f"开始训练: Epoch {CONFIG['START_EPOCH']} -> {CONFIG['NUM_EPOCHS']}")
    print(f"当前使用的噪声强度 (NOISE_STD): {CONFIG['NOISE_STD']}")
    
    start_time = time.time()
    pbar = tqdm(range(CONFIG["START_EPOCH"], CONFIG["NUM_EPOCHS"]), desc="Training", unit="epoch")

    for epoch in pbar:
        perm = torch.randperm(num_samples, device='cuda')
        
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, num_samples, CONFIG["BATCH_SIZE"]):
            batch_idxs = perm[i : i + CONFIG['BATCH_SIZE']]
            
            b_points = points_gpu[batch_idxs]
            b_sdfs = sdfs_gpu[batch_idxs]
            b_indices = indices_gpu[batch_idxs]
            
            # [策略] 3500 轮后停止加噪声，进行微调
            current_noise = CONFIG["NOISE_STD"] if epoch < CONFIG["NOISE_OFF_EPOCH"] else 0.0
            
            loss = workspace.train_step(
                b_indices, b_points, b_sdfs, noise_std=current_noise
            )
            
            epoch_loss += loss
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        pbar.set_postfix({"Loss": f"{avg_loss:.6f}", "Noise": f"{current_noise:.4f}"})
        
        # 每 100 轮保存一次
        if (epoch + 1) % CONFIG["LOG_EVERY"] == 0:
            elapsed = time.time() - start_time
            tqdm.write(f"Epoch {epoch+1}/{CONFIG['NUM_EPOCHS']} | Loss: {avg_loss:.6f} | Time: {elapsed:.1f}s")
            torch.save(workspace.model.state_dict(), os.path.join(CONFIG["SAVE_DIR"], 'model_latest.pth'))
            torch.save(workspace.latents.state_dict(), os.path.join(CONFIG["SAVE_DIR"], 'latents_latest.pth'))
            
            # 每 1000 轮额外备份
            if (epoch + 1) % CONFIG["SAVE_EVERY"] == 0:
                torch.save(workspace.model.state_dict(), os.path.join(CONFIG["SAVE_DIR"], f'model_{epoch+1}.pth'))

    print("训练结束")
if __name__ == "__main__":
    main()
