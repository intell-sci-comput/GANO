"""
代码说明：
用于将 genpde.py 生成的原始复数物理场数据进行归一化预处理。
1. 将复数 (Complex) 拆分为实部和虚部 (2 通道)。
2. 计算全局 Mean 和 Std。
3. 归一化数据并保存，供 GI-Transolver 训练和反演使用。
"""

import os
import numpy as np
import torch

# 配置路径
CONFIG = {
    "load_path": "../../data/hh/scattering_dataset_scat_fields_k7.npz",
    "save_npz": "../../data/hh/scattering_dataset_normalized.npz",
    "save_pt": "../../data/hh/normalization_stats.pt" # 专供反演脚本调用
}

def main():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    load_path = os.path.normpath(os.path.join(cur_dir, CONFIG["load_path"]))
    save_npz = os.path.normpath(os.path.join(cur_dir, CONFIG["save_npz"]))
    save_pt = os.path.normpath(os.path.join(cur_dir, CONFIG["save_pt"]))

    if not os.path.exists(load_path):
        print(f"[!] 错误: 找不到原始 PDE 数据 {load_path}")
        return

    print(f"[*] 加载原始 PDE 场数据: {load_path}")
    data = np.load(load_path)
    fields_complex = data['fields']  # Shape: (N, A, 256, 256)

    print("[*] 正在拆分复数 (提取实部和虚部)...")
    fields_real = np.real(fields_complex)
    fields_imag = np.imag(fields_complex)
    # 堆叠成 2 通道: (N, A, 256, 256, 2)
    fields_stacked = np.stack([fields_real, fields_imag], axis=-1).astype(np.float32)

    print("[*] 计算全局 Mean 和 Std...")
    mean = np.mean(fields_stacked, axis=(0, 1, 2, 3))
    std = np.std(fields_stacked, axis=(0, 1, 2, 3))
    print(f"    Mean: {mean}")
    print(f"    Std:  {std}")

    print("[*] 执行数据归一化...")
    fields_norm = (fields_stacked - mean) / (std + 1e-8)

    print(f"[*] 保存归一化数据集至: {save_npz}")
    np.savez_compressed(
        save_npz,
        fields_norm=fields_norm,
        mean=mean,
        std=std
    )

    print(f"[*] 保存统计量至: {save_pt}")
    torch.save({
        'mean': torch.from_numpy(mean), 
        'std': torch.from_numpy(std)
    }, save_pt)

    print("[*] 归一化完成！数据已准备就绪，可以安全启动 GI-Transolver 了。")

if __name__ == "__main__":
    main()