# dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm # 建议安装: pip install tqdm

class MultiSDFDataset(Dataset):
    def __init__(self, file_paths, num_samples_per_scene=None):
        """
        file_paths: list of strings, 每个 .npz 文件的路径
        num_samples_per_scene: 为了显存/内存考虑，可以限制每个场景只取部分点 (例如 16384)，None 表示全取
        """
        self.all_data = []
        self.scene_indices = []
        
        print(f"正在加载 {len(file_paths)} 个场景的数据...")
        
        # 遍历所有文件加载数据
        for scene_idx, filepath in tqdm(enumerate(file_paths), total=len(file_paths)):
            try:
                # 假设 npz 结构是 'points' 和 'sdf' (或是 'pos', 'neg' 等，根据你的预处理决定)
                # 这里假设你的 npz 里直接有 points (N, 3) 和 sdf (N, 1)
                data = np.load(filepath)
                
                # 兼容性处理：有些预处理脚本把 key 命名为 'pos' 和 'neg'
                if 'points' in data:
                    points = data['points']
                    sdfs = data['sdf']
                else:
                    # 这是一个常见的 DeepSDF 数据变种，如果你的数据不同，请修改这里
                    points = np.concatenate([data['pos'], data['neg']], axis=0)
                    sdfs = np.concatenate([data['pos_sdf'], data['neg_sdf']], axis=0)

                # 随机下采样 (如果点太多，比如每个文件有50万个点，内存会爆)
                if num_samples_per_scene and points.shape[0] > num_samples_per_scene:
                    indices = np.random.choice(points.shape[0], num_samples_per_scene, replace=False)
                    points = points[indices]
                    sdfs = sdfs[indices]

                # 构造数据条目: (x, y, z, sdf, scene_id)
                # 我们把 scene_id 拼在最后，方便 DataLoader 搬运
                N = points.shape[0]
                
                # points: (N, 3), sdfs: (N, 1)
                sdfs = sdfs.reshape(-1, 1)
                
                # 创建一个全为 scene_idx 的列向量
                idx_col = np.full((N, 1), scene_idx, dtype=np.float32)
                
                # 合并: (N, 5) -> [x, y, z, sdf, scene_id]
                combined = np.hstack([points, sdfs, idx_col])
                self.all_data.append(combined)
                
            except Exception as e:
                print(f"跳过损坏的文件 {filepath}: {e}")

        # 将列表转换为巨大的 Tensor
        # 700 * 100000 * 5 * 4 bytes ≈ 1.4 GB，内存完全放得下
        self.all_data = np.vstack(self.all_data)
        self.all_data = torch.from_numpy(self.all_data).float()
        
        print(f"数据加载完毕。总点数: {self.all_data.shape[0]}")

    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self, idx):
        # row: [x, y, z, sdf, scene_id]
        row = self.all_data[idx]
        
        point = row[0:3]
        sdf = row[3:4] # 保持 (1,) 维度
        scene_id = row[4].long() # 转回整数用于 Embedding 索引
        
        return scene_id, point, sdf