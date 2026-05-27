"""
代码说明：
Stable-SDF 核心网络结构 (DeepSDFWithPE)。
包含用于解决坐标回归光谱偏差问题的高频位置编码层 (PositionalEncoding)，
以及基于 Weight Norm 的主干 MLP。
该模型可通用于汽车、机翼和 HH 等 2D/3D 任务。
"""

import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    """
    位置编码层：将低维坐标映射到高维频域
    解决 'Spectral Bias' 问题，让模型能学到高频细节（如粗糙边界）
    """
    def __init__(self, num_freqs=6, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.register_buffer('freq_bands', 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs))

    def forward(self, x):
        res = []
        if self.include_input:
            res.append(x)
            
        for freq in self.freq_bands:
            for func in [torch.sin, torch.cos]:
                res.append(func(x * freq * np.pi))
                
        return torch.cat(res, dim=-1)

class DeepSDFWithPE(nn.Module):
    """
    带有位置编码的 DeepSDF 模型
    Args:
        latent_dim: 隐变量 (z_code) 的维度
        hidden_dim: 隐藏层神经元数量
        num_layers: MLP 层数
        num_freqs: 位置编码频率数量 (建议 4-8)
    """
    def __init__(self, latent_dim=32, hidden_dim=128, num_layers=4, num_freqs=6):
        super().__init__()
        
        # 1. 初始化位置编码
        self.pe = PositionalEncoding(num_freqs=num_freqs, include_input=True)
        coord_dim = 2 + 4 * num_freqs
        
        input_dim = coord_dim + latent_dim 
        self.skip_layer = num_layers // 2
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                in_dim = input_dim
            elif i == self.skip_layer:
                in_dim = hidden_dim + input_dim
            else:
                in_dim = hidden_dim
            
            linear = torch.nn.utils.weight_norm(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Sequential(linear, nn.ReLU(inplace=True)))
            
        self.layers = nn.ModuleList(layers)
        self.out_layer = nn.Linear(hidden_dim, 1)

    def forward(self, coords, z_code):
        coords_pe = self.pe(coords)
        x_input = torch.cat([coords_pe, z_code], dim=-1)
        
        x = x_input
        for i, layer in enumerate(self.layers):
            if i == self.skip_layer:
                x = torch.cat([x, x_input], dim=-1)
            x = layer(x)
            
        return self.out_layer(x)