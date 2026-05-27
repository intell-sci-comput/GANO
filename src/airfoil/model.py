"""
代码说明：
2D 机翼任务的 Stable-SDF 解码器。
包含二维位置编码和带 skip connection 的 DeepSDF MLP，用于从翼型 latent code 解码 SDF。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt



class PositionalEncoding(nn.Module):
    """
    位置编码层：将低维坐标映射到高维频域
    解决 'Spectral Bias' 问题，让模型能学到高频细节（如粗糙边界）
    """
    def __init__(self, num_freqs=6, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        # 生成频率带: 2^0, 2^1, ... , 2^(N-1)
        # 注册为 buffer，不会作为参数更新
        self.register_buffer('freq_bands', 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs))

    def forward(self, x):
        # x shape: (..., d) 例如 (B, N, 2)
        res = []
        if self.include_input:
            res.append(x)
            
        for freq in self.freq_bands:
            for func in [torch.sin, torch.cos]:
                res.append(func(x * freq * np.pi))
                
        # 拼接所有频率特征
        # 输出维度 = input_dim + input_dim * 2 * num_freqs
        return torch.cat(res, dim=-1)

class DeepSDFWithPE(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=128, num_layers=4, num_freqs=6):
        """
        带有位置编码的 DeepSDF 模型
        Args:
            num_freqs: 频率数量。越高能拟合越细微的噪点，但也更容易过拟合。建议 4-8 之间。
        """
        super().__init__()
        
        # 1. 初始化位置编码
        self.pe = PositionalEncoding(num_freqs=num_freqs, include_input=True)
        
        # 计算编码后的坐标维度
        # 2D坐标 -> 2 + 2 * 2 * num_freqs = 2 + 4 * num_freqs
        coord_dim = 2 + 4 * num_freqs
        
        # 输入层维度 = 编码后的坐标 + 隐向量
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
            
            # 使用 Weight Norm
            linear = torch.nn.utils.weight_norm(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Sequential(linear, nn.ReLU(inplace=True)))
            
        self.layers = nn.ModuleList(layers)
        self.out_layer = nn.Linear(hidden_dim, 1)

    def forward(self, coords, z_code):
        # 1. 先对坐标进行位置编码
        # coords: (B, N, 2) -> coords_pe: (B, N, 2 + 4*freqs)
        coords_pe = self.pe(coords)
        
        # 2. 拼接隐向量
        x_input = torch.cat([coords_pe, z_code], dim=-1)
        
        x = x_input
        for i, layer in enumerate(self.layers):
            if i == self.skip_layer:
                x = torch.cat([x, x_input], dim=-1)
            x = layer(x)
            
        return self.out_layer(x)
