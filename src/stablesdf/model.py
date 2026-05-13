import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.parametrizations import weight_norm

class PositionalEncoding(nn.Module):
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

class DeepSDFNet(nn.Module):
    def __init__(self, latent_size=256, hidden_dim=512, num_freqs=4):
        super(DeepSDFNet, self).__init__()
        self.latent_size = latent_size
        
        self.pe = PositionalEncoding(num_freqs=num_freqs, include_input=True)
        coord_dim = 3 + 6 * num_freqs
        input_dim = coord_dim + latent_size
        
        self.layer1 = weight_norm(nn.Linear(input_dim, hidden_dim))
        self.layer2 = weight_norm(nn.Linear(hidden_dim, hidden_dim))
        self.layer3 = weight_norm(nn.Linear(hidden_dim, hidden_dim))
        self.layer4 = weight_norm(nn.Linear(hidden_dim, hidden_dim - input_dim)) 
        self.layer5 = weight_norm(nn.Linear(hidden_dim, hidden_dim))
        self.layer6 = weight_norm(nn.Linear(hidden_dim, hidden_dim))
        self.layer7 = weight_norm(nn.Linear(hidden_dim, hidden_dim))
        self.layer8 = nn.Linear(hidden_dim, 1)

        self.act = nn.Softplus(beta=100)

    def forward(self, x, z):
        x_pe = self.pe(x)
        inputs = torch.cat([z, x_pe], dim=1)

        out = self.act(self.layer1(inputs))
        out = self.act(self.layer2(out))
        out = self.act(self.layer3(out))
        out = self.act(self.layer4(out))
        out = torch.cat([out, inputs], dim=1) 
        out = self.act(self.layer5(out))
        out = self.act(self.layer6(out))
        out = self.act(self.layer7(out))
        out = self.layer8(out)
        return out
    
class DeepSDFWorkspace:
    def __init__(self, num_scenes, latent_size=256):
        self.model = DeepSDFNet(latent_size=latent_size).cuda()
        # 初始化 Latent Codes: N(0, 0.01)
        self.latents = nn.Embedding(num_scenes, latent_size).cuda()
        torch.nn.init.normal_(self.latents.weight.data, 0.0, 0.01)

        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.latents.parameters()}
        ], lr=1e-4)
        
        # 初始化 GradScaler
        self.scaler = torch.cuda.amp.GradScaler()

    def train_step(self, scene_indices, points, gt_sdf, sigma=0.1, noise_std=0.0):
        self.model.train()
        self.optimizer.zero_grad()

        # 1. 获取 Latent Codes
        batch_vecs_clean = self.latents(scene_indices)

        # 2. 注入噪声 (Denoising Auto-Decoder 核心)
        if noise_std > 0 and self.model.training:
            noise = torch.randn_like(batch_vecs_clean) * noise_std
            batch_vecs_input = batch_vecs_clean + noise
        else:
            batch_vecs_input = batch_vecs_clean

        # 3. 前向传播
        with torch.cuda.amp.autocast():
            pred_sdf = self.model(points, batch_vecs_input)
            loss = self.compute_loss(pred_sdf, gt_sdf, batch_vecs_clean, sigma)

        # 4. 使用 Scaler 进行反向传播和参数更新
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    def compute_loss(self, pred, target, z_vector, delta=0.1, sigma_reg=1e-4):
        # 1. Clamping (保持原样)
        clamp_pred = torch.clamp(pred, -delta, delta)
        clamp_target = torch.clamp(target, -delta, delta)
        
        # 2. 计算基础 L1 Loss
        abs_error = torch.abs(clamp_pred - clamp_target)
        
        # [新增] 3. 表面加权 (Surface Weighting)
        # 逻辑：如果真实值的绝对值很小（靠近表面），我们给它更大的权重
        # 例如：距离表面 0.01 以内的点，权重为 5倍；其他点权重为 1倍
        is_near_surface = torch.abs(target) < 0.02
        weights = torch.ones_like(abs_error)
        weights[is_near_surface] = 6.0  # 权重倍率可调，3.0-10.0 均可
        
        l1_loss = (abs_error * weights).mean() # 使用加权平均

        # 4. 正则化 (保持原样)
        reg_loss = torch.mean(torch.sum(z_vector ** 2, dim=1)) * sigma_reg
        
        return l1_loss + reg_loss

# 推理函数不需要改动，保持原样即可
def infer_latent_code(model, point_samples, sdf_samples, latent_size=256, num_steps=500):
    """
    对未见过的形状进行推理（寻找最优 Latent Code）
    """
    model.eval() # 冻结网络参数
    
    # 随机初始化一个新的 latent code
    z = torch.ones(1, latent_size).normal_(0, 0.01).cuda()
    z.requires_grad = True # 开启梯度
    
    # 定义只针对 z 的优化器
    optimizer = torch.optim.Adam([z], lr=5e-3)
    
    for i in range(num_steps):
        optimizer.zero_grad()
        
        # 扩展 z 以匹配采样点的 batch size
        z_expanded = z.expand(point_samples.shape[0], -1)
        
        pred_sdf = model(point_samples, z_expanded)
        
        # 计算损失
        loss = torch.abs(torch.clamp(pred_sdf, -0.1, 0.1) - torch.clamp(sdf_samples, -0.1, 0.1)).mean()
        loss += 1e-4 * torch.mean(z ** 2)
        
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f"Step {i}, Loss: {loss.item()}")
            
    return z.detach() # 返回优化好的 Latent Code