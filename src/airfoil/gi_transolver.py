"""
2D 机翼任务：GI-Transolver 模型定义
说明：结合物理注意力机制处理不规则网格，支持几何隐向量(z)的多级特征注入。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.init import trunc_normal_ as trunc_normal_

# -------------------------
# Utils
# -------------------------
ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight

# -------------------------
# Core Modules
# -------------------------
class Physics_Attention_Irregular_Mesh(nn.Module):
    def __init__(
        self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64,
        use_slice_z_add: bool = True, z_dim: int = 256, z_add_dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)

        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.use_slice_z_add = use_slice_z_add
        if use_slice_z_add:
            self.z_proj = nn.Linear(z_dim, inner_dim, bias=False)
            self.z_drop = nn.Dropout(z_add_dropout) if z_add_dropout > 0 else nn.Identity()
        else:
            self.z_proj = None
            self.z_drop = nn.Identity()

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, z=None):
        B, N, C = x.shape
        fx_mid = (self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head)
                  .permute(0, 2, 1, 3).contiguous())
        x_mid = (self.in_project_x(x).reshape(B, N, self.heads, self.dim_head)
                 .permute(0, 2, 1, 3).contiguous())

        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(2)

        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None])

        if (self.z_proj is not None) and (z is not None):
            z_inj = self.z_drop(self.z_proj(z))
            z_inj = z_inj.view(B, self.heads, 1, self.dim_head)
            slice_token = slice_token + z_inj

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice = torch.matmul(attn, v)

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)

class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
        super().__init__()
        act_cls = ACTIVATION.get(act, None)
        if act_cls is None:
            raise NotImplementedError(f"Unknown activation: {act}")
        act_mod = act_cls() if callable(act_cls) else act_cls()
        self.n_layers = n_layers
        self.res = res

        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_mod)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), (act_cls() if callable(act_cls) else act_cls()))
             for _ in range(n_layers)]
        )
        self.linear_post = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            x = self.linears[i](x) + x if self.res else self.linears[i](x)
        x = self.linear_post(x)
        return x

class Transolver_block(nn.Module):
    def __init__(self, num_heads: int, hidden_dim: int, dropout: float, act="gelu",
                 mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32,
                 use_slice_z_add: bool = True, z_dim: int = 256, z_add_dropout: float = 0.0):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads, dropout=dropout,
            slice_num=slice_num, use_slice_z_add=use_slice_z_add, z_dim=z_dim, z_add_dropout=z_add_dropout,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)

        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx, z=None):
        fx = self.Attn(self.ln_1(fx), z=z) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx

# -------------------------
# Transolver Model
# -------------------------
class GITransolver(nn.Module):
    def __init__(
        self, space_dim=2, n_layers=5, n_hidden=256, dropout=0.0, n_head=8,
        act="gelu", mlp_ratio=1, fun_dim=0, out_dim=1, slice_num=32, ref=8, unified_pos=False,
        use_slice_z_add=True, z_dim=256, z_add_dropout=0.0, z_inject_layers=-1
    ):
        super().__init__()
        self.__name__ = "Transolver"
        self.ref = ref
        self.unified_pos = unified_pos
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.use_slice_z_add = use_slice_z_add
        self.z_dim = z_dim
        self.z_inject_layers = z_inject_layers

        if self.unified_pos:
            grid_feat_dim = (self.ref ** self.space_dim)
            self.preprocess = MLP(fun_dim + grid_feat_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.blocks = nn.ModuleList(
            [Transolver_block(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout, act=act, mlp_ratio=mlp_ratio,
                out_dim=out_dim, slice_num=slice_num, last_layer=(_ == n_layers - 1),
                use_slice_z_add=use_slice_z_add, z_dim=z_dim, z_add_dropout=z_add_dropout)
             for _ in range(n_layers)]
        )

        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float))
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, my_pos):
        B = my_pos.shape[0]
        device, dtype = my_pos.device, my_pos.dtype

        if self.space_dim == 2:
            gridx = torch.linspace(-1.5, 1.5, self.ref, device=device, dtype=dtype)
            gridy = torch.linspace(-1.5, 1.5, self.ref, device=device, dtype=dtype)
            grid_ref = torch.stack(torch.meshgrid(gridx, gridy, indexing="ij"), dim=-1).reshape(1, -1, 2)
        elif self.space_dim == 3:
            gridx = torch.linspace(-1.5, 1.5, self.ref, device=device, dtype=dtype)
            gridy = torch.linspace(0, 2, self.ref, device=device, dtype=dtype)
            gridz = torch.linspace(-4, 4, self.ref, device=device, dtype=dtype)
            grid_ref = torch.stack(torch.meshgrid(gridx, gridy, gridz, indexing="ij"), dim=-1).reshape(1, -1, 3)
        else:
            raise ValueError(f"Unsupported space_dim={self.space_dim} for unified_pos.")

        grid_ref = grid_ref.repeat(B, 1, 1)
        pos_feat = torch.cdist(my_pos, grid_ref)
        return pos_feat

    def forward(self, data):
        x, pos, condition = data
        if self.unified_pos:
            x = self.get_grid(pos)

        fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]

        if (self.z_inject_layers is None) or (self.z_inject_layers == -1):
            upto = len(self.blocks)
        else:
            upto = int(max(0, min(len(self.blocks), self.z_inject_layers)))

        for i, block in enumerate(self.blocks):
            z_i = condition if (self.use_slice_z_add and (condition is not None) and (i < upto)) else None
            fx = block(fx, z=z_i)

        return fx