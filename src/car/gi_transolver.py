"""
代码说明：
3D 汽车压力场任务的 GI-Transolver 模型定义。
输入为每个表面采样点的归一化坐标和 SDF 法向量 [x, y, z, nx, ny, nz]，
并通过 slice latent gated z injection 注入 Stable-SDF latent code，输出点压力。
"""

from typing import Optional

import torch
import torch.nn as nn


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


def trunc_normal_(tensor, std=0.02):
    return nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)


class SliceLatentGatedZInject(nn.Module):
    """
    在 slice token 上用门控方式注入形状 latent code。
    slice_token: [B, H, G, Dh]
    z: [B, z_dim]
    """

    def __init__(self, dim_head: int, z_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim_head, dim_head),
            nn.SiLU(),
            nn.Linear(dim_head, z_dim),
            nn.Sigmoid(),
        )
        self.z_proj = nn.Linear(z_dim, dim_head, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, slice_token: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        gate = self.gate_mlp(slice_token)
        z_gated = z[:, None, None, :] * gate
        z_inj = self.drop(self.z_proj(z_gated))
        return slice_token + z_inj


class Physics_Attention_Irregular_Mesh(nn.Module):
    """
    Transolver 的不规则网格 slice attention。
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 64,
        use_slice_z_inject: bool = False,
        z_dim: int = 256,
        z_inject_dropout: float = 0.0,
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
        nn.init.orthogonal_(self.in_project_slice.weight)

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.z_inject = (
            SliceLatentGatedZInject(dim_head=dim_head, z_dim=z_dim, dropout=z_inject_dropout)
            if use_slice_z_inject
            else None
        )

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = x.shape

        fx_mid = (
            self.in_project_fx(x)
            .view(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        x_mid = (
            self.in_project_x(x)
            .view(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        slice_logits = self.in_project_slice(x_mid) / self.temperature
        slice_w = self.softmax(slice_logits)
        slice_norm = slice_w.sum(2)

        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_w)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None])

        if self.z_inject is not None and z is not None:
            slice_token = self.z_inject(slice_token, z)

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)

        attn = self.softmax((q @ k.transpose(-1, -2)) * self.scale)
        attn = self.dropout(attn)
        out_slice = attn @ v

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_w)
        out_x = out_x.permute(0, 2, 1, 3).contiguous().view(B, N, self.heads * self.dim_head)
        return self.to_out(out_x)


class MLP(nn.Module):
    def __init__(self, n_input: int, n_hidden: int, n_output: int, act: str = "gelu", res: bool = False):
        super().__init__()
        act_cls = ACTIVATION[act]
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            act_cls() if callable(act_cls) else act_cls,
            nn.Linear(n_hidden, n_output),
        )
        self.res = res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return x + y if self.res and x.shape[-1] == y.shape[-1] else y


class TransolverBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        slice_num: int,
        mlp_ratio: int,
        out_dim: int,
        last: bool,
        use_slice_z_inject: bool,
        z_dim: int,
        z_inject_dropout: float,
        act: str = "gelu",
    ):
        super().__init__()
        self.last = last

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = Physics_Attention_Irregular_Mesh(
            dim=hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
            use_slice_z_inject=use_slice_z_inject,
            z_dim=z_dim,
            z_inject_dropout=z_inject_dropout,
        )

        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, act=act, res=False)

        if last:
            self.ln3 = nn.LayerNorm(hidden_dim)
            self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attn(self.ln1(x), z=z) + x
        x = self.mlp(self.ln2(x)) + x
        if self.last:
            return self.head(self.ln3(x))
        return x


class Transolver(nn.Module):
    """
    汽车压力场 GI-Transolver。
    x6: [B, N, 6]，包含坐标和 SDF 法向量。
    z: [B, z_dim]，Stable-SDF latent code。
    """

    def __init__(
        self,
        in_dim: int = 6,
        n_layers: int = 5,
        n_hidden: int = 256,
        n_head: int = 8,
        dropout: float = 0.0,
        mlp_ratio: int = 2,
        slice_num: int = 32,
        out_dim: int = 1,
        act: str = "gelu",
        use_slice_z_inject: bool = True,
        z_dim: int = 256,
        z_inject_dropout: float = 0.0,
        z_inject_layers: int = -1,
    ):
        super().__init__()
        self.__name__ = "Transolver"
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.use_slice_z_inject = use_slice_z_inject
        self.z_inject_layers = z_inject_layers

        self.pre = nn.Sequential(
            nn.Linear(in_dim, n_hidden * 2),
            nn.GELU(),
            nn.Linear(n_hidden * 2, n_hidden),
        )

        self.blocks = nn.ModuleList(
            [
                TransolverBlock(
                    hidden_dim=n_hidden,
                    num_heads=n_head,
                    dropout=dropout,
                    slice_num=slice_num,
                    mlp_ratio=mlp_ratio,
                    out_dim=out_dim,
                    last=(i == n_layers - 1),
                    use_slice_z_inject=use_slice_z_inject,
                    z_dim=z_dim,
                    z_inject_dropout=z_inject_dropout,
                    act=act,
                )
                for i in range(n_layers)
            ]
        )

        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden))
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x6: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x6.dim() != 3 or x6.size(-1) != self.in_dim:
            raise ValueError(f"x6 must be [B, N, {self.in_dim}], got {tuple(x6.shape)}")

        x = self.pre(x6) + self.placeholder[None, None, :]

        if self.z_inject_layers is None or self.z_inject_layers == -1:
            upto = len(self.blocks)
        else:
            upto = int(max(0, min(len(self.blocks), self.z_inject_layers)))

        for i, block in enumerate(self.blocks):
            z_i = z if (self.use_slice_z_inject and z is not None and i < upto) else None
            x = block(x, z=z_i)

        return x
