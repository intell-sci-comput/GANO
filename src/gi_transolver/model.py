"""代码说明：GI-Transolver 核心网络结构，包含 Physics Attention 和 Slice Latent Gated Z Inject 机制。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def trunc_normal_(tensor, std=0.02):
    return nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


# -------------------------
# NEW: Slice latent gated Z injection
# -------------------------
class SliceLatentGatedZInject(nn.Module):
    """
    slice_token: [B,H,G,Dh]
    z: [B,z_dim]

    gate = gate_mlp(slice_token) -> [B,H,G,z_dim] in (0,1)
    z_gated = z * gate
    z_inj = z_proj(z_gated) -> [B,H,G,Dh]
    slice_token += z_inj
    """
    def __init__(self, dim_head: int, z_dim: int, dropout: float = 0.0):
        super().__init__()
        self.z_dim = z_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim_head, dim_head),
            nn.SiLU(),
            nn.Linear(dim_head, z_dim),
            nn.Sigmoid(),
        )
        self.z_proj = nn.Linear(z_dim, dim_head, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, slice_token: torch.Tensor, z: torch.Tensor):
        gate = self.gate_mlp(slice_token)                      # [B,H,G,z_dim]
        z_gated = z[:, None, None, :] * gate                   # [B,H,G,z_dim]
        z_inj = self.z_proj(z_gated)                           # [B,H,G,Dh]
        z_inj = self.drop(z_inj)
        return slice_token + z_inj


# -------------------------
# Core Modules
# -------------------------
class Physics_Attention_Irregular_Mesh(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        slice_num=64,
        # NEW: gated z inject after slice
        use_slice_z_inject: bool = False,
        z_dim: int = 64,
        z_inject_dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # Transolver: learnable temperature
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)

        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        # ✅ NEW: slice_latent gated z injection
        self.use_slice_z_inject = use_slice_z_inject
        self.z_inject = (
            SliceLatentGatedZInject(dim_head=dim_head, z_dim=z_dim, dropout=z_inject_dropout)
            if use_slice_z_inject
            else None
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, z=None):
        """
        x: [B, N, C]
        z: [B, z_dim] (optional)
        """
        B, N, C = x.shape

        fx_mid = (
            self.in_project_fx(x)
            .view(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # [B,H,N,Dh]

        x_mid = (
            self.in_project_x(x)
            .view(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # [B,H,N,Dh]

        # (1) Slice
        slice_logits = self.in_project_slice(x_mid) / self.temperature  # [B,H,N,G]
        slice_weights = self.softmax(slice_logits)                      # [B,H,N,G]
        slice_norm = slice_weights.sum(2)                               # [B,H,G]

        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)  # [B,H,G,Dh]
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None])

        # ✅ (1.5) NEW: gated z inject on slice latents
        if self.z_inject is not None and (z is not None):
            slice_token = self.z_inject(slice_token, z)                 # [B,H,G,Dh]

        # (2) Attention among slice tokens
        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale        # [B,H,G,G]
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v)                         # [B,H,G,Dh]

        # (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)  # [B,H,N,Dh]
        out_x = out_x.permute(0, 2, 1, 3).contiguous().view(B, N, self.heads * self.dim_head)
        return self.to_out(out_x)


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
        super().__init__()
        act_cls = ACTIVATION.get(act, None)
        if act_cls is None:
            raise NotImplementedError(f"Unknown activation: {act}")

        def make_act():
            return act_cls() if callable(act_cls) else act_cls()

        self.n_layers = n_layers
        self.res = res

        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), make_act())
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), make_act()) for _ in range(n_layers)]
        )
        self.linear_post = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            x = self.linears[i](x) + x if self.res else self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_block(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=32,
        # NEW
        use_slice_z_inject=False,
        z_dim=64,
        z_inject_dropout=0.0,
    ):
        super().__init__()
        self.last_layer = last_layer

        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
            use_slice_z_inject=use_slice_z_inject,
            z_dim=z_dim,
            z_inject_dropout=z_inject_dropout,
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
# Transolver Model (theta in coords + slice-z-inject)
# -------------------------
class Model(nn.Module):
    def __init__(
        self,
        space_dim=2,
        n_layers=4,
        n_hidden=256,
        dropout=0.0,
        n_head=8,
        act="gelu",
        mlp_ratio=1,
        fun_dim=0,
        out_dim=2,
        slice_num=32,
        ref=8,
        unified_pos=False,

        # --- theta Fourier features into coords (2D) ---
        use_theta_in_coord: bool = True,
        theta_feat_dim: int = 2,   # 固定 2: sin/cos

        # --- slice latent gated z injection ---
        use_slice_z_inject: bool = True,
        z_dim: int = 64,
        z_inject_dropout: float = 0.0,

        # --- NEW: how many layers inject z ---
        #   -1: all layers
        #    K: inject only in first K blocks (K<=n_layers)
        z_inject_layers: int = -1,
    ):
        super().__init__()
        self.__name__ = "Transolver"
        self.ref = ref
        self.unified_pos = unified_pos

        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.z_dim = z_dim

        # theta features
        self.use_theta_in_coord = use_theta_in_coord
        self.theta_feat_dim = theta_feat_dim if use_theta_in_coord else 0
        if self.use_theta_in_coord:
            assert self.theta_feat_dim == 2, "当前实现固定 theta_feat_dim=2 (sin/cos)。"

        # z inject control
        self.use_slice_z_inject = use_slice_z_inject
        self.z_inject_layers = z_inject_layers  # -1 means all layers

        # preprocess input dim
        # x will be: [coords(space_dim) (+theta_feat_dim)] (+fun_dim) (+unified_pos distances)
        if self.unified_pos:
            # ref^space_dim distances
            dist_dim = ref ** space_dim
            in_dim = fun_dim + space_dim + self.theta_feat_dim + dist_dim
        else:
            in_dim = fun_dim + space_dim + self.theta_feat_dim

        self.preprocess = MLP(
            in_dim,
            n_hidden * 2,
            n_hidden,
            n_layers=0,
            res=False,
            act=act,
        )

        # blocks
        self.blocks = nn.ModuleList(
            [
                Transolver_block(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    out_dim=out_dim,
                    slice_num=slice_num,
                    last_layer=(_ == n_layers - 1),
                    use_slice_z_inject=use_slice_z_inject,
                    z_dim=z_dim,
                    z_inject_dropout=z_inject_dropout,
                )
                for _ in range(n_layers)
            ]
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
        """
        unified_pos encoding: cdist(pos, grid_ref)
        支持 space_dim=2 或 3：
          - 2D: ref^2 grid on [-1,1]x[-1,1]
          - 3D: ref^3 grid on [-1.5,1.5]x[0,2]x[-4,4] (保持你之前风格)
        """
        B = my_pos.shape[0]
        d = my_pos.shape[-1]
        if d == 2:
            gridx = torch.linspace(-1.0, 1.0, self.ref, device=my_pos.device, dtype=my_pos.dtype)
            gridy = torch.linspace(-1.0, 1.0, self.ref, device=my_pos.device, dtype=my_pos.dtype)
            grid_ref = torch.stack(torch.meshgrid(gridx, gridy, indexing="ij"), dim=-1).reshape(1, -1, 2)
        elif d == 3:
            gridx = torch.linspace(-1.5, 1.5, self.ref, device=my_pos.device, dtype=my_pos.dtype)
            gridy = torch.linspace(0, 2, self.ref, device=my_pos.device, dtype=my_pos.dtype)
            gridz = torch.linspace(-4, 4, self.ref, device=my_pos.device, dtype=my_pos.dtype)
            grid_ref = torch.stack(torch.meshgrid(gridx, gridy, gridz, indexing="ij"), dim=-1).reshape(1, -1, 3)
        else:
            raise ValueError(f"unified_pos only supports pos dim 2 or 3, got {d}")

        grid_ref = grid_ref.repeat(B, 1, 1)     # [B, Ref^d, d]
        pos = torch.cdist(my_pos, grid_ref)     # [B, N, Ref^d]
        return pos

    @staticmethod
    def _parse_condition(condition):
        """
        condition:
          - z
          - (z, theta)
          - {"z": z, "theta": theta}
        """
        z = None
        theta = None
        if condition is None:
            return z, theta

        if isinstance(condition, dict):
            z = condition.get("z", None)
            theta = condition.get("theta", None)
        elif isinstance(condition, (tuple, list)) and len(condition) >= 2:
            z, theta = condition[0], condition[1]
        else:
            z = condition
        return z, theta

    def _theta_to_feat(self, theta: torch.Tensor, B: int, N: int, device, dtype):
        """
        theta: [B,1] or [B]
        feat:  [B,N,2] = [sin(theta), cos(theta)]
        """
        if theta is None:
            return None
        if theta.dim() == 1:
            theta = theta.view(B, 1)
        elif theta.dim() == 2 and theta.shape[1] != 1:
            # 如果你给了更多维度，这里只取第一维
            theta = theta[:, :1]
        theta = theta.to(device=device, dtype=dtype)

        sin_t = torch.sin(theta)  # [B,1]
        cos_t = torch.cos(theta)  # [B,1]
        feat = torch.cat([sin_t, cos_t], dim=-1)        # [B,2]
        feat = feat[:, None, :].expand(B, N, 2)         # [B,N,2]
        return feat

    def forward(self, data):
        """
        data = (x, pos, condition)
        x:   [B,N,space_dim]
        pos: [B,N,space_dim]
        condition: z or (z,theta) or dict
        """
        x, pos, condition = data
        B, N, _ = x.shape
        z, theta = self._parse_condition(condition)

        # ---- theta Fourier features into coords ----
        if self.use_theta_in_coord:
            theta_feat = self._theta_to_feat(theta, B, N, x.device, x.dtype)  # [B,N,2] or None
            if theta_feat is None:
                # 允许 theta 不提供：用 0 占位
                theta_feat = torch.zeros(B, N, 2, device=x.device, dtype=x.dtype)
            x = torch.cat([x, theta_feat], dim=-1)  # [B,N,space_dim+2]

        # ---- unified_pos (optional) ----
        if self.unified_pos:
            new_pos = self.get_grid(pos)            # [B,N,ref^d]
            x = torch.cat([x, new_pos], dim=-1)

        # ---- preprocess ----
        fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]

        # ---- z injection layer control ----
        if self.z_inject_layers is None or self.z_inject_layers == -1:
            z_inject_upto = len(self.blocks)
        else:
            z_inject_upto = int(max(0, min(len(self.blocks), self.z_inject_layers)))

        for i, block in enumerate(self.blocks):
            z_i = z if (self.use_slice_z_inject and (z is not None) and (i < z_inject_upto)) else None
            fx = block(fx, z=z_i)

        return fx


# -------------------------
# Quick usage
# -------------------------
if __name__ == "__main__":
# -------------------------
# Quick usage + stats
# -------------------------
    def count_trainable_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @torch.no_grad()
    def benchmark_single_sample_infer_ms(
        model: nn.Module,
        data,
        device: torch.device,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
        warmup: int = 20,
        repeat: int = 100,
    ):
        """
        Baseline-style single-sample inference time in milliseconds:
        - model.eval()
        - warmup runs (not timed)
        - timed runs with CUDA events + torch.cuda.synchronize
        Returns: (mean_ms, p50_ms, p90_ms)
        """
        model.eval()

        # move data to device (keep structure)
        x, pos, condition = data
        x = x.to(device)
        pos = pos.to(device)
        if isinstance(condition, (tuple, list)):
            condition = tuple(c.to(device) if torch.is_tensor(c) else c for c in condition)
        elif isinstance(condition, dict):
            condition = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in condition.items()}
        else:
            condition = condition.to(device) if torch.is_tensor(condition) else condition

        data = (x, pos, condition)

        # warmup
        if device.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(warmup):
            if use_amp and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    _ = model(data)
            else:
                _ = model(data)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # timed
        times_ms = []

        if device.type == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)

            for _ in range(repeat):
                starter.record()
                if use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                        _ = model(data)
                else:
                    _ = model(data)
                ender.record()
                torch.cuda.synchronize()
                times_ms.append(starter.elapsed_time(ender))  # ms
        else:
            # CPU fallback: perf_counter
            import time as _time
            for _ in range(repeat):
                t0 = _time.perf_counter()
                _ = model(data)
                t1 = _time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)

        times_ms = torch.tensor(times_ms, dtype=torch.float32)
        mean_ms = float(times_ms.mean().item())
        p50_ms = float(times_ms.median().item())
        p90_ms = float(times_ms.kthvalue(max(1, int(0.9 * len(times_ms)))).values.item())
        return mean_ms, p50_ms, p90_ms

    def bytes_to_mib(x: int) -> float:
        return float(x) / (1024 ** 2)

    if __name__ == "__main__":
        B, N = 1, 4096 # 单样本：B=1，按你训练常用点数 50k（想测 4096 就改 N）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = Model(
            space_dim=2, fun_dim=0, out_dim=2, n_hidden=256,
            use_theta_in_coord=True,
            use_slice_z_inject=True, z_dim=64,
            z_inject_layers=-1, 
            slice_num=32,
        ).to(device)

        # 统计可训练参数量
        trainable = count_trainable_params(model)
        total = sum(p.numel() for p in model.parameters())
        print(f"[Params] trainable={trainable/1e6:.3f}M ({trainable}) | total={total/1e6:.3f}M ({total})")

        # 准备单样本输入
        coords = torch.randn(B, N, 2, device=device)
        z = torch.randn(B, 64, device=device)
        theta = torch.randn(B, 1, device=device)
        data = (coords, coords, (z, theta))

        # 可选：显存峰值统计（更接近“本进程占用”）
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # 推理时间（ms）
        mean_ms, p50_ms, p90_ms = benchmark_single_sample_infer_ms(
            model=model,
            data=data,
            device=device,
            use_amp=True,                 # 你训练用 bf16，这里对齐
            amp_dtype=torch.bfloat16,
            warmup=20,
            repeat=100,
        )
        print(f"[InferTime] mean={mean_ms:.3f} ms | p50={p50_ms:.3f} ms | p90={p90_ms:.3f} ms   (B=1, N={N})")

        # 打印峰值显存
        if device.type == "cuda":
            peak_alloc = bytes_to_mib(torch.cuda.max_memory_allocated())
            peak_resv = bytes_to_mib(torch.cuda.max_memory_reserved())
            print(f"[GPU Mem Peak] allocated={peak_alloc:.1f} MiB | reserved={peak_resv:.1f} MiB")

        # 产出一次输出形状（sanity）
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out = model(data)
        print("out:", tuple(out.shape))