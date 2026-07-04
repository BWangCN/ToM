"""Flow-matching DiT action expert (CLAUDE.md §0 Decoder row).

d_model=512, depth=8, heads=8, AdaLN(-Zero) time conditioning, cross-attention
over projected VLM hidden states. Rectified flow: x_t=(1-t)x0 + t*x1 with
x0~N(0,I); the network predicts velocity v = x1 - x0; MSE loss; 10 Euler steps
at inference. Runs in fp32 (it is tiny); condition states arrive in bf16 and
are cast (and, per knowledge insulation, already detached upstream).
"""

import math

import torch
import torch.nn as nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class DiTBlock(nn.Module):
    def __init__(self, d, heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm3 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, d * mlp_ratio), nn.GELU(), nn.Linear(d * mlp_ratio, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 9 * d))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, t_emb, cond, cond_mask):
        (s1, g1, b1, s2, g2, b2, s3, g3, b3) = self.ada(t_emb)[:, None, :].chunk(9, dim=-1)
        h = self.norm1(x) * (1 + s1) + b1
        x = x + g1 * self.attn(h, h, h, need_weights=False)[0]
        h = self.norm2(x) * (1 + s2) + b2
        x = x + g2 * self.cross(h, cond, cond, key_padding_mask=cond_mask, need_weights=False)[0]
        h = self.norm3(x) * (1 + s3) + b3
        x = x + g3 * self.mlp(h)
        return x


class ActionExpert(nn.Module):
    def __init__(self, d_cond: int, d_model=512, depth=8, n_heads=8, mlp_ratio=4,
                 chunk_len=8, action_dim=2):
        super().__init__()
        self.chunk_len, self.action_dim = chunk_len, action_dim
        self.x_in = nn.Linear(action_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, chunk_len, d_model) * 0.02)
        self.t_mlp = nn.Sequential(nn.Linear(256, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.cond_norm = nn.LayerNorm(d_cond)
        self.cond_proj = nn.Linear(d_cond, d_model)
        self.blocks = nn.ModuleList(DiTBlock(d_model, n_heads, mlp_ratio) for _ in range(depth))
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.x_out = nn.Linear(d_model, action_dim)
        nn.init.zeros_(self.final_ada[1].weight)
        nn.init.zeros_(self.final_ada[1].bias)
        nn.init.zeros_(self.x_out.weight)
        nn.init.zeros_(self.x_out.bias)

    def forward(self, x_t, t, cond, cond_mask=None):
        """x_t (B,H,2) fp32; t (B,) in [0,1]; cond (B,Lc,d_cond); cond_mask True=PAD."""
        cond = self.cond_proj(self.cond_norm(cond.float()))
        t_emb = self.t_mlp(timestep_embedding(t, 256))
        x = self.x_in(x_t) + self.pos
        for blk in self.blocks:
            x = blk(x, t_emb, cond, cond_mask)
        s, b = self.final_ada(t_emb)[:, None, :].chunk(2, dim=-1)
        return self.x_out(self.final_norm(x) * (1 + s) + b)

    def flow_loss(self, x1, cond, cond_mask=None, generator=None):
        """Rectified-flow MSE. x1 (B,H,2) = GT chunk (fp32)."""
        B = x1.shape[0]
        dev = x1.device
        t = torch.rand(B, device=dev, generator=generator)
        x0 = torch.randn(x1.shape, device=dev, generator=generator)
        x_t = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
        v_target = x1 - x0
        v_pred = self(x_t, t, cond, cond_mask)
        return nn.functional.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample(self, cond, cond_mask=None, steps=10, generator=None):
        B = cond.shape[0]
        dev = cond.device
        x = torch.randn(B, self.chunk_len, self.action_dim, device=dev, generator=generator)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((B,), i * dt, device=dev)
            x = x + self(x, t, cond, cond_mask) * dt
        return x.clamp(-1.0, 1.0)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
