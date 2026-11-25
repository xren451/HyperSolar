"""Reusable architecture blocks for the SMF registry screen.

This module intentionally contains architecture primitives, not dataset logic.
Dataset adapters and the trainer must state whether inputs are dense raster
frames or sparse station tokens; the two protocols must never be conflated.
Tensor conventions:
  raster encoder: [B,T,C,H,W] -> [B,T,D,H/8,W/8]
  station cloud branch: [B,L,N,D] and historical ramp tokens [B,N,D_r].
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


def _conv(c0: int, c1: int, stride: int = 1) -> nn.Sequential:
    groups = min(8, c1)
    while c1 % groups:
        groups -= 1
    return nn.Sequential(
        nn.Conv2d(c0, c1, 3, stride, 1, bias=False), nn.GroupNorm(groups, c1), nn.GELU(),
        nn.Conv2d(c1, c1, 3, 1, 1, bias=False), nn.GroupNorm(groups, c1), nn.GELU())


class PeriodicPosition(nn.Module):
    """Calendar/lead periodic features, separate from learned coordinates."""
    def __init__(self, width: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(8, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        # phase[...,0:2] are normalized daily and annual phases when supplied;
        # otherwise callers pass normalized lead/hour and normalized day-of-year.
        if phase.shape[-1] < 2:
            raise ValueError("periodic PE requires daily and annual phase columns")
        daily, annual = phase[..., 0], phase[..., 1]
        z = torch.stack([torch.sin(2*math.pi*daily), torch.cos(2*math.pi*daily),
                         torch.sin(2*math.pi*annual), torch.cos(2*math.pi*annual),
                         daily, annual,
                         torch.sin(4*math.pi*daily), torch.cos(4*math.pi*daily)], -1)
        return self.proj(z)


def apply_rope(q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
    """Apply rotary position encoding to [B,H,L,D] query/key tensors."""
    d = q.shape[-1]
    rotary = d - (d % 2)
    if rotary == 0:
        return q, k
    inv = 1.0 / (10000 ** (torch.arange(0, rotary, 2, device=q.device, dtype=q.dtype) / rotary))
    angle = positions.to(q.dtype)[..., None] * inv
    sin = angle.sin().unsqueeze(1)
    cos = angle.cos().unsqueeze(1)

    def rotate(x):
        xr, tail = x[..., :rotary], x[..., rotary:]
        a, b = xr[..., 0::2], xr[..., 1::2]
        out = torch.stack([a*cos - b*sin, a*sin + b*cos], -1).flatten(-2)
        return torch.cat([out, tail], -1)
    return rotate(q), rotate(k)


class RotaryAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float = 0.0):
        super().__init__()
        if width % heads:
            raise ValueError("width must divide heads")
        self.width, self.heads, self.dk = width, heads, width // heads
        self.qkv = nn.Linear(width, 3*width)
        self.proj = nn.Linear(width, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, mask=None):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, self.dk).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, k, positions)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.drop.p if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(b, n, d))


class RotaryBlock(nn.Module):
    def __init__(self, width: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = RotaryAttention(width, heads)
        self.mlp = nn.Sequential(nn.Linear(width, mlp_ratio*width), nn.GELU(), nn.Linear(mlp_ratio*width, width))

    def forward(self, x, positions, mask=None):
        x = x + self.attn(self.n1(x), positions, mask)
        return x + self.mlp(self.n2(x))


class WindowBlock(nn.Module):
    """Local window attention on an 8x8 latent grid, with optional half-window shift."""
    def __init__(self, width: int, heads: int, window: int = 4, shift: int = 0):
        super().__init__()
        self.width, self.heads, self.window, self.shift = width, heads, window, shift
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B,T,D,H,W]. Pad latent dimensions so arbitrary patch grids work.
        b, t, d, h, w = x.shape
        s = self.shift
        if s:
            x = torch.roll(x, shifts=(-s, -s), dims=(-2, -1))
        ph, pw = (-h) % self.window, (-w) % self.window
        z = F.pad(x, (0, pw, 0, ph))
        hp, wp = z.shape[-2:]
        z = z.permute(0, 1, 3, 4, 2).reshape(b*t, hp//self.window, self.window, wp//self.window, self.window, d)
        z = z.permute(0, 1, 3, 2, 4, 5).reshape(-1, self.window*self.window, d)
        a = self.norm1(z)
        z = z + self.attn(a, a, a, need_weights=False)[0]
        z = z + self.mlp(self.norm2(z))
        z = z.reshape(b*t, hp//self.window, wp//self.window, self.window, self.window, d)
        z = z.permute(0, 1, 3, 2, 4, 5).reshape(b, t, hp, wp, d)[..., :h, :w, :]
        z = z.permute(0, 1, 4, 2, 3)
        return torch.roll(z, shifts=(s, s), dims=(-2, -1)) if s else z


class RasterEncoder(nn.Module):
    """V2 / Vision-Swin / residual-ViT / residual-Swin-global-ViT backbones."""
    NAMES = {"SMF-V2", "ENC-VISION-SWIN", "ENC-RVIT", "ENC-RES-SWIN-VIT"}

    def __init__(self, channels: int, width: int = 128, heads: int = 4, name: str = "SMF-V2", preserve_grid: bool = False):
        super().__init__()
        if name not in self.NAMES:
            raise ValueError(f"unknown raster encoder {name}")
        self.name = name
        # Small SolarCube tiles are already 16x16. Keep an 8x8 latent grid so
        # Swin windows retain meaningful local structure instead of padding a
        # 2x2 feature map to one 4x4 window.
        self.preserve_grid = preserve_grid
        self.stem = _conv(channels, 64, 2)
        self.grid_projection = nn.Conv2d(64, width, 1) if preserve_grid else nn.Identity()
        self.down1 = _conv(64, 96, 2)
        self.down2 = _conv(96, width, 2)
        if name == "ENC-VISION-SWIN":
            self.local = nn.ModuleList([WindowBlock(width, heads, 4, 0), WindowBlock(width, heads, 4, 2)])
            self.global_blocks = nn.ModuleList()
        elif name == "ENC-RVIT":
            self.local = nn.ModuleList()
            self.global_blocks = nn.ModuleList([RotaryBlock(width, heads), RotaryBlock(width, heads)])
        elif name == "ENC-RES-SWIN-VIT":
            self.local = nn.ModuleList([WindowBlock(width, heads, 4, 0), WindowBlock(width, heads, 4, 2)])
            self.global_blocks = nn.ModuleList([RotaryBlock(width, heads)])
        else:
            self.local = nn.ModuleList()
            enc = nn.TransformerEncoderLayer(width, heads, 4*width, .1, batch_first=True, norm_first=True, activation="gelu")
            self.global_blocks = nn.TransformerEncoder(enc, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("raster input must be [B,T,C,H,W]")
        b, t, c, h, w = x.shape
        z = self.stem(x.reshape(b*t, c, h, w))
        if self.preserve_grid:
            z = self.grid_projection(z)
            oh, ow = z.shape[-2:]
        else:
            z = self.down2(self.down1(z))
            oh, ow = h//8, w//8
        z = z.view(b, t, -1, oh, ow)
        if self.local:
            for layer in self.local:
                z = layer(z)
        if self.global_blocks:
            bt, d, gh, gw = b*t, z.shape[2], z.shape[3], z.shape[4]
            tokens = z.permute(0, 1, 3, 4, 2).reshape(bt, gh*gw, d)
            if isinstance(self.global_blocks, nn.TransformerEncoder):
                tokens = self.global_blocks(tokens)
            else:
                pos = torch.arange(gh*gw, device=x.device, dtype=x.dtype)[None].expand(bt, -1)
                for layer in self.global_blocks:
                    tokens = layer(tokens, pos)
            z = tokens.view(b, t, gh, gw, d).permute(0, 1, 4, 2, 3)
        return z


class SparseKNNMessagePassing(nn.Module):
    """Sparse edge-index message passing; no dense N x N adjacency allocation."""
    def __init__(self, width: int):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(2*width + 2, width), nn.GELU(), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, coords: torch.Tensor, edge_index: torch.Tensor):
        # x [B,L,N,D], coords [N,2], edge_index [2,E] directed source->target.
        src, dst = edge_index[0].long(), edge_index[1].long()
        rel = coords[src] - coords[dst]
        m = self.msg(torch.cat([x[..., src, :], x[..., dst, :], rel.expand(*x.shape[:-2], -1, -1)], -1))
        agg = torch.zeros_like(x)
        agg.index_add_(-2, dst, m)
        deg = torch.bincount(dst, minlength=x.shape[-2]).to(x.dtype).clamp_min(1)
        return self.norm(x + agg / deg.view(*([1] * (x.ndim-2)), -1, 1))


class RampEventEncoder(nn.Module):
    """Causal ramp summary from observed history only; input [B,H,N] and mask."""
    def __init__(self, width: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(5, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, history: torch.Tensor, mask: torch.Tensor):
        if history.ndim != 3 or mask.shape != history.shape:
            raise ValueError("history and mask must be [B,H,N]")
        m = mask.bool() & torch.isfinite(history)
        y = torch.where(m, history, torch.zeros_like(history))
        dy = F.pad(y[:, 1:] - y[:, :-1], (0, 0, 1, 0))
        valid = m & F.pad(m[:, 1:] & m[:, :-1], (0, 0, 1, 0))
        weights = valid.to(y.dtype)
        denom = weights.sum(1).clamp_min(1)
        mean_slope = (dy * weights).sum(1) / denom
        max_up = dy.masked_fill(~valid, -torch.inf).amax(1)
        max_dn = dy.masked_fill(~valid, torch.inf).amin(1)
        max_up = torch.where(torch.isfinite(max_up), max_up, 0)
        max_dn = torch.where(torch.isfinite(max_dn), max_dn, 0)
        # Latest event time is normalized to [-1,0], and quality is observed fraction.
        idx = torch.arange(history.shape[1], device=history.device, dtype=y.dtype)
        latest = ((weights * idx[None, :, None]).amax(1) / max(history.shape[1]-1, 1))
        quality = m.to(y.dtype).mean(1)
        feat = torch.stack([mean_slope, max_up, max_dn, latest, quality], -1)
        return self.proj(feat)


class CloudRampHead(nn.Module):
    """Cloud1/2/3 state and interaction head for station/grid token sequences."""
    def __init__(self, width: int, variant: str, heads: int = 4):
        super().__init__()
        if variant not in ("Cloud1", "Cloud2", "Cloud3"):
            raise ValueError("variant must be Cloud1, Cloud2, or Cloud3")
        self.variant = variant
        self.ramp = RampEventEncoder(width)
        self.state = nn.Sequential(nn.Linear(2*width, width), nn.GELU(), nn.Linear(width, width))
        self.future_state = nn.Sequential(nn.Linear(2*width, width), nn.GELU(), nn.Linear(width, width))
        self.cross = nn.MultiheadAttention(width, heads, batch_first=True) if variant != "Cloud1" else None
        self.graph = SparseKNNMessagePassing(width) if variant != "Cloud1" else None
        self.refine = nn.Sequential(nn.Linear(3*width, width), nn.GELU(), nn.Linear(width, width)) if variant == "Cloud3" else None

    def forward(self, regional_state: torch.Tensor, historical_ramp: torch.Tensor,
                field_context: torch.Tensor, point_context: torch.Tensor | None = None,
                coords: torch.Tensor | None = None, edge_index: torch.Tensor | None = None):
        # regional_state [B,L,N,D], historical_ramp [B,H,N], field_context [B,L,N,D].
        ramp = self.ramp(historical_ramp[0], historical_ramp[1])
        ramp = ramp[:, None].expand(-1, regional_state.shape[1], -1, -1)
        if self.variant == "Cloud1":
            delta = self.state(torch.cat([regional_state, ramp], -1))
            fused = self.future_state(torch.cat([regional_state + delta, field_context], -1))
            return fused, delta
        if coords is None or edge_index is None:
            raise ValueError("Cloud2/3 require validated sparse coordinates and edge_index")
        state = self.graph(regional_state + ramp, coords, edge_index)
        q = state.flatten(1, 2); kv = field_context.flatten(1, 2)
        exchange, _ = self.cross(q, kv, kv, need_weights=False)
        exchange = exchange.view_as(state)
        fused = self.future_state(torch.cat([state + exchange, field_context], -1))
        if self.variant == "Cloud3":
            if point_context is None:
                raise ValueError("Cloud3 requires preliminary point prediction/context")
            p = point_context[:, :, None, :].expand(-1, -1, fused.shape[2], -1)
            fused = fused + self.refine(torch.cat([fused, field_context, p], -1))
        return fused, fused - regional_state
