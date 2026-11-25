#!/usr/bin/env python3
"""V9-CloudHG model components: latent cloud-state fusion + history/future HG.

This module is deliberately separate from the registered V9 trainer so that
existing V5/V9 experiments and their checkpoints remain reproducible.
Cloud state is an *effective latent cloud/radiative state* supervised through
CSI when COD/cloud labels are unavailable. No future observation is an input.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from train_solarcube_encoders_csi import EncoderCSI
from train_solarcube_v8_v3encoder_diffusion_message_csi import ConditionalResidualDiffusion


class HistoryFutureCloudHypergraph(nn.Module):
    """Source-aware hypergraph; history informs future queries, never vice versa.

    Nodes per lead: satellite history grid (256), NWP history grid (256),
    joint future grid queries (256), point-history token (1), point-future
    query (1). Hyperedges include local 4x4 neighborhoods, one regional edge,
    and same-cell source/future edges. Historical nodes are read-only memory.
    """
    def __init__(self, width: int = 64, grid: int = 16, patch: int = 4):
        super().__init__()
        self.grid, self.patch = grid, patch
        self.local_edges = (grid // patch) ** 2
        self.n_nodes = 3 * grid * grid + 2
        self.n_edges = self.local_edges + 1 + grid * grid
        self.edge = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.node = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width))
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def incidence(self, position: torch.Tensor, device, dtype) -> torch.Tensor:
        b = position.shape[0]; g = self.grid; p = self.patch
        r = torch.arange(g, device=device).repeat_interleave(g)
        c = torch.arange(g, device=device).repeat(g)
        patch_id = (r // p) * (g // p) + (c // p)
        h = torch.zeros(b, self.n_nodes, self.n_edges, device=device, dtype=dtype)
        # Three grid node sets: satellite history, NWP history, future queries.
        for group in range(3):
            start = group * g * g
            h[:, start:start+g*g, :self.local_edges] = F.one_hot(
                patch_id, self.local_edges).to(dtype)[None]
            h[:, start:start+g*g, self.local_edges] = 1
        # Local/global assignment for historical and future point tokens.
        col = torch.floor(position[:, 0] * g).long().clamp(0, g-1)
        row = torch.floor(position[:, 1] * g).long().clamp(0, g-1)
        point_patch = (row // p) * (g // p) + col // p
        local = F.one_hot(point_patch, self.local_edges).to(dtype)
        hist_point, future_point = 3*g*g, 3*g*g+1
        h[:, hist_point, :self.local_edges] = local
        h[:, hist_point, self.local_edges] = 1
        h[:, future_point, :self.local_edges] = local
        h[:, future_point, self.local_edges] = 1
        # Per-cell edges explicitly couple satellite, NWP, and future field state.
        for cell in range(g*g):
            edge_id = self.local_edges + 1 + cell
            h[:, cell, edge_id] = 1
            h[:, g*g + cell, edge_id] = 1
            h[:, 2*g*g + cell, edge_id] = 1
        return h

    def forward(self, nodes: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        # nodes [B,L,N,D]. Historical nodes participate in messages but are frozen.
        b, leads, n, d = nodes.shape
        if n != self.n_nodes:
            raise ValueError(f"expected {self.n_nodes} HG nodes, got {n}")
        h = self.incidence(position, nodes.device, nodes.dtype)
        edge_degree = h.sum(1).clamp_min(1)
        e = torch.einsum("bne,blnd->bled", h, nodes) / edge_degree[:, None, :, None]
        e = self.edge(e)
        node_degree = h.sum(-1).clamp_min(1)
        msg = torch.einsum("bne,bled->blnd", h, e) / node_degree[:, None, :, None]
        delta = self.node(torch.cat([nodes, msg], dim=-1))
        out = nodes.clone()
        future_start = 2 * self.grid * self.grid
        future_field_end = 3 * self.grid * self.grid
        future_point = self.n_nodes - 1
        out[:, :, future_start:future_field_end] = (
            nodes[:, :, future_start:future_field_end]
            + torch.sigmoid(self.gate) * delta[:, :, future_start:future_field_end]
        )
        out[:, :, future_point] = (
            nodes[:, :, future_point]
            + torch.sigmoid(self.gate) * delta[:, :, future_point]
        )
        return out


class V9CloudStateHG(nn.Module):
    """V5 encoder + interpretable latent cloud-state fusion + residual DDPM."""
    def __init__(self, width: int = 64, heads: int = 4, diffusion_steps: int = 8):
        super().__init__()
        self.base = EncoderCSI("V5", width=width, heads=heads)
        self.hg = HistoryFutureCloudHypergraph(width)
        # Source-isolated passes share the pretrained-by-training V5 raster weights.
        self.source_norm = nn.LayerNorm(width)
        self.cloud_field = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.cloud_point = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.diffusion = ConditionalResidualDiffusion(
            horizon=24, field_nodes=256, steps=diffusion_steps, cond_dim=2*width+2
        )

    def _source_grid(self, sat: torch.Tensor, nwp: torch.Tensor,
                     time_hist: torch.Tensor, keep: str) -> torch.Tensor:
        """Encode one source alone; zero the other source before shared raster encoder."""
        nwp16 = F.interpolate(nwp.flatten(0, 1), size=(16, 16), mode="bilinear",
                              align_corners=False).view(*nwp.shape[:2], 9, 16, 16)
        if keep == "sat":
            x = torch.cat([sat, torch.zeros_like(nwp16)], dim=2)
        elif keep == "nwp":
            x = torch.cat([torch.zeros_like(sat), nwp16], dim=2)
        else:
            raise ValueError(keep)
        z = self.base.encoder(x)
        z = z + self.base.clock(time_hist)[:, :, None, None, :].permute(0, 1, 4, 2, 3)
        b, t, d, h, w = z.shape
        tok = z.permute(0, 3, 4, 1, 2).reshape(b*h*w, t, d)
        tok = self.base.temporal(tok)
        grid = (tok[:, -1] + tok.mean(1)).view(b, h, w, d).permute(0, 3, 1, 2)
        return F.interpolate(grid, (16, 16), mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        base = self.base
        joint_grid = base.encode(batch)                         # [B,256,D]
        sat_grid = self._source_grid(batch["sat"], batch["nwp"], batch["time_hist"], "sat")
        nwp_grid = self._source_grid(batch["sat"], batch["nwp"], batch["time_hist"], "nwp")
        sat_grid, nwp_grid = self.source_norm(sat_grid), self.source_norm(nwp_grid)
        b, cells, d = joint_grid.shape
        tf = base.clock(batch["time_future"])
        field_future = joint_grid[:, None] + tf[:, :, None, :]
        local, _ = base.local(batch["point_hist"], batch["point_hist_mask"])
        xy = (2 * batch["position"] - 1).view(b, 1, 1, 2)
        grid_map = joint_grid.transpose(1, 2).reshape(b, d, 16, 16)
        site = F.grid_sample(grid_map, xy, mode="bilinear", padding_mode="border",
                             align_corners=True).reshape(b, d)
        point_future = base.point_fuse(torch.cat([site, joint_grid.mean(1), local], -1))[:, None] + tf
        local_history = self.cloud_point(local)[:, None, None, :].expand(-1, tf.shape[1], 1, -1)
        nodes = torch.cat([
            sat_grid[:, None].expand(-1, tf.shape[1], -1, -1),
            nwp_grid[:, None].expand(-1, tf.shape[1], -1, -1),
            field_future,
            local_history,
            point_future[:, :, None, :],
        ], dim=2)
        nodes = self.hg(nodes, batch["position"])
        future = nodes[:, :, 512:768]
        point = nodes[:, :, 769]
        cloud_field = self.cloud_field(future)
        cloud_point = self.cloud_point(point)
        field_csi = 2.5 * torch.sigmoid(base.field_csi(cloud_field)[..., 0])
        point_csi = 2.5 * torch.sigmoid(base.point_csi(cloud_point)[..., 0])
        cond = torch.cat([cloud_field.mean((1, 2)), cloud_point.mean(1),
                          field_csi.mean((1, 2), keepdim=False)[:, None],
                          point_csi.mean(1, keepdim=True)], dim=-1)
        return {"field_csi": field_csi, "point_csi": point_csi,
                "cloud_field": cloud_field, "cloud_point": cloud_point,
                "diffusion_cond": cond}

    def diffusion_loss(self, output: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """DDPM learns the CSI residual around deterministic cloud-state forecasts."""
        residual = torch.cat([
            batch["field_csi"] - output["field_csi"],
            (batch["point_csi"] - output["point_csi"])[..., None],
        ], dim=-1)
        mask = torch.cat([batch["field_csi_mask"], batch["point_csi_mask"][..., None]], dim=-1)
        return self.diffusion.loss(residual, mask, output["diffusion_cond"])

    @staticmethod
    def _masked_mse(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.bool() & torch.isfinite(x) & torch.isfinite(y)
        err = torch.where(valid, (x-y).square(), torch.zeros_like(x))
        return err.sum() / valid.sum().clamp_min(1)

    def joint_loss(self, output: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
                   diffusion_weight: float = 0.1) -> Dict[str, torch.Tensor]:
        """Train-ready objective: field/point GHI + CSI supervision + residual DDPM."""
        field_ghi = output["field_csi"] * batch["field_cs"]
        point_ghi = output["point_csi"] * batch["point_cs"]
        field_ghi_loss = self._masked_mse(field_ghi / 1000, batch["field_ghi"] / 1000,
                                          batch["field_mask"])
        point_ghi_loss = self._masked_mse(point_ghi / 1000, batch["point_ghi"] / 1000,
                                          batch["point_mask"])
        field_csi_loss = self._masked_mse(output["field_csi"], batch["field_csi"],
                                          batch["field_csi_mask"])
        point_csi_loss = self._masked_mse(output["point_csi"], batch["point_csi"],
                                          batch["point_csi_mask"])
        diffusion = self.diffusion_loss(output, batch)
        deterministic = field_ghi_loss + point_ghi_loss + 0.5 * (field_csi_loss + point_csi_loss)
        return {"loss": deterministic + diffusion_weight * diffusion,
                "deterministic": deterministic, "diffusion": diffusion,
                "field_ghi": field_ghi_loss, "point_ghi": point_ghi_loss,
                "field_csi": field_csi_loss, "point_csi": point_csi_loss}

    def forecast_samples(self, output: Dict[str, torch.Tensor], members: int = 8) -> torch.Tensor:
        residual = self.diffusion.sample(output["diffusion_cond"], members=members)
        field = output["field_csi"][:, None] + residual[..., :256]
        point = output["point_csi"][:, None, :, None] + residual[..., 256:257]
        return torch.cat([field.clamp_(0, 2.5), point.clamp_(0, 2.5)], dim=-1)
