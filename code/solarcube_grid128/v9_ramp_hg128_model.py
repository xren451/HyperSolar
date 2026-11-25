#!/usr/bin/env python3
"""V9 extension with explicit observed-ramp attributes on site-local hyperedges.

This keeps the V9-CloudState-HG encoder and forecast heads fixed. The only
structural change is an edge-conditioned message: ramp statistics computed
from the observed 72 h station history are embedded and added to the one local
hyperedge containing that station before messages are sent to future queries.
No future irradiance/ramp values enter the graph.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from v9_ramp_hg128_encoder import EncoderCSI
from train_solarcube_v8_v3encoder_diffusion_message_csi import ConditionalResidualDiffusion


class RampConditionedHistoryFutureHypergraph(nn.Module):
    """V9 hypergraph where observed site ramp is an attribute of its local edge."""

    def __init__(self, width: int = 64, grid: int = 16, patch: int = 4):
        super().__init__()
        self.grid, self.patch = grid, patch
        self.local_edges = (grid // patch) ** 2
        self.n_nodes = 3 * grid * grid + 2
        self.n_edges = self.local_edges + 1 + grid * grid
        self.edge = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.node = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width))
        # Ramp vector: normalized mean, mean slope, max positive/negative step,
        # adjacent-valid fraction, and history-valid fraction (from V5 local TS).
        self.ramp_edge = nn.Sequential(nn.Linear(6, width), nn.GELU(), nn.Linear(width, width))
        self.ramp_gate = nn.Parameter(torch.tensor(-2.0))
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def incidence(self, position: torch.Tensor, device, dtype) -> torch.Tensor:
        b = position.shape[0]; g = self.grid; p = self.patch
        r = torch.arange(g, device=device).repeat_interleave(g)
        c = torch.arange(g, device=device).repeat(g)
        patch_id = (r // p) * (g // p) + (c // p)
        h = torch.zeros(b, self.n_nodes, self.n_edges, device=device, dtype=dtype)
        for group in range(3):
            start = group * g * g
            h[:, start:start+g*g, :self.local_edges] = F.one_hot(patch_id, self.local_edges).to(dtype)[None]
            h[:, start:start+g*g, self.local_edges] = 1
        col = torch.floor(position[:, 0] * g).long().clamp(0, g-1)
        row = torch.floor(position[:, 1] * g).long().clamp(0, g-1)
        point_patch = (row // p) * (g // p) + col // p
        local = F.one_hot(point_patch, self.local_edges).to(dtype)
        hist_point, future_point = 3*g*g, 3*g*g+1
        h[:, hist_point, :self.local_edges] = local
        h[:, hist_point, self.local_edges] = 1
        h[:, future_point, :self.local_edges] = local
        h[:, future_point, self.local_edges] = 1
        for cell in range(g*g):
            edge_id = self.local_edges + 1 + cell
            h[:, cell, edge_id] = 1
            h[:, g*g + cell, edge_id] = 1
            h[:, 2*g*g + cell, edge_id] = 1
        return h

    def forward(self, nodes: torch.Tensor, position: torch.Tensor,
                ramp_attr: torch.Tensor) -> torch.Tensor:
        # nodes [B,L,N,D]; ramp_attr [B,6] uses observed history only.
        b, leads, n, d = nodes.shape
        if n != self.n_nodes:
            raise ValueError(f"expected {self.n_nodes} HG nodes, got {n}")
        if ramp_attr.shape != (b, 6):
            raise ValueError(f"expected ramp attributes {(b, 6)}, got {tuple(ramp_attr.shape)}")
        h = self.incidence(position, nodes.device, nodes.dtype)
        edge_degree = h.sum(1).clamp_min(1)
        e = torch.einsum("bne,blnd->bled", h, nodes) / edge_degree[:, None, :, None]

        # Attach ramp information to the site-containing 4x4 local hyperedge.
        col = torch.floor(position[:, 0] * self.grid).long().clamp(0, self.grid-1)
        row = torch.floor(position[:, 1] * self.grid).long().clamp(0, self.grid-1)
        local_id = (row // self.patch) * (self.grid // self.patch) + col // self.patch
        ramp_msg = self.ramp_edge(ramp_attr)
        ramp_context = torch.zeros(b, self.n_edges, d, device=nodes.device, dtype=nodes.dtype)
        ramp_context.scatter_(1, local_id[:, None, None].expand(-1, 1, d), ramp_msg[:, None])
        e = e + torch.sigmoid(self.ramp_gate) * ramp_context[:, None]
        e = self.edge(e)

        node_degree = h.sum(-1).clamp_min(1)
        msg = torch.einsum("bne,bled->blnd", h, e) / node_degree[:, None, :, None]
        delta = self.node(torch.cat([nodes, msg], dim=-1))
        out = nodes.clone()
        future_start = 2 * self.grid * self.grid
        future_field_end = 3 * self.grid * self.grid
        future_point = self.n_nodes - 1
        gate = torch.sigmoid(self.gate)
        out[:, :, future_start:future_field_end] = nodes[:, :, future_start:future_field_end] + gate * delta[:, :, future_start:future_field_end]
        out[:, :, future_point] = nodes[:, :, future_point] + gate * delta[:, :, future_point]
        return out


class V9RampHyperedgeCSI(nn.Module):
    """V9 V5 encoder + CSI latent state + ramp-conditioned local edge + DDPM."""
    def __init__(self, width: int = 64, heads: int = 4, diffusion_steps: int = 8):
        super().__init__()
        self.base = EncoderCSI("V5", width=width, heads=heads)
        self.hg = SparseRampHG(width,grid=128,patch=32)
        self.source_norm = nn.LayerNorm(width)
        self.cloud_field = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.cloud_point = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.diffusion = ConditionalResidualDiffusion(horizon=24, field_nodes=14400,
                                                      steps=diffusion_steps, cond_dim=2*width+2)

    def _source_grid(self, sat: torch.Tensor, nwp: torch.Tensor,
                     time_hist: torch.Tensor, keep: str) -> torch.Tensor:
        nwp16 = F.interpolate(nwp.flatten(0, 1), size=(128, 128), mode="bilinear",
                              align_corners=False).view(*nwp.shape[:2], 9, 128, 128)
        sat128=F.interpolate(sat.flatten(0,1),size=(128,128),mode="bilinear",align_corners=False).view(*sat.shape[:2],6,128,128)
        if keep == "sat": x = torch.cat([sat128, torch.zeros_like(nwp16)], dim=2)
        elif keep == "nwp": x = torch.cat([torch.zeros_like(sat128), nwp16], dim=2)
        else: raise ValueError(keep)
        z = self.base.encoder(x)
        z = z + self.base.clock(time_hist)[:, :, None, None, :].permute(0, 1, 4, 2, 3)
        b, t, d, h, w = z.shape
        tok = z.permute(0, 3, 4, 1, 2).reshape(b*h*w, t, d)
        tok = self.base.temporal(tok)
        grid = (tok[:, -1] + tok.mean(1)).view(b, h, w, d).permute(0, 3, 1, 2)
        return F.interpolate(grid, (128, 128), mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        base = self.base
        joint_grid = base.encode(batch)
        sat_grid = self.source_norm(self._source_grid(batch["sat"], batch["nwp"], batch["time_hist"], "sat"))
        nwp_grid = self.source_norm(self._source_grid(batch["sat"], batch["nwp"], batch["time_hist"], "nwp"))
        b, _, d = joint_grid.shape
        tf = base.clock(batch["time_future"])
        field_future = joint_grid[:, None] + tf[:, :, None, :]
        local, ramp_attr = base.local(batch["point_hist"], batch["point_hist_mask"])
        xy = (2 * batch["position"] - 1).view(b, 1, 1, 2)
        grid_map = joint_grid.transpose(1, 2).reshape(b, d, 128, 128)
        site = F.grid_sample(grid_map, xy, mode="bilinear", padding_mode="border", align_corners=True).reshape(b, d)
        point_future = base.point_fuse(torch.cat([site, joint_grid.mean(1), local], -1))[:, None] + tf
        local_history = self.cloud_point(local)[:, None, None, :].expand(-1, tf.shape[1], 1, -1)
        nodes = torch.cat([sat_grid[:, None].expand(-1, tf.shape[1], -1, -1),
                           nwp_grid[:, None].expand(-1, tf.shape[1], -1, -1),
                           field_future, local_history, point_future[:, :, None, :]], dim=2)
        nodes = self.hg(nodes, batch["position"], ramp_attr)
        future, point = nodes[:, :, 32768:49152], nodes[:, :, 49153]
        cloud_field, cloud_point = self.cloud_field(future), self.cloud_point(point)
        field_csi = 2.5 * torch.sigmoid(base.field_csi(cloud_field)[..., 0])
        point_csi = 2.5 * torch.sigmoid(base.point_csi(cloud_point)[..., 0])
        cond = torch.cat([cloud_field.mean((1, 2)), cloud_point.mean(1),
                          field_csi.mean((1, 2))[:, None], point_csi.mean(1, keepdim=True)], dim=-1)
        return {"field_csi": field_csi, "point_csi": point_csi,
                "cloud_field": cloud_field, "cloud_point": cloud_point,
                "diffusion_cond": cond, "ramp_attr": ramp_attr}

    @staticmethod
    def _to_native120(x):
        b,h,n=x.shape
        if n != 128*128: raise ValueError(f"expected 128-grid field, got {n}")
        return F.interpolate(x.reshape(b*h,1,128,128),size=(120,120),mode="area").reshape(b,h,14400)

    def diffusion_loss(self, output, batch):
        # Residual diffusion is supervised on the observed 120x120 target grid.
        field_native=self._to_native120(output["field_csi"])
        residual=torch.cat([batch["field_csi"]-field_native,
                            (batch["point_csi"]-output["point_csi"])[...,None]],dim=-1)
        mask=torch.cat([batch["field_csi_mask"],batch["point_csi_mask"][...,None]],dim=-1)
        return self.diffusion.loss(residual,mask,output["diffusion_cond"])

    @staticmethod
    def _masked_mse(x,y,mask):
        valid=mask.bool() & torch.isfinite(x) & torch.isfinite(y)
        return torch.where(valid,(x-y).square(),torch.zeros_like(x)).sum()/valid.sum().clamp_min(1)

    def joint_loss(self,output,batch,diffusion_weight:float=0.1):
        b,h,_=output["field_csi"].shape
        cs128=F.interpolate(batch["field_cs"].reshape(b*h,1,120,120),size=(128,128),mode="bilinear",align_corners=False).reshape(b,h,16384)
        ghi_native=self._to_native120(output["field_csi"]*cs128)
        csi_native=self._to_native120(output["field_csi"])
        point_ghi=output["point_csi"]*batch["point_cs"]
        fg=self._masked_mse(ghi_native/1000,batch["field_ghi"]/1000,batch["field_mask"])
        pg=self._masked_mse(point_ghi/1000,batch["point_ghi"]/1000,batch["point_mask"])
        fc=self._masked_mse(csi_native,batch["field_csi"],batch["field_csi_mask"])
        pc=self._masked_mse(output["point_csi"],batch["point_csi"],batch["point_csi_mask"])
        diff=self.diffusion_loss(output,batch)
        det=fg+pg+0.5*(fc+pc)
        return {"loss":det+diffusion_weight*diff,"deterministic":det,"diffusion":diff,
                "field_ghi":fg,"point_ghi":pg,"field_csi":fc,"point_csi":pc}

    def forecast_samples(self,output,members:int=8):
        residual=self.diffusion.sample(output["diffusion_cond"],members=members)
        b,m,h,_=residual.shape
        r120=residual[...,:14400].reshape(b*m*h,1,120,120)
        r128=F.interpolate(r120,size=(128,128),mode="bilinear",align_corners=False).reshape(b,m,h,16384)
        field=output["field_csi"][:,None]+r128
        point=output["point_csi"][:,None,:,None]+residual[...,14400:14401]
        return torch.cat([field.clamp(0,2.5),point.clamp(0,2.5)],dim=-1)


class SparseRampHG(RampConditionedHistoryFutureHypergraph):
    def forward(self,nodes,position,ramp_attr):
        outputs=[];g=self.grid;p=self.patch;n=g*g;L=nodes.shape[1];D=nodes.shape[-1]
        cells=torch.arange(n,device=nodes.device);patch=(cells//g//p)*(g//p)+(cells%g//p)
        for b in range(nodes.shape[0]):
            row=(position[b,1]*g).long().clamp(0,g-1);col=(position[b,0]*g).long().clamp(0,g-1);pid=(row//p)*(g//p)+col//p
            ni=torch.cat([cells,cells+n,cells+2*n]);ei=patch.repeat(3)
            src=torch.cat([ni,ni,ni,torch.tensor([3*n,3*n+1,3*n,3*n+1],device=nodes.device)])
            dst=torch.cat([ei,torch.full_like(ni,self.local_edges),cells.repeat(3)+self.local_edges+1,torch.stack([pid,pid,pid*0+self.local_edges,pid*0+self.local_edges])])
            H=torch.sparse_coo_tensor(torch.stack([src,dst]),torch.ones(len(src),device=nodes.device),(3*n+2,self.n_edges)).coalesce()
            deg=torch.bincount(dst,minlength=self.n_edges).clamp_min(1).float();ndeg=torch.bincount(src,minlength=3*n+2).clamp_min(1).float()
            x=nodes[b].float();edge=torch.sparse.mm(H.t(),x.permute(1,0,2).reshape(3*n+2,L*D)).reshape(self.n_edges,L,D).permute(1,0,2)/deg[None,:,None]
            ramp=self.ramp_edge(ramp_attr[b:b+1]).float();context=torch.zeros_like(edge);context[:,pid]=ramp
            edge=self.edge(edge+torch.sigmoid(self.ramp_gate)*context)
            msg=torch.sparse.mm(H,edge.float().permute(1,0,2).reshape(self.n_edges,L*D)).reshape(3*n+2,L,D).permute(1,0,2)/ndeg[None,:,None]
            delta=self.node(torch.cat([x,msg],-1));updated=x.clone();updated[:,2*n:3*n]=x[:,2*n:3*n]+torch.sigmoid(self.gate)*delta[:,2*n:3*n];updated[:,-1]=x[:,-1]+torch.sigmoid(self.gate)*delta[:,-1];outputs.append(updated)
        return torch.stack(outputs)
