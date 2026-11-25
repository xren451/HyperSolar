#!/usr/bin/env python3
"""V9: V5 ResNet-Swin-ViT CSI forecaster + minimal future hypergraph passing.

The hypergraph is applied independently at each forecast lead to latent future
field-cell and selected-point queries. Sixteen 4x4 local hyperedges and one
global hyperedge share these *predicted latent states*; no future target or
future observation enters the message pass. All data, masks, splits and losses
match the V5 full-year SolarCube encoder screen.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_solarcube_encoders_csi import EncoderCSI, metric_rows
from train_solarcube_v7_fieldpoint_csi import SolarCubeWindows, masked_mse


class FutureHypergraph(nn.Module):
    """One local-plus-global hypergraph update over 257 future query nodes."""
    def __init__(self, width: int, grid_size: int = 16, patch_size: int = 4):
        super().__init__()
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.local_edges = (grid_size // patch_size) ** 2
        self.edge_update = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, width)
        )
        self.node_update = nn.Sequential(
            nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width)
        )
        # Start with a modest residual so the added branch cannot overwhelm
        # the matched V5 representation at initialization.
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

    def incidence(self, position: torch.Tensor) -> torch.Tensor:
        """Build [B,257,17] incidence: 16 local tiles + one global edge."""
        bsz = position.shape[0]
        dev = position.device
        row = torch.arange(self.grid_size, device=dev).repeat_interleave(self.grid_size)
        col = torch.arange(self.grid_size, device=dev).repeat(self.grid_size)
        patch_cols = self.grid_size // self.patch_size
        patch_id = (row // self.patch_size) * patch_cols + (col // self.patch_size)
        field_local = F.one_hot(patch_id, num_classes=self.local_edges).float()
        field_global = torch.ones(self.grid_size * self.grid_size, 1, device=dev)
        field_h = torch.cat([field_local, field_global], dim=-1)
        field_h = field_h.unsqueeze(0).expand(bsz, -1, -1)

        # Dataset positions are [column,row] in [0,1]; attach the point token
        # to the local tile containing its selected station.
        point_col = torch.floor(position[:, 0] * self.grid_size).long().clamp(0, self.grid_size - 1)
        point_row = torch.floor(position[:, 1] * self.grid_size).long().clamp(0, self.grid_size - 1)
        point_patch = (point_row // self.patch_size) * patch_cols + (point_col // self.patch_size)
        point_local = F.one_hot(point_patch, num_classes=self.local_edges).float()
        point_global = torch.ones(bsz, 1, device=dev)
        point_h = torch.cat([point_local, point_global], dim=-1).unsqueeze(1)
        return torch.cat([field_h, point_h], dim=1)

    def forward(self, future_nodes: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        # future_nodes: [B, lead, 257, D]
        bsz, leads, nodes, width = future_nodes.shape
        h = self.incidence(position)
        degree_edge = h.sum(dim=1).clamp_min(1.0)
        edge_state = torch.einsum("bne,bhnd->bhed", h, future_nodes)
        edge_state = edge_state / degree_edge[:, None, :, None]
        edge_message = self.edge_update(edge_state)
        degree_node = h.sum(dim=-1).clamp_min(1.0)
        node_message = torch.einsum("bne,bhed->bhnd", h, edge_message)
        node_message = node_message / degree_node[:, None, :, None]
        delta = self.node_update(torch.cat([future_nodes, node_message], dim=-1))
        return future_nodes + torch.sigmoid(self.gate_logit) * delta


class V9(nn.Module):
    """V5 deterministic CSI model with one future-query hypergraph update."""
    def __init__(self, width: int = 64, heads: int = 4):
        super().__init__()
        self.base = EncoderCSI("V5", width=width, heads=heads)
        self.hypergraph = FutureHypergraph(width=width)

    def forward(self, batch: dict[str, torch.Tensor]):
        base = self.base
        grid = base.encode(batch)  # [B,256,D], from history only
        bsz, cells, width = grid.shape
        future_clock = base.clock(batch["time_future"])
        field_query = grid[:, None] + future_clock[:, :, None, :]

        local, _ = base.local(batch["point_hist"], batch["point_hist_mask"])
        xy = (2 * batch["position"] - 1).view(bsz, 1, 1, 2)
        grid_map = grid.transpose(1, 2).reshape(bsz, width, 16, 16)
        site = F.grid_sample(
            grid_map, xy, mode="bilinear", padding_mode="border", align_corners=True
        ).reshape(bsz, width)
        point_query = base.point_fuse(
            torch.cat([site, grid.mean(1), local], dim=-1)
        )[:, None, :] + future_clock

        nodes = torch.cat([field_query, point_query[:, :, None, :]], dim=2)
        nodes = self.hypergraph(nodes, batch["position"])
        field_query, point_query = nodes[:, :, :cells], nodes[:, :, cells]
        field_csi = 2.5 * torch.sigmoid(base.field_csi(field_query)[..., 0])
        point_csi = 2.5 * torch.sigmoid(base.point_csi(point_query)[..., 0])
        return field_csi, point_csi


def mean24(rows: list[dict]) -> dict:
    return {
        m: float(np.mean([r[m] for r in rows]))
        for m in ("MAE", "RMSE", "CRPS") if rows
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--sitelist", required=True)
    ap.add_argument("--nwp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("V9 full SolarCube experiment requires CUDA")

    datasets = {
        split: SolarCubeWindows(a.data, a.prepared, a.sitelist, a.nwp, split, stride=a.stride)
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(
            ds, batch_size=a.batch, shuffle=(split == "train"),
            num_workers=0, pin_memory=True,
        )
        for split, ds in datasets.items()
    }
    device = torch.device("cuda")
    model = V9().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    log = (out / "train.log").open("w", buffering=1)
    start = time.time()
    config = {
        "version": "V9-V5-FutureHypergraph-CSI",
        "dataset": "SolarCube 2018 all 19 regions",
        "protocol": "matched to V5: chronological 70/15/15; history 72h; horizon 24h; stride 3h",
        "encoder": "same ENC-RES-SWIN-VIT ResNet+Swin+global-ViT raster encoder and temporal/local branches as V5",
        "hypergraph": {
            "purpose": "future latent information transfer before field/point CSI decoding",
            "nodes": "256 future field-cell queries plus one selected-point query at each lead",
            "hyperedges": "16 non-overlapping 4x4 local tiles plus one global region edge",
            "message": "mean node-to-hyperedge aggregation, learned edge update, normalized hyperedge-to-node residual update",
            "future_target_leakage": False,
        },
        "outputs": "field and selected-point CSI; GHI reconstructed as CSI times future Haurwitz clear-sky GHI",
        "loss": "same masked field-GHI + point-GHI + 0.5*(field-CSI + point-CSI) objective as V5",
        "probability": "deterministic; CRPS=MAE; prediction intervals not applicable",
        "early_stopping": {"monitor": "validation_joint_loss", "patience": a.patience, "min_epochs": 5, "restore_best": True},
        "batch": a.batch, "stride_hours": a.stride, "seed": a.seed, "max_epochs": a.epochs,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "README.md").write_text(
        "# V9 SolarCube future-hypergraph CSI experiment\n\n"
        "V9 keeps the completed V5 encoder, data protocol, split and losses. Its only intended structural change is a lightweight hypergraph update over forecast-query latent nodes at each lead: 16 local 4x4 tile edges plus one global edge connect future field-cell tokens and the selected-point token. The update sees only encoded history and known future time/lead embeddings; future irradiance labels are used only in the loss.\n\n"
        "Artifacts: full train.log, config.json, README.md, best.pt, training_history.json, metrics.json, and test_predictions.npz.\n\n"
        + json.dumps(config, indent=2) + "\n"
    )
    log.write(json.dumps({
        "event": "started", "train_windows": len(datasets["train"]),
        "val_windows": len(datasets["val"]), "test_windows": len(datasets["test"]),
        "parameters": sum(p.numel() for p in model.parameters()),
        "early_stopping": config["early_stopping"], "gpu": torch.cuda.get_device_name(0),
    }) + "\n")

    def move(batch):
        return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

    best, best_epoch, bad = float("inf"), 0, 0
    history = []
    for epoch in range(1, a.epochs + 1):
        model.train()
        train_losses = []
        for raw in loaders["train"]:
            b = move(raw)
            optimizer.zero_grad(set_to_none=True)
            fcsi, pcsi = model(b)
            fghi, pghi = fcsi * b["field_cs"], pcsi * b["point_cs"]
            fl = masked_mse((fghi - b["field_ghi"]) / 1000, torch.zeros_like(fghi), b["field_mask"])
            pl = masked_mse((pghi - b["point_ghi"]) / 1000, torch.zeros_like(pghi), b["point_mask"])
            fcl = masked_mse(fcsi, b["field_csi"], b["field_csi_mask"])
            pcl = masked_mse(pcsi, b["point_csi"], b["point_csi_mask"])
            loss = fl + pl + 0.5 * (fcl + pcl)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append([float(loss.detach()), float(fl.detach()), float(pl.detach()), float(fcl.detach()), float(pcl.detach())])

        model.eval()
        val_losses = []
        with torch.no_grad():
            for raw in loaders["val"]:
                b = move(raw)
                fcsi, pcsi = model(b)
                fghi, pghi = fcsi * b["field_cs"], pcsi * b["point_cs"]
                val_losses.append(float(
                    masked_mse((fghi - b["field_ghi"]) / 1000, torch.zeros_like(fghi), b["field_mask"])
                    + masked_mse((pghi - b["point_ghi"]) / 1000, torch.zeros_like(pghi), b["point_mask"])
                    + 0.5 * (masked_mse(fcsi, b["field_csi"], b["field_csi_mask"])
                             + masked_mse(pcsi, b["point_csi"], b["point_csi_mask"]))
                ))
        val = float(np.mean(val_losses))
        row = {
            "epoch": epoch,
            "train_loss_components": np.mean(train_losses, axis=0).tolist(),
            "val_joint_loss": val,
            "elapsed_sec": time.time() - start,
            "peak_gpu_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
        }
        history.append(row)
        log.write(json.dumps(row) + "\n")
        if val < best - 1e-7:
            best, best_epoch, bad = val, epoch, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
        if epoch >= 5 and bad >= a.patience:
            log.write(json.dumps({"event": "early_stopping", "epoch": epoch, "best_epoch": best_epoch, "patience": a.patience}) + "\n")
            break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True))
    model.eval()
    arrays = {k: [] for k in ["field_ghi_prediction", "point_ghi_prediction", "field_ghi_truth", "point_ghi_truth", "field_mask", "point_mask", "field_csi_prediction", "point_csi_prediction", "field_csi_truth", "point_csi_truth", "field_csi_mask", "point_csi_mask", "region", "origin"]}
    with torch.no_grad():
        for raw in loaders["test"]:
            b = move(raw)
            fcsi, pcsi = model(b)
            fg, pg = fcsi * b["field_cs"], pcsi * b["point_cs"]
            values = {
                "field_ghi_prediction": fg, "point_ghi_prediction": pg,
                "field_ghi_truth": b["field_ghi"], "point_ghi_truth": b["point_ghi"],
                "field_mask": b["field_mask"], "point_mask": b["point_mask"],
                "field_csi_prediction": fcsi, "point_csi_prediction": pcsi,
                "field_csi_truth": b["field_csi"], "point_csi_truth": b["point_csi"],
                "field_csi_mask": b["field_csi_mask"], "point_csi_mask": b["point_csi_mask"],
                "region": b["region"], "origin": b["origin"],
            }
            for k, v in values.items(): arrays[k].append(v.detach().cpu().numpy())
    arrays = {k: np.concatenate(v) for k, v in arrays.items()}
    metrics = {
        "version": config["version"], "encoder": "V5 ENC-RES-SWIN-VIT",
        "best_epoch": best_epoch, "epochs_ran": len(history), "train_seconds": time.time() - start,
        "test_windows": len(arrays["field_ghi_truth"]), "parameters": sum(p.numel() for p in model.parameters()),
        "peak_gpu_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
        "field_GHI_by_lead": metric_rows(arrays["field_ghi_prediction"], arrays["field_ghi_truth"], arrays["field_mask"]),
        "point_GHI_by_lead": metric_rows(arrays["point_ghi_prediction"], arrays["point_ghi_truth"], arrays["point_mask"]),
        "field_CSI_by_lead": metric_rows(arrays["field_csi_prediction"], arrays["field_csi_truth"], arrays["field_csi_mask"]),
        "point_CSI_by_lead": metric_rows(arrays["point_csi_prediction"], arrays["point_csi_truth"], arrays["point_csi_mask"]),
    }
    for name, key in [("field_GHI_mean24", "field_GHI_by_lead"), ("point_GHI_mean24", "point_GHI_by_lead"), ("field_CSI_mean24", "field_CSI_by_lead"), ("point_CSI_mean24", "point_CSI_by_lead")]:
        metrics[name] = mean24(metrics[key])
    np.savez_compressed(out / "test_predictions.npz", **arrays)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    log.write(json.dumps({"event": "finished", "result": metrics}) + "\n")
    log.close()


if __name__ == "__main__":
    main()
