#!/usr/bin/env python3
"""V8: V3 deterministic raster encoder + sampled residual diffusion/message flow.

The V3 Vision-Swin raster encoder, temporal Transformer, and local multiscale
station-history encoder are kept intact. A conditional DDPM models the joint
field/point CSI residual around V3's deterministic forecast. Its differentiable
reverse samples are passed into both directions of the field/point message
module before the refined CSI heads run. This differs from V7, where only a
deterministic residual hint entered message passing and diffusion samples were
added after refinement.

Data windows and clear-sky/CSI labels are supplied by the audited SolarCube
loader in train_solarcube_v7_fieldpoint_csi.py. The default protocol matches
the completed V3 full19/full-year experiment (72h history, 24h lead, stride 3,
chronological split), with patience-based early stopping and best-checkpoint
restore.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent if CODE_DIR.name == "code" else CODE_DIR
for import_dir in (ROOT, CODE_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from models_registry import RasterEncoder
from train_solarcube_v7_fieldpoint_csi import SolarCubeWindows, MultiScaleLocal, masked_mse


def crps_ensemble(samples: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Empirical CRPS along ensemble dimension 1; inputs [B,M,...]."""
    first = (samples - truth.unsqueeze(1)).abs().mean(1)
    pair = (samples.unsqueeze(2) - samples.unsqueeze(1)).abs().mean((1, 2))
    return first - 0.5 * pair


class ConditionalResidualDiffusion(nn.Module):
    """Small joint DDPM for [24x256 field CSI, 24x1 point CSI] residuals."""
    def __init__(self, horizon=24, field_nodes=256, width=256, steps=8, cond_dim=130):
        super().__init__()
        self.horizon = horizon
        self.field_nodes = field_nodes
        self.channels = field_nodes + 1
        self.dim = horizon * self.channels
        self.steps = steps
        self.time = nn.Embedding(steps, width)
        self.net = nn.Sequential(
            nn.Linear(self.dim + cond_dim + width, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, self.dim),
        )
        beta = torch.linspace(1e-4, 0.02, steps)
        alpha = 1.0 - beta
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("abar", alpha.cumprod(0))

    def predict_eps(self, x_t, cond, step):
        return self.net(torch.cat([x_t, cond, self.time(step)], dim=-1))

    def loss(self, residual, mask, cond):
        """Standard masked noise-prediction objective for the true CSI residual."""
        batch = residual.shape[0]
        x0 = residual.reshape(batch, -1)
        valid = mask.reshape(batch, -1).bool()
        x0 = torch.where(valid, x0, torch.zeros_like(x0))
        step = torch.randint(self.steps, (batch,), device=x0.device)
        eps = torch.randn_like(x0)
        a = self.abar[step, None]
        x_t = a.sqrt() * x0 + (1.0 - a).sqrt() * eps
        eps_hat = self.predict_eps(x_t, cond, step)
        err = (eps_hat - eps).square()
        return (err * valid.float()).sum() / valid.float().sum().clamp_min(1.0)

    def sample(self, cond, members=1, generator=None):
        """Differentiable ancestral sampling; samples remain in the forecast graph."""
        batch = cond.shape[0]
        c = cond[:, None].expand(-1, members, -1).reshape(batch * members, -1)
        x = torch.randn(batch * members, self.dim, device=cond.device,
                        dtype=cond.dtype, generator=generator)
        for k in reversed(range(self.steps)):
            step = torch.full((batch * members,), k, device=x.device, dtype=torch.long)
            eps_hat = self.predict_eps(x, c, step)
            x = (x - self.beta[k] / (1.0 - self.abar[k]).sqrt() * eps_hat) / self.alpha[k].sqrt()
            if k > 0:
                z = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
                x = x + self.beta[k].sqrt() * z
        return x.view(batch, members, self.horizon, self.channels)


class V8(nn.Module):
    """V3 encoder plus sampled residual -> bidirectional messages -> CSI decoders."""
    def __init__(self, width=64, heads=4, diffusion_steps=8):
        super().__init__()
        self.width = width
        self.horizon = 24
        # Exact V3 deterministic spatial encoder registry selection.
        self.encoder = RasterEncoder(channels=15, width=width, heads=heads,
                                     name="ENC-VISION-SWIN", preserve_grid=True)
        self.clock = nn.Sequential(nn.Linear(6, width), nn.GELU(), nn.Linear(width, width))
        tl = nn.TransformerEncoderLayer(width, heads, 2 * width, 0.1,
              batch_first=True, norm_first=True, activation="gelu")
        self.temporal = nn.TransformerEncoder(tl, 1)
        self.local = MultiScaleLocal(width)
        self.field_csi = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        self.point_fuse = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, width))
        self.point_csi = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

        # Cross-scale residual-message blocks. They receive the sampled DDPM
        # residual itself, plus the pre-message forecast and counterpart signal.
        self.point_to_field = nn.Linear(width, width)
        self.field_to_point = nn.Linear(width, width)
        self.field_msg = nn.Sequential(nn.Linear(2 * width + 4, width), nn.GELU(), nn.Linear(width, width))
        self.point_msg = nn.Sequential(nn.Linear(2 * width + 4, width), nn.GELU(), nn.Linear(width, width))
        self.field_refine = nn.Sequential(nn.Linear(width + 1, width), nn.GELU(), nn.Linear(width, 1))
        self.point_refine = nn.Sequential(nn.Linear(width + 1, width), nn.GELU(), nn.Linear(width, 1))
        # Start conservatively; the learned gates can grow if residual samples help.
        self.field_residual_gain = nn.Parameter(torch.tensor(-2.0))
        self.point_residual_gain = nn.Parameter(torch.tensor(-2.0))
        self.diffusion = ConditionalResidualDiffusion(cond_dim=2 * width + 2, steps=diffusion_steps)

    def encode(self, b):
        nwp = F.interpolate(b["nwp"].flatten(0, 1), size=(16, 16), mode="bilinear",
                            align_corners=False).view(*b["nwp"].shape[:2], 9, 16, 16)
        x = torch.cat([b["sat"], nwp], 2)
        z = self.encoder(x)
        z = z + self.clock(b["time_hist"])[:, :, None, None, :].permute(0, 1, 4, 2, 3)
        batch, hist, dim, height, width = z.shape
        tokens = z.permute(0, 3, 4, 1, 2).reshape(batch * height * width, hist, dim)
        tokens = self.temporal(tokens)
        latent = (tokens[:, -1] + tokens.mean(1)).view(batch, height, width, dim).permute(0, 3, 1, 2)
        # Keep V3's common 16x16 forecast grid and feature layout.
        return F.interpolate(latent, size=(16, 16), mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)

    def deterministic_forecast(self, b):
        grid = self.encode(b)                         # [B,256,D]
        batch, nodes, dim = grid.shape
        tf = self.clock(b["time_future"])             # [B,24,D]
        future = grid[:, None] + tf[:, :, None, :]    # [B,24,256,D]
        field0 = 2.5 * torch.sigmoid(self.field_csi(future)[..., 0])

        local, _ = self.local(b["point_hist"], b["point_hist_mask"])
        xy = (2 * b["position"] - 1).view(batch, 1, 1, 2)
        grid_image = grid.transpose(1, 2).reshape(batch, dim, 16, 16)
        site = F.grid_sample(grid_image, xy, mode="bilinear", padding_mode="border",
                             align_corners=True).reshape(batch, dim)
        point_feature = self.point_fuse(torch.cat([site, grid.mean(1), local], -1))[:, None] + tf
        point0 = 2.5 * torch.sigmoid(self.point_csi(point_feature)[..., 0])
        cond = torch.cat([grid.mean(1), local,
                          field0.mean((1, 2), keepdim=False)[:, None],
                          point0.mean(1, keepdim=True)], -1)
        return grid, future, point_feature, field0, point0, cond

    def message_update_and_decode(self, future, point_feature, field0, point0, residual):
        """Pass every generated residual member through both messages and decoders."""
        batch, members, horizon, channels = residual.shape
        field_res = residual[..., :256]
        point_res = residual[..., 256]
        field_latent = future[:, None].expand(-1, members, -1, -1, -1)
        point_latent = point_feature[:, None].expand(-1, members, -1, -1)
        f0 = field0[:, None].expand(-1, members, -1, -1)
        p0 = point0[:, None].expand(-1, members, -1)
        fr = field_res
        pr = point_res

        # Field receives the station forecast and station residual at every grid node.
        fctx = torch.stack([
            f0,
            p0[..., None].expand(-1, -1, -1, 256),
            fr,
            pr[..., None].expand(-1, -1, -1, 256),
        ], -1)
        point_context = self.point_to_field(point_latent).unsqueeze(-2).expand(-1, -1, -1, 256, -1)
        field_updated = field_latent + self.field_msg(torch.cat([field_latent, point_context, fctx], -1))

        # Point receives the spatially aggregated field forecast/residual and its
        # own sampled residual; this is the reverse (regional-to-local) message.
        pctx = torch.stack([
            p0,
            f0.mean(-1),
            fr.mean(-1),
            pr,
        ], -1)
        regional_context = self.field_to_point(field_latent.mean(-2))
        point_updated = point_latent + self.point_msg(torch.cat([point_latent, regional_context, pctx], -1))

        fdelta = self.field_refine(torch.cat([field_updated, fr[..., None]], -1))[..., 0]
        pdelta = self.point_refine(torch.cat([point_updated, pr[..., None]], -1))[..., 0]
        fgain = torch.sigmoid(self.field_residual_gain)
        pgain = torch.sigmoid(self.point_residual_gain)
        field = (f0 + fgain * fr + 0.25 * torch.tanh(fdelta)).clamp(0.0, 2.5)
        point = (p0 + pgain * pr + 0.25 * torch.tanh(pdelta)).clamp(0.0, 2.5)
        return field, point

    def forward(self, b, members=1):
        grid, future, point_feature, field0, point0, cond = self.deterministic_forecast(b)
        residual = self.diffusion.sample(cond, members=members)
        field, point = self.message_update_and_decode(future, point_feature, field0, point0, residual)
        return field, point, field0, point0, residual, cond


def aggregate_metrics(pred, truth, mask):
    """Return lead-wise ensemble GHI metrics; pred is [N,M,H,...]."""
    rows = []
    for h in range(truth.shape[1]):
        valid = mask[:, h].astype(bool)
        if not valid.any():
            continue
        p = pred[:, :, h]
        y = truth[:, h]
        mean = p.mean(1)
        err = mean - y
        crps = np.abs(p - y[:, None]).mean(1) - 0.5 * np.abs(p[:, :, None] - p[:, None, :]).mean((1, 2))
        lo, hi = np.quantile(p, [0.05, 0.95], axis=1)
        spread = p.std(1)
        rows.append({
            "lead_hour": h + 1,
            "n": int(valid.sum()),
            "MAE": float(np.abs(err[valid]).mean()),
            "RMSE": float(np.sqrt(np.square(err[valid]).mean())),
            "CRPS": float(crps[valid].mean()),
            "PI90_coverage": float(((y >= lo) & (y <= hi))[valid].mean()),
            "PI90_width": float((hi - lo)[valid].mean()),
            "spread_skill": float(spread[valid].mean() / (np.sqrt(np.square(err[valid]).mean()) + 1e-8)),
        })
    return rows


def mean_leads(rows):
    return {k: float(np.mean([r[k] for r in rows if isinstance(r.get(k), (int, float))]))
            for k in ("MAE", "RMSE", "CRPS", "PI90_coverage", "PI90_width", "spread_skill")}


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def evaluate(model, loader, device, members, seed):
    model.eval()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    field_pred, point_pred = [], []
    field_truth, point_truth, field_mask, point_mask = [], [], [], []
    field_csi_pred, point_csi_pred = [], []
    field_csi_truth, point_csi_truth, field_csi_mask, point_csi_mask = [], [], [], []
    with torch.no_grad():
        for batch0 in loader:
            batch = move_batch(batch0, device)
            field, point, *_ = model(batch, members=members)
            fghi = field * batch["field_cs"][:, None]
            pghi = point * batch["point_cs"][:, None, :]
            field_pred.append(fghi.cpu().numpy())
            point_pred.append(pghi.cpu().numpy())
            field_truth.append(batch0["field_ghi"].numpy())
            point_truth.append(batch0["point_ghi"].numpy())
            field_mask.append(batch0["field_mask"].numpy())
            point_mask.append(batch0["point_mask"].numpy())
            field_csi_pred.append(field.cpu().numpy())
            point_csi_pred.append(point.cpu().numpy())
            field_csi_truth.append(batch0["field_csi"].numpy())
            point_csi_truth.append(batch0["point_csi"].numpy())
            field_csi_mask.append(batch0["field_csi_mask"].numpy())
            point_csi_mask.append(batch0["point_csi_mask"].numpy())
    fp, pp = np.concatenate(field_pred), np.concatenate(point_pred)
    fy, py = np.concatenate(field_truth), np.concatenate(point_truth)
    fm, pm = np.concatenate(field_mask), np.concatenate(point_mask)
    fcp, pcp = np.concatenate(field_csi_pred), np.concatenate(point_csi_pred)
    fct, pct = np.concatenate(field_csi_truth), np.concatenate(point_csi_truth)
    fcm, pcm = np.concatenate(field_csi_mask), np.concatenate(point_csi_mask)
    result = {
        "field_GHI_by_lead": aggregate_metrics(fp, fy, fm),
        "point_GHI_by_lead": aggregate_metrics(pp, py, pm),
        "field_CSI_by_lead": aggregate_metrics(fcp, fct, fcm),
        "point_CSI_by_lead": aggregate_metrics(pcp, pct, pcm),
    }
    return result, (fp, pp, fy, py, fm, pm, fcp, pcp, fct, pct, fcm, pcm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--prepared", required=True)
    ap.add_argument("--sitelist", required=True); ap.add_argument("--nwp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--eval-members", type=int, default=8)
    ap.add_argument("--diffusion-steps", type=int, default=8); ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("V8 full SolarCube run expects CUDA; no training was started")
    datasets = {s: SolarCubeWindows(a.data, a.prepared, a.sitelist, a.nwp, s, stride=a.stride)
                for s in ("train", "val", "test")}
    loaders = {s: DataLoader(datasets[s], batch_size=a.batch, shuffle=(s == "train"),
                             num_workers=a.workers, pin_memory=True,
                             persistent_workers=(a.workers > 0)) for s in datasets}
    model = V8(diffusion_steps=a.diffusion_steps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    best, best_epoch, bad = float("inf"), 0, 0
    history = []
    start = time.time()
    log = (out / "train.log").open("w", buffering=1)
    config = {
        "version": "V8-V3Encoder-SampledResidualMessage-CSI",
        "dataset": "SolarCube 2018 all 19 regions",
        "protocol": "same full19/year split and 72h-to-24h stride3 window protocol as V3",
        "encoder": "exact V3 RasterEncoder ENC-VISION-SWIN + temporal Transformer + MultiScaleLocal",
        "outputs": "field CSI, selected-point CSI, reconstructed GHI field and point; joint probabilistic ensemble",
        "diffusion": {"target": "joint field+point CSI residual around V3 deterministic forecast",
                      "reverse_steps": a.diffusion_steps,
                      "message_path": "differentiable DDPM residual samples enter both field-to-point and point-to-field messages before refined CSI decoding",
                      "samples_train": 1, "samples_validation_test": a.eval_members},
        "message_flow": "one simultaneous bidirectional latent update, then field and point refinement heads",
        "early_stopping": {"monitor": "validation ensemble-mean joint normalized GHI + CSI loss",
                           "patience": a.patience, "restore_best": True},
        "batch": a.batch, "workers": a.workers, "seed": a.seed, "max_epochs": a.epochs,
        "artifacts": ["train.log", "README.md", "config.json", "best.pt", "training_history.json",
                      "metrics.json", "test_predictions.npz"],
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "README.md").write_text("# V8 SolarCube CSI residual-message model\n\n" +
        "V8 retains V3's Vision-Swin raster encoder, temporal Transformer, and local multiscale point-history encoder. "
        "A conditional DDPM samples joint field/point CSI residuals. The generated samples are explicit inputs to both "
        "directions of the message block; updated field and point latent states are then decoded again. This is the "
        "material architectural change from V7, whose stochastic diffusion samples were only added after message refinement.\n\n" +
        "The run writes the full log, config, best checkpoint, training history, test metrics, and test truths/predictions.\n\n" +
        json.dumps(config, indent=2) + "\n")
    log.write(json.dumps({"event": "started", "train_windows": len(datasets["train"]),
        "val_windows": len(datasets["val"]), "test_windows": len(datasets["test"]),
        "parameters": sum(p.numel() for p in model.parameters()), "device": str(device),
        "patience": a.patience}) + "\n")

    def validation_loss():
        # Fixed validation RNG makes patience/best-epoch comparisons less noisy.
        metrics, arrays = evaluate(model, loaders["val"], device, a.eval_members, a.seed + 9000)
        fp, pp, fy, py, fm, pm, fcp, pcp, fct, pct, fcm, pcm = arrays
        fmean, pmean = fp.mean(1), pp.mean(1)
        fcmean, pcmean = fcp.mean(1), pcp.mean(1)
        def masked_mse_np(pred, truth, mask):
            valid = mask.astype(bool)
            return float(np.square(pred[valid] - truth[valid]).mean()) if valid.any() else 0.0
        joint = (masked_mse_np((fmean - fy) / 1000.0, np.zeros_like(fy), fm) +
                 masked_mse_np((pmean - py) / 1000.0, np.zeros_like(py), pm) +
                 0.5 * (masked_mse_np(fcmean, fct, fcm) + masked_mse_np(pcmean, pct, pcm)))
        return joint, metrics

    try:
        for epoch in range(1, a.epochs + 1):
            model.train(); train_rows = []
            for batch0 in loaders["train"]:
                b = move_batch(batch0, device)
                optimizer.zero_grad(set_to_none=True)
                field, point, field0, point0, residual, cond = model(b, members=1)
                field = field[:, 0]; point = point[:, 0]
                fghi = field * b["field_cs"]
                pghi = point * b["point_cs"]
                fl = masked_mse((fghi - b["field_ghi"]) / 1000.0, torch.zeros_like(fghi), b["field_mask"])
                pl = masked_mse((pghi - b["point_ghi"]) / 1000.0, torch.zeros_like(pghi), b["point_mask"])
                cfl = masked_mse(field, b["field_csi"], b["field_csi_mask"])
                cpl = masked_mse(point, b["point_csi"], b["point_csi_mask"])
                target_residual = torch.cat([b["field_csi"] - field0,
                                             (b["point_csi"] - point0)[..., None]], -1)
                residual_mask = torch.cat([b["field_csi_mask"], b["point_csi_mask"][..., None]], -1)
                dloss = model.diffusion.loss(target_residual.detach(), residual_mask, cond)
                loss = fl + pl + 0.5 * (cfl + cpl) + 0.03 * dloss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss at epoch {epoch}")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                train_rows.append([float(loss.detach()), float(fl.detach()), float(pl.detach()),
                                   float(cfl.detach()), float(cpl.detach()), float(dloss.detach())])

            model.eval()
            val, val_metrics = validation_loss()
            row = {"epoch": epoch, "train_loss_components": np.mean(train_rows, axis=0).tolist(),
                   "val_joint_loss": val, "val_field_GHI": mean_leads(val_metrics["field_GHI_by_lead"]),
                   "val_point_GHI": mean_leads(val_metrics["point_GHI_by_lead"]),
                   "elapsed_sec": time.time() - start,
                   "peak_gpu_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20}
            history.append(row); log.write(json.dumps(row) + "\n")
            if val < best - 1e-7:
                best, best_epoch, bad = val, epoch, 0
                torch.save({"state_dict": model.state_dict(), "epoch": epoch, "config": config}, out / "best.pt")
            else:
                bad += 1
            if epoch >= 5 and bad >= a.patience:
                log.write(json.dumps({"event": "early_stopping", "epoch": epoch,
                                      "best_epoch": best_epoch, "patience": a.patience}) + "\n")
                break

        checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        # Test ensemble predictions are obtained only from the restored best checkpoint.
        test_metrics, arrays = evaluate(model, loaders["test"], device, a.eval_members, a.seed + 19000)
        fp, pp, fy, py, fm, pm, f_csi_pred, p_csi_pred, fct, pct, fcm, pcm = arrays
        result = {
            "version": config["version"], "encoder": "ENC-VISION-SWIN (same as V3)",
            "best_epoch": best_epoch, "epochs_ran": len(history), "train_seconds": time.time() - start,
            "test_windows": len(fp), "parameters": sum(p.numel() for p in model.parameters()),
            "peak_gpu_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
            "field_GHI_by_lead": aggregate_metrics(fp, fy, fm),
            "point_GHI_by_lead": aggregate_metrics(pp, py, pm),
            "field_CSI_by_lead": test_metrics["field_CSI_by_lead"],
            "point_CSI_by_lead": test_metrics["point_CSI_by_lead"],
        }
        result["field_GHI_mean24"] = mean_leads(result["field_GHI_by_lead"])
        result["point_GHI_mean24"] = mean_leads(result["point_GHI_by_lead"])
        result["field_CSI_mean24"] = mean_leads(result["field_CSI_by_lead"])
        result["point_CSI_mean24"] = mean_leads(result["point_CSI_by_lead"])
        np.savez_compressed(out / "test_predictions.npz",
            field_ghi_prediction_members=fp, field_ghi_truth=fy, field_mask=fm,
            point_ghi_prediction_members=pp, point_ghi_truth=py, point_mask=pm,
            field_csi_prediction_members=f_csi_pred, field_csi_truth=fct, field_csi_mask=fcm,
            point_csi_prediction_members=p_csi_pred, point_csi_truth=pct, point_csi_mask=pcm)
        (out / "metrics.json").write_text(json.dumps(result, indent=2))
        (out / "training_history.json").write_text(json.dumps(history, indent=2))
        log.write(json.dumps({"event": "finished", "result": result}) + "\n")
        (out / "DONE.json").write_text(json.dumps({"best_epoch": best_epoch,
            "epochs_ran": len(history), "train_seconds": result["train_seconds"]}, indent=2))
    except Exception as exc:
        log.write(json.dumps({"event": "failed", "type": type(exc).__name__, "message": str(exc),
                              "elapsed_sec": time.time() - start}) + "\n")
        raise
    finally:
        log.close()


if __name__ == "__main__":
    main()
