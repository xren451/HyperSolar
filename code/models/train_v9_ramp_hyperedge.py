#!/usr/bin/env python3
"""Train/evaluate V9-RampHyperedge-CSI on full SolarCube, with early stopping."""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_solarcube_v7_fieldpoint_csi import SolarCubeWindows
from v9_ramp_hyperedge import V9RampHyperedgeCSI


def scores(pred, truth, mask):
    m = np.asarray(mask, bool) & np.isfinite(pred) & np.isfinite(truth)
    if not m.any(): return {"n": 0, "MAE": None, "RMSE": None}
    e = np.asarray(pred)[m] - np.asarray(truth)[m]
    return {"n": int(m.sum()), "MAE": float(np.abs(e).mean()),
            "RMSE": float(np.sqrt(np.square(e).mean()))}


def crps_ensemble(samples, truth, mask):
    # samples [B,M,...]; empirical CRPS, evaluated only where labels are valid.
    first = np.abs(samples - truth[:, None]).mean(axis=1)
    pair = np.abs(samples[:, :, None] - samples[:, None, :]).mean(axis=(1, 2))
    crps = first - 0.5 * pair
    m = np.asarray(mask, bool) & np.isfinite(crps)
    return float(crps[m].mean()) if m.any() else None


def main():
    p = argparse.ArgumentParser()
    for name in ("data", "prepared", "sitelist", "nwp", "out"): p.add_argument("--"+name, required=True)
    p.add_argument("--epochs", type=int, default=60); p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch", type=int, default=4); p.add_argument("--stride", type=int, default=3)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--diffusion-weight", type=float, default=0.1)
    p.add_argument("--diffusion-steps", type=int, default=8); p.add_argument("--eval-members", type=int, default=8)
    a = p.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for full SolarCube training")
    ds = {s: SolarCubeWindows(a.data,a.prepared,a.sitelist,a.nwp,s,stride=a.stride) for s in ("train","val","test")}
    dl = {s: DataLoader(ds[s],batch_size=a.batch,shuffle=s=="train",num_workers=0,pin_memory=True) for s in ds}
    dev=torch.device("cuda"); model=V9RampHyperedgeCSI(diffusion_steps=a.diffusion_steps).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5); best=float("inf"); bad=0; hist=[]; start=time.time()
    cfg={"version":"V9-RampConditioned-Hyperedge-CSI-ResidualDDPM","seed":a.seed,
         "data":"SolarCube 2018 all 19 regions","split":"chronological 70/15/15",
         "history_hours":72,"horizon_hours":24,"stride_hours":a.stride,"batch":a.batch,
         "encoder":"V5 ResNet-Swin-ViT; source-isolated satellite/NWP history passes share the raster encoder",
         "cloud_state":"latent effective cloud/radiative state; CSI-supervised because COD/cloud-property targets are not present in this data loader",
         "hypergraph":"V9 history-future hypergraph; fixed 4x4 local, global and same-cell cross-source edges; history-to-future messages; future labels excluded",
         "ramp_hyperedge":"multi-scale station TS encoder emits six observed-history statistics [normalized mean, mean slope, max positive step, max negative step, adjacent-valid fraction, history-valid fraction]; these are projected into an edge attribute and added to the station-containing 4x4 local hyperedge before edge-to-node message passing; ramp gate is learned; no future ramp/label is used",
         "stochastic":"conditional DDPM over joint field/site CSI residual; sampling adds residual to deterministic CSI and clips to [0,2.5]",
         "diffusion_steps":a.diffusion_steps,"diffusion_weight":a.diffusion_weight,"eval_members":a.eval_members,
         "early_stopping":{"metric":"deterministic validation joint loss","patience":a.patience,"restore_best":True}}
    (out/"config.json").write_text(json.dumps(cfg,indent=2)+"\n")
    model_path = Path(__file__).with_name("v9_ramp_hyperedge.py").resolve()
    train_path = Path(__file__).resolve()
    (out/"README.md").write_text(
        "# V9 Ramp-conditioned hyperedge CSI experiment\n\n"
        f"Training code: `{train_path}`\nModel code: `{model_path}`\nDataset loader: configured dataset adapter\n\n"
        "SolarCube 2018, all 19 regions; chronological 70/15/15 split; 72 h history, 24 h forecast, 3 h stride. "
        "This is V9's V5 ResNet-Swin-ViT/source-isolated encoder, history-to-future hypergraph, joint field/site CSI forecasts and conditional residual DDPM, with one isolated change: observed station ramp statistics are encoded as an attribute on the station-containing local 4x4 hyperedge. Future ramp/labels are excluded. COD is absent from the audited loader; the learned state is a latent cloud-radiative state, not COD-supervised cloud state.\n\n"
        f"Seed {a.seed}; patience {a.patience}; max epochs {a.epochs}; diffusion steps {a.diffusion_steps}; ensemble members {a.eval_members}. "
        "Artifacts: config, train log, early-stopping history, best checkpoint, test truth/prediction archive and metrics.\n"
    )
    log=(out/"train.log").open("w",buffering=1); log.write(json.dumps({"event":"started","gpu":torch.cuda.get_device_name(0),"parameters":sum(x.numel() for x in model.parameters())})+"\n")
    def move(batch): return {k:v.to(dev,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
    for ep in range(1,a.epochs+1):
        model.train(); tr=[]
        for raw in dl["train"]:
            b=move(raw); opt.zero_grad(set_to_none=True); o=model(b)
            ls=model.joint_loss(o,b,a.diffusion_weight); loss=ls["loss"]
            if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite loss at epoch {ep}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            tr.append([float(ls[k].detach()) for k in ("loss","deterministic","diffusion")])
        model.eval(); va=[]
        with torch.no_grad():
            for raw in dl["val"]:
                b=move(raw); o=model(b); ls=model.joint_loss(o,b,a.diffusion_weight)
                # Exclude sampled diffusion noise from early stopping for stability.
                va.append(float(ls["deterministic"]))
        val=float(np.mean(va)); row={"epoch":ep,"train_loss":np.mean(tr,axis=0).tolist(),"val_deterministic":val,
              "elapsed_sec":time.time()-start,"peak_gpu_allocated_MiB":torch.cuda.max_memory_allocated()/2**20}
        hist.append(row); log.write(json.dumps(row)+"\n")
        if val < best-1e-7:
            best=val; bad=0; torch.save(model.state_dict(),out/"best.pt"); best_ep=ep
        else: bad+=1
        if ep>=5 and bad>=a.patience: break
    model.load_state_dict(torch.load(out/"best.pt",map_location=dev,weights_only=True)); model.eval()
    fields=[]; points=[]; ftruth=[]; ptruth=[]; fm=[]; pm=[]; ensf=[]; ensp=[]
    with torch.no_grad():
        for raw in dl["test"]:
            b=move(raw); o=model(b); residual=model.forecast_samples(o,a.eval_members)
            fcs=residual[...,:256]; pcs=residual[...,256]
            fields.append((fcs*b["field_cs"][:,None]).cpu().numpy()); points.append((pcs*b["point_cs"][:,None]).cpu().numpy())
            ftruth.append(raw["field_ghi"].numpy()); ptruth.append(raw["point_ghi"].numpy())
            fm.append(raw["field_mask"].numpy()); pm.append(raw["point_mask"].numpy())
            ensf.append((o["field_csi"]*b["field_cs"]).cpu().numpy()); ensp.append((o["point_csi"]*b["point_cs"]).cpu().numpy())
    fields=np.concatenate(fields); points=np.concatenate(points); ftruth=np.concatenate(ftruth); ptruth=np.concatenate(ptruth)
    fm=np.concatenate(fm); pm=np.concatenate(pm); detf=np.concatenate(ensf); detp=np.concatenate(ensp)
    metrics={"best_epoch":best_ep,"epochs_ran":len(hist),"train_seconds":time.time()-start,
      "field_GHI_deterministic":scores(detf,ftruth,fm),"point_GHI_deterministic":scores(detp,ptruth,pm),
      "field_GHI_ensemble_mean":scores(fields.mean(1),ftruth,fm),"point_GHI_ensemble_mean":scores(points.mean(1),ptruth,pm),
      "field_GHI_CRPS":crps_ensemble(fields,ftruth,fm),"point_GHI_CRPS":crps_ensemble(points,ptruth,pm),
      "peak_gpu_allocated_MiB":torch.cuda.max_memory_allocated()/2**20}
    np.savez_compressed(out/"test_predictions.npz",field_ghi_samples=fields,point_ghi_samples=points,
      field_ghi_truth=ftruth,point_ghi_truth=ptruth,field_mask=fm,point_mask=pm,
      field_ghi_deterministic=detf,point_ghi_deterministic=detp)
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n"); (out/"training_history.json").write_text(json.dumps(hist,indent=2)+"\n")
    log.write(json.dumps({"event":"finished","metrics":metrics})+"\n"); log.close()

if __name__=="__main__": main()
