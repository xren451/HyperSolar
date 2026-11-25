#!/usr/bin/env python3
"""V6: validation-fitted adaptive fusion of frozen SMF-V2 and clear sky.

The forecast at each site/lead is a convex blend of the frozen V2 ensemble and
future deterministic clear-sky GHI. Blend weights are fitted on held-out V2
validation forecasts, conditioned on recent observed clear-sky index. Test
labels are never used to select weights.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_irradiance_fieldpoint_station import Windows, metrics_by_lead
from models_field_point import Version2FieldPoint


def names(a):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def get_channels(z):
    fs = names(z["feature_names"])
    ghi_name = next((x for x in ("GHI", "CERES_all_sky_Wm2") if x in fs), None)
    cs_name = next((x for x in ("ClearskyGHI", "CERES_clear_sky_Wm2") if x in fs), None)
    if ghi_name is None or cs_name is None:
        raise ValueError(f"Cannot find GHI and clear-sky channels: {fs}")
    return fs.index(ghi_name), fs.index(cs_name)


def csi_bins(z, origins, sites, ghi_idx, cs_idx):
    x = np.asarray(z["features"], np.float32)
    csi = np.zeros((len(origins), len(sites)), np.float32)
    valid = np.zeros_like(csi, dtype=bool)
    for i, t in enumerate(origins):
        lo = max(0, int(t) - 5)
        g = x[lo:int(t)+1, sites, ghi_idx]
        c = x[lo:int(t)+1, sites, cs_idx]
        m = np.isfinite(g) & np.isfinite(c) & (c > 20)
        count = m.sum(axis=0)
        csi[i] = np.divide(np.where(m, g, 0).sum(axis=0), np.maximum(count, 1)) / np.maximum(
            np.divide(np.where(m, c, 0).sum(axis=0), np.maximum(count, 1)), 1)
        valid[i] = count > 0
    # 0=not enough daytime history; 1..4=cloud attenuation regimes.
    b = np.digitize(np.clip(csi, 0, 2), [0.3, 0.7, 1.1]).astype(np.int8) + 1
    b[~valid] = 0
    return b, csi


def clear_values(z, origins, sites, cs_idx, horizon=24):
    ids = np.asarray(origins)[:, None] + np.arange(1, horizon + 1)[None, :]
    return np.asarray(z["features"][ids][:, :, sites, cs_idx], np.float32)


def infer_val(z, ds, model, train_ds, device, members, batch_size=8):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    fm, pm, fy, fmask, py, pmask, origins = [], [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        start = time.time()
        for bi, b0 in enumerate(loader, 1):
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b0.items()}
            f, p, _, _ = model.forecast(b, members)
            f = f.float().cpu().numpy() * train_ds.std[None, None, None, :] + train_ds.mean[None, None, None, :]
            p = p.float().cpu().numpy() * train_ds.std[ds.selected_site] + train_ds.mean[ds.selected_site]
            fm.append(f.mean(axis=1)); pm.append(p.mean(axis=1))
            fy.append(b0["target_raw"].numpy()); fmask.append(b0["target_mask"].numpy().astype(bool))
            py.append(b0["target_raw"].numpy()[..., ds.selected_site]); pmask.append(b0["target_mask"].numpy()[..., ds.selected_site].astype(bool))
            origins.append(b0["origin"].numpy())
            if bi % 100 == 0:
                print(json.dumps({"event":"validation_inference_progress","batch":bi,
                                  "batches":len(loader),"elapsed_sec":time.time()-start}), flush=True)
    return {"field_mean": np.concatenate(fm), "point_mean": np.concatenate(pm),
            "field_truth": np.concatenate(fy), "field_mask": np.concatenate(fmask),
            "point_truth": np.concatenate(py), "point_mask": np.concatenate(pmask),
            "origins": np.concatenate(origins)}


def fit_weights(v2, clear, truth, mask, regimes, shrink_examples=100):
    """Per-lead/site/regime ridge-shrunk least-squares convex blend weights."""
    n, h, sites = truth.shape
    table = np.zeros((h, sites, 5), np.float32)
    counts = np.zeros((h, sites, 5), np.int32)
    for lead in range(h):
        for s in range(sites):
            ok = mask[:, lead, s] & np.isfinite(v2[:, lead, s]) & np.isfinite(clear[:, lead, s])
            dx = clear[:, lead, s] - v2[:, lead, s]
            dy = truth[:, lead, s] - v2[:, lead, s]
            if not ok.any():
                continue
            den_all = float(np.square(dx[ok]).sum())
            alpha_all = float(np.clip(np.sum(dx[ok] * dy[ok]) / max(den_all, 1e-9), 0, 1))
            lam = shrink_examples * float(np.mean(np.square(dx[ok]))) + 1e-9
            for regime in range(5):
                sel = ok & (regimes[:, s] == regime)
                cnt = int(sel.sum()); counts[lead, s, regime] = cnt
                den = float(np.square(dx[sel]).sum())
                num = float(np.sum(dx[sel] * dy[sel]))
                table[lead, s, regime] = np.clip((num + lam * alpha_all) / (den + lam), 0, 1)
    return table, counts


def select_weights(table, regimes):
    n, sites = regimes.shape; h = table.shape[0]
    out = np.empty((n, h, sites), np.float32)
    for s in range(sites):
        for lead in range(h):
            out[:, lead, s] = table[lead, s, regimes[:, s]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['NSRDB', 'SolarAlabama'])
    ap.add_argument('--data', required=True); ap.add_argument('--v2-run', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:2'); ap.add_argument('--members', type=int, default=16)
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--shrink-examples', type=int, default=100)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    print(json.dumps({"event":"started","dataset":a.dataset,"device":a.device,
                      "v2_run":a.v2_run,"out":str(out)}), flush=True)
    zf = np.load(a.data, allow_pickle=True); z = {k: zf[k] for k in zf.files}; zf.close()
    cfg = json.load(open(Path(a.v2_run) / 'config.json')); selected = int(cfg.get('selected_site', 0))
    n, sites, nf = z['features'].shape; train_ds = Windows(z, 'train', selected_site=selected)
    val_ds = Windows(z, 'val', selected_site=selected); device = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    model = Version2FieldPoint(nf, sites, z['coords'], selected_site=selected,
        width=int(cfg.get('width', 64)), heads=4, ablation='full').to(device)
    model.load_state_dict(torch.load(Path(a.v2_run) / 'best.pt', map_location=device, weights_only=True))
    val = infer_val(z, val_ds, model, train_ds, device, a.members)
    print(json.dumps({"event":"validation_inference_complete","dataset":a.dataset,
                      "windows":len(val['origins'])}), flush=True)
    ghi_idx, cs_idx = get_channels(z); all_sites = np.arange(sites, dtype=np.int64)
    val_reg, val_csi = csi_bins(z, val['origins'], all_sites, ghi_idx, cs_idx)
    val_cs = clear_values(z, val['origins'], all_sites, cs_idx)
    field_table, field_counts = fit_weights(val['field_mean'], val_cs, val['field_truth'], val['field_mask'], val_reg, a.shrink_examples)
    # Point head is calibrated independently from the field head at the selected coordinate.
    ps = np.asarray([selected], dtype=np.int64); preg, pcsi = csi_bins(z, val['origins'], ps, ghi_idx, cs_idx)
    pcs = clear_values(z, val['origins'], ps, cs_idx)
    point_table, point_counts = fit_weights(val['point_mean'][..., None], pcs, val['point_truth'][..., None], val['point_mask'][..., None], preg, a.shrink_examples)
    print(json.dumps({"event":"gate_fitting_complete","dataset":a.dataset,
                      "field_bins":int((field_counts>0).sum()),"point_bins":int((point_counts>0).sum())}), flush=True)
    v2 = np.load(Path(a.v2_run) / 'test_predictions.npz', allow_pickle=True)
    origins = v2['origins']; fens = v2['field_ensemble']; pens = v2['point_ensemble']
    fy, fm = v2['field_truth'], v2['field_mask'].astype(bool); py, pm = v2['point_truth'], v2['point_mask'].astype(bool)
    fmean, pmean = fens.mean(axis=1), pens.mean(axis=1)
    test_reg, test_csi = csi_bins(z, origins, all_sites, ghi_idx, cs_idx); csf = clear_values(z, origins, all_sites, cs_idx)
    preg_test, _ = csi_bins(z, origins, ps, ghi_idx, cs_idx); csp = clear_values(z, origins, ps, cs_idx)[..., 0]
    af = select_weights(field_table, test_reg); ap = select_weights(point_table, preg_test)[..., 0]
    v6f = (1 - af[:, None]) * fens + af[:, None] * csf[:, None]
    v6p = (1 - ap[:, None]) * pens + ap[:, None] * csp[:, None, :]
    clear_f = np.broadcast_to(csf[:, None], fens.shape).copy(); clear_p = np.broadcast_to(csp[:, None, :], pens.shape).copy()
    day_f = fm & (csf > 20); day_p = pm & (csp > 20)
    def summ(p, y, m):
        rows = metrics_by_lead(p, y, m)
        return {'metrics_by_lead': rows, 'mean24': {k: float(np.mean([r[k] for r in rows])) for k in ['MAE','RMSE','CRPS','PI90_coverage','PI90_width']}}
    result = {
      'dataset':a.dataset,'version':'V6','selected_site':str(z['site_names'][selected]),'test_origins':int(len(origins)),
      'fusion':'frozen SMF-V2 ensemble + future clear-sky GHI; validation-fitted, site/lead/recent-CSI conditioned convex gate',
      'field':{'V6_all_hours':summ(v6f,fy,fm),'V2_all_hours':summ(fens,fy,fm),'clear_sky_all_hours':summ(clear_f,fy,fm),
               'V6_daylight':summ(v6f,fy,day_f),'V2_daylight':summ(fens,fy,day_f),'clear_sky_daylight':summ(clear_f,fy,day_f)},
      'point':{'V6_all_hours':summ(v6p[...,None],py[...,None],pm[...,None]),'V2_all_hours':summ(pens[...,None],py[...,None],pm[...,None]),
               'clear_sky_all_hours':summ(clear_p[...,None],py[...,None],pm[...,None]),'V6_daylight':summ(v6p[...,None],py[...,None],day_p[...,None]),
               'V2_daylight':summ(pens[...,None],py[...,None],day_p[...,None]),'clear_sky_daylight':summ(clear_p[...,None],py[...,None],day_p[...,None])},
      'daylight_definition':'future clear-sky GHI > 20 W m-2; same full-year test origins and masks'}
    np.savez_compressed(out/'test_predictions.npz',field_ensemble=v6f.astype(np.float32),field_truth=fy,field_mask=fm,
        point_ensemble=v6p.astype(np.float32),point_truth=py,point_mask=pm,field_clear_sky_weight=af,point_clear_sky_weight=ap,
        field_clear_sky=csf,point_clear_sky=csp,field_recent_csi=test_csi,point_recent_csi=csi_bins(z,origins,ps,ghi_idx,cs_idx)[1][:,0],
        origins=origins,timestamps=v2['timestamps'],coords=v2['coords'],site_names=v2['site_names'],selected_site=v2['selected_site'])
    np.savez_compressed(out/'gate_parameters.npz',field_weights=field_table,field_counts=field_counts,
        point_weights=point_table,point_counts=point_counts,regime_edges=np.asarray([.3,.7,1.1]),shrink_examples=a.shrink_examples)
    config={'dataset':a.dataset,'version':'V6','base_model':'frozen SMF-V2 checkpoint','data':str(Path(a.data).resolve()),
        'v2_run':str(Path(a.v2_run).resolve()),'split':'same chronological full-year 70/15/15 split as V2',
        'gate_fit':'validation forecasts only; test labels never used to fit weights','history_hours':72,'horizon_hours':24,
        'fusion':'y6=(1-alpha)*y_v2+alpha*GHI_clear','alpha_scope':'separate field/point; per site, lead and previous-6h CSI regime; clipped to [0,1]',
        'CSI_bins':['missing/day-night history','<0.3','0.3-0.7','0.7-1.1','>=1.1'],'shrink_examples':a.shrink_examples,
        'daylight_definition':'future clear-sky GHI >20 W m-2','device':str(device),'selected_site':str(z['site_names'][selected])}
    (out/'config.json').write_text(json.dumps(config,indent=2));(out/'test_metrics.json').write_text(json.dumps(result,indent=2))
    (out/'README.md').write_text(f'''# {a.dataset} V6 adaptive clear-sky / SMF-V2 fusion\n\nV6 freezes the existing V2 checkpoint and learns a convex blend on held-out validation forecasts. Separate field and selected-point blend tables use site, lead time and the observed mean CSI over the preceding six hours. Per-regime least-squares weights are shrunk toward the site/lead weight and clipped to [0,1]. Test labels are used only for final evaluation.\n\nForecast: `y6 = (1-alpha) * y_v2 + alpha * GHI_clear`. This is a light validation-fitted fusion model, not end-to-end retraining of V2. The future clear-sky branch must be available from an operational astronomical clear-sky model to claim deployment-ready results. SolarAlabama labels are CERES all-sky product values, not pyranometer truth.\n\nArtifacts: `config.json`, `gate_parameters.npz`, `test_metrics.json`, `test_predictions.npz`.\n''')
    with (out/'train.log').open('w',encoding='utf-8') as log:
      log.write(json.dumps({'event':'validation_gate_fitted','dataset':a.dataset,'field_fit_origins':len(val['origins']),
        'field_examples':int(val['field_mask'].sum()),'point_examples':int(val['point_mask'].sum()),
        'field_gate_bins_with_data':int((field_counts>0).sum()),'point_gate_bins_with_data':int((point_counts>0).sum()),
        'validation_best_v2_checkpoint':str(Path(a.v2_run)/'best.pt')})+'\n')
      log.write(json.dumps({'event':'test_complete','test_origins':len(origins),'out':str(out)})+'\n')
    print(json.dumps({'dataset':a.dataset,'field_V6':result['field']['V6_all_hours']['mean24'],
      'field_V2':result['field']['V2_all_hours']['mean24'],'field_clear_sky':result['field']['clear_sky_all_hours']['mean24'],
      'point_V6':result['point']['V6_all_hours']['mean24'],'point_V2':result['point']['V2_all_hours']['mean24'],
      'point_clear_sky':result['point']['clear_sky_all_hours']['mean24'],
      'daylight_field_MAE':{k:result['field'][k]['mean24']['MAE'] for k in ['V6_daylight','V2_daylight','clear_sky_daylight']},
      'daylight_point_MAE':{k:result['point'][k]['mean24']['MAE'] for k in ['V6_daylight','V2_daylight','clear_sky_daylight']},'output':str(out)},indent=2))


if __name__ == '__main__':
    main()
