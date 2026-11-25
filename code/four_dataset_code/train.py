"""Train the archived V9 Ramp-HG residual DDPM with explicit data adapters."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import ROOT, INPUT_KEYS, Windows, prepare

from model import V9TemporalRamp as V9RampHyperedgeCSI


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def move(raw):
    return {k: v.cuda(non_blocking=True) for k, v in raw.items()}


def forecast(model, batch):
    return model({k: batch[k] for k in INPUT_KEYS})


def cpu_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def evaluate(model, loader, members=0):
    det, samples, truth, mask, origins, regions = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for raw in loader:
            b = move(raw)
            o = forecast(model, b)
            det.append((o['point_csi']*b['point_cs']).cpu().numpy())
            if members:
                sampled = model.forecast_samples(o, members)[..., -1]
                samples.append((sampled*b['point_cs'][:, None]).cpu().numpy())
            truth.append(raw['point_ghi'].numpy())
            mask.append(raw['point_mask'].numpy())
            origins.append(raw['origin'].numpy())
            if 'region' in raw:
                regions.append(raw['region'].numpy())
    result = dict(point_ghi_deterministic=np.concatenate(det),
                  point_ghi_truth=np.concatenate(truth), point_mask=np.concatenate(mask),
                  origins=np.concatenate(origins))
    if members:
        result['point_ghi_samples'] = np.concatenate(samples)
    if regions:
        result['regions'] = np.concatenate(regions)
    return result


def preview(model, dataset, origin, out, epoch, ds):
    index = ds.rows.index((8, origin)) if dataset == 'SolarCube' else ds.origins.index(origin)
    raw = ds[index]
    b = {k: v[None].cuda() for k, v in raw.items()}
    model.eval()
    with torch.no_grad():
        o = forecast(model, b)
        point = o['point_csi']*b['point_cs']
        samples = model.forecast_samples(o, 8)[..., -1]*b['point_cs'][:, None]
    path = out/'preview.tmp.npz'
    np.savez_compressed(path, point_ghi_deterministic=point.cpu().numpy()[0],
                        point_ghi_samples=samples.cpu().numpy()[0],
                        point_ghi_truth=raw['point_ghi'].numpy(), point_mask=raw['point_mask'].numpy(),
                        origin=origin, checkpoint_epoch=epoch, final=False)
    path.replace(out/'preview.npz')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no silent CPU training fallback')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    prepare(a.dataset)
    figures = json.loads((ROOT/'case_contract.json').read_text())
    origin = figures[a.dataset]['origin']
    ds = {s: Windows(a.dataset, s, origin if s == 'test' else None) for s in ['train', 'val', 'test']}
    out = ROOT/'results'/a.dataset
    if a.smoke:
        out = ROOT/'smoke'/a.dataset
    out.mkdir(parents=True, exist_ok=True)
    if not a.smoke and (out/'DONE.json').exists():
        print('ALREADY COMPLETE', a.dataset, flush=True)
        return
    model = V9RampHyperedgeCSI().cuda()
    initialization = 'random seed42, trained from scratch'
    if a.dataset == 'SolarCube' and not a.smoke:
        archived = Path(os.environ['HYPERSOLAR_SOLARCUBE_WARMSTART'])
        state = torch.load(archived, map_location='cpu', weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected
        allowed = ('observed_history.', 'observed_position.', 'lead_position.', 'horizon_decoder.', 'ramp_residual.', 'ramp_class.')
        assert all(k.startswith(allowed) for k in missing)
        initialization = 'warm-start existing train/validation-only V9 backbone, new TemporalRamp modules initialized independently: '+str(archived)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    original = Path(__file__).parent/'original'
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in original.glob('*.py')}
    config = dict(model=V9RampHyperedgeCSI.VERSION, architecture='original grid16 V9 backbone + observed-history cross-attention future decoder',
                  dataset=a.dataset, data=ds['train'].info, seed=42,
                  width=64, heads=4, history_hours=72, horizon_hours=24, history_frames=72,
                  diffusion_steps=8, diffusion_weight=.1, members=8,
                  max_epochs=a.epochs, patience=a.patience, min_epochs=5, batch=a.batch,
                  adaptation_source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                            for p in Path(__file__).parent.glob('*.py')},
                  optimizer='AdamW lr1e-4 wd1e-5, gradient clip1',
                  split='chronological70/15/15; entire72h history and24h targets confined to split',
                  windows={s: len(v) for s, v in ds.items()},
                  requested_figure_origin=origin, original_source_sha256=hashes,
                  parameters=sum(v.numel() for v in model.parameters()),
                  training_from_scratch=a.dataset != 'SolarCube', initialization=initialization, future_truth_blending=False,
                  adaptation='data adapters plus explicitly new TemporalRamp head/loss; not archived V9 results or native120 field protocol',
                  ramp_losses={'point_delta': .5, 'point_curvature': .25, 'point_up_down_stable': .02,
                               'event_threshold_Wm2_per_hour': 100, 'event_extra_weight': 2},
                  temporal_decoder={'layers': 2, 'heads': 4, 'memory': '72 observed point-history tokens only',
                                    'future_queries': '24 known lead embeddings + original V9 latent queries'})
    atomic_json(out/'config.json', config)
    if a.smoke:
        model.eval()
        raw = ds['train'][0]
        b = {k: v[None].cuda() for k, v in raw.items()}
        o = forecast(model, b)
        mutated = {k: v.clone() for k, v in b.items()}
        for k in ['point_ghi', 'field_ghi', 'point_csi', 'field_csi']:
            mutated[k] += 10000
        oo = model(mutated)
        assert torch.equal(o['point_csi'], oo['point_csi'])
        assert torch.equal(o['field_csi'], oo['field_csi'])
        model.train()
        losses = model.joint_loss(forecast(model, b), b, .1)
        assert torch.isfinite(losses['loss'])
        losses['loss'].backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        ramp_grad = sum(float(p.grad.abs().sum()) for p in model.hg.ramp_edge.parameters() if p.grad is not None)
        diff_grad = sum(float(p.grad.abs().sum()) for p in model.diffusion.parameters() if p.grad is not None)
        decoder_grad = sum(float(p.grad.abs().sum()) for p in model.horizon_decoder.parameters() if p.grad is not None)
        assert ramp_grad > 0 and diff_grad > 0 and decoder_grad > 0
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = forecast(model, b)
            sampled = model.forecast_samples(pred, 8)
        assert sampled.shape == (1, 8, 24, 257) and torch.isfinite(sampled).all()
        status = dict(dataset=a.dataset, observation_only_forward=True, future_target_perturbation_invariant=True,
                      finite_loss=float(losses['loss'].detach()), gradients_finite=True,
                      ramp_edge_gradient_l1=ramp_grad, diffusion_gradient_l1=diff_grad,
                      temporal_decoder_gradient_l1=decoder_grad,
                      ensemble_shape=list(sampled.shape), ensemble_std=float(sampled.std(1).mean()),
                      train_end=ds['train'].info['cuts'][1], statistics_fit_end=ds['train'].info['stat_fit_end_exclusive'],
                      split_boundaries_asserted=True, max_gpu_MiB=torch.cuda.max_memory_allocated()/2**20)
        atomic_json(out/'PASS.json', status)
        print(json.dumps(status), flush=True)
        return
    if not (ROOT/'smoke'/a.dataset/'PASS.json').exists():
        raise RuntimeError('Smoke/leakage/gradient audit must pass before full training')
    loaders = {s: DataLoader(v, batch_size=a.batch, shuffle=s=='train', num_workers=4 if a.dataset=='SolarCube' else 2,
                             persistent_workers=True, pin_memory=True) for s, v in ds.items()}
    start = time.time()
    hist = []
    best, bad, best_epoch = float('inf'), 0, 0
    log = (out/'train.jsonl').open('w', buffering=1)
    log.write(json.dumps(dict(event='started', gpu=torch.cuda.get_device_name(0), windows=config['windows']))+'\n')
    for epoch in range(1, a.epochs+1):
        model.train()
        tr = []
        for step, raw in enumerate(loaders['train'], 1):
            b = move(raw)
            opt.zero_grad(set_to_none=True)
            o = forecast(model, b)
            loss = model.joint_loss(o, b, .1)
            if not torch.isfinite(loss['loss']):
                raise FloatingPointError((a.dataset, epoch, step, 'nonfinite training loss'))
            loss['loss'].backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(grad):
                raise FloatingPointError((a.dataset, epoch, step, 'nonfinite gradient'))
            opt.step()
            tr.append([float(loss[k].detach()) for k in ['loss', 'deterministic', 'diffusion']])
            if step % 50 == 0:
                status = dict(state='training', epoch=epoch, batch=step, batches=len(loaders['train']),
                              elapsed_seconds=time.time()-start, best_epoch=best_epoch)
                atomic_json(out/'status.json', status)
                print(a.dataset, status, flush=True)
        model.eval()
        vals = []
        with torch.no_grad():
            for raw in loaders['val']:
                b = move(raw)
                v = model.joint_loss(forecast(model, b), b, .1)['deterministic']
                vals.append(float(v))
        val = float(np.mean(vals))
        if not np.isfinite(val):
            raise FloatingPointError('Nonfinite validation loss')
        rec = dict(epoch=epoch, train_loss=np.mean(tr, 0).tolist(), val_deterministic=val,
                   elapsed_seconds=time.time()-start, max_gpu_MiB=torch.cuda.max_memory_allocated()/2**20)
        hist.append(rec)
        log.write(json.dumps(rec)+'\n')
        improved = val < best-1e-7
        if improved:
            best, bad, best_epoch = val, 0, epoch
            tmp = out/'best.tmp.pt'
            torch.save(cpu_state(model), tmp)
            tmp.replace(out/'best.pt')
            # Fixed, prespecified visualization only; never used for fitting or model selection.
            preview(model, a.dataset, origin, out, epoch, ds['test'])
        else:
            bad += 1
        atomic_json(out/'status.json', dict(state='training', **rec, best_epoch=best_epoch, bad_epochs=bad))
        print(a.dataset, rec, 'best', best_epoch, flush=True)
        if epoch >= 5 and bad >= a.patience:
            break
    model.load_state_dict(torch.load(out/'best.pt', map_location='cuda', weights_only=True))
    torch.manual_seed(4242)
    model.eval()
    atomic_json(out/'status.json', dict(state='testing', best_epoch=best_epoch, epochs=len(hist)))
    result = evaluate(model, loaders['test'], members=8)
    timestamps = ds['test'].times if a.dataset == 'SolarCube' else np.load(ds['test'].path/'timestamps.npy')
    result['timestamps'] = timestamps[result['origins'][:, None]+np.arange(1, 25)]
    if a.dataset == 'SolarCube':
        result['point_lat_lon'] = np.array([[float(ds['test'].sites[int(r)]['lats']),
                                             float(ds['test'].sites[int(r)]['lons'])] for r in result['regions']])
        result['point_name'] = np.array([ds['test'].sites[int(r)]['name'] for r in result['regions']])
    else:
        result['point_lat_lon'] = np.array(ds['test'].info['point_lat_lon'])
        result['point_name'] = np.array(ds['test'].info['point_name'])
    result['checkpoint_epoch'] = np.array(best_epoch)
    result['final'] = np.array(True)
    np.savez_compressed(out/'test_predictions.npz', **result)
    perlead = []
    samples = result['point_ghi_samples']
    for h in range(24):
        valid = result['point_mask'][:, h]
        yy = result['point_ghi_truth'][:, h]
        e = result['point_ghi_deterministic'][:, h]-yy
        se = samples[:, :, h]
        ordered = np.sort(se, axis=1)
        weights = 2*np.arange(1, 9)-9
        crps = abs(se-yy[:, None]).mean(1)-(ordered*weights).sum(1)/64
        perlead.append(dict(lead_hour=h+1, n=int(valid.sum()),
                            deterministic_MAE=float(abs(e[valid]).mean()),
                            deterministic_RMSE=float(np.sqrt((e[valid]**2).mean())),
                            ensemble_mean_MAE=float(abs(se.mean(1)[valid]-yy[valid]).mean()),
                            ensemble_CRPS=float(crps[valid].mean())))
    metrics = dict(best_epoch=best_epoch, epochs=len(hist), elapsed_seconds=time.time()-start,
                   by_lead=perlead, mean24={k: float(np.mean([r[k] for r in perlead]))
                                          for k in perlead[0] if k not in ['n', 'lead_hour']},
                   protocol='new adaptation, separate from older baseline table metrics')
    atomic_json(out/'metrics.json', metrics)
    atomic_json(out/'training_history.json', hist)
    atomic_json(out/'DONE.json', dict(dataset=a.dataset, best_epoch=best_epoch,
                                     predictions=str(out/'test_predictions.npz'), elapsed_seconds=time.time()-start))
    atomic_json(out/'status.json', dict(state='complete', best_epoch=best_epoch, epochs=len(hist)))
    print('COMPLETE', a.dataset, json.dumps(metrics['mean24']), flush=True)


if __name__ == '__main__':
    import fcntl
    selected = sys.argv[sys.argv.index('--dataset')+1]
    lock_dir = ROOT/'locks'
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir/(selected+'.lock')).open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        main()
