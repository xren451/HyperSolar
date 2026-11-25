"""Event scores on matched complete test data, not on selected pictures."""
import json
from pathlib import Path
import zipfile
import numpy as np
from data import ROOT, R, B

BASELINES = ['Persistence', 'Clear-sky', 'SolarFormer', 'SolarSeer', 'GraphCast']


def column(path, key, node):
    with zipfile.ZipFile(path) as z:
        with z.open(key+'.npy') as f:
            v = np.lib.format.read_magic(f)
            shape, order, dtype = (np.lib.format.read_array_header_1_0(f) if v == (1, 0)
                                    else np.lib.format.read_array_header_2_0(f))
            assert not order
            n = int(np.prod(shape[:-1]))
            result = np.empty(n, dtype=dtype)
            for lo in range(0, n, 128):
                size = min(128, n-lo)
                result[lo:lo+size] = np.frombuffer(f.read(size*shape[-1]*dtype.itemsize), dtype=dtype).reshape(size, shape[-1])[:, node]
    return result.reshape(shape[:-1])


def scores(prediction, truth, valid, clear):
    pair = valid[:, 1:] & valid[:, :-1] & (clear[:, 1:] > 20) & (clear[:, :-1] > 20)
    dy = np.diff(truth, axis=1)
    dp = np.diff(prediction, axis=1)
    dc = np.diff(clear, axis=1)
    event = pair & (abs(dy) >= 100) & (abs(dy-dc) >= 100)
    detected = pair & (abs(dp) >= 100) & (abs(dp-dc) >= 100)
    correct = event & detected & (dy*dp > 0)
    tp, npred, ntrue = int(correct.sum()), int(detected.sum()), int(event.sum())
    p = tp/max(npred, 1)
    recall = tp/max(ntrue, 1)
    matched = 0
    for i in range(len(truth)):
        candidates = list(np.flatnonzero(detected[i]))
        for h in np.flatnonzero(event[i]):
            possible = [j for j in candidates if abs(j-h) <= 1 and dy[i, h]*dp[i, j] > 0]
            if possible:
                j = min(possible, key=lambda k: abs(k-h))
                candidates.remove(j)
                matched += 1
    wm = np.ma.array(abs(prediction-truth), mask=~valid).mean(0)
    out = dict(n_windows=len(truth), n_points=int(valid.sum()), true_weather_ramp_events=ntrue,
               predicted_weather_ramp_events=npred, mean24_MAE=float(wm.mean()),
               daytime_delta_MAE=float(abs(dp-dy)[pair].mean()) if pair.any() else None,
               event_delta_MAE=float(abs(dp-dy)[event].mean()) if event.any() else None,
               event_direction_accuracy=float((dp*dy > 0)[event].mean()) if event.any() else None,
               event_precision=p, event_recall=recall, event_F1=2*p*recall/max(p+recall, 1e-9),
               event_recall_within_1h=matched/max(ntrue, 1),
               event_precision_within_1h=matched/max(npred, 1))
    return out


def compare(dataset):
    out = ROOT/'results'/dataset
    if (out/'ramp_comparison.json').exists():
        return json.loads((out/'ramp_comparison.json').read_text())
    source = np.load(out/'test_predictions.npz')
    origins = source['origins']
    regions = source['regions'] if 'regions' in source else np.zeros(len(origins), int)
    ids = list(zip(regions.tolist(), origins.tolist()))
    id_set = set(ids)
    yy, mask = source['point_ghi_truth'], source['point_mask'].astype(bool)
    d = B/'smf_baseline_matrix_20261106/results'/dataset
    arrays, indices = {}, {}
    if dataset == 'SolarCube':
        clear = np.stack([np.load(ROOT/'cache/SolarCube'/f'clear_region{r:02d}.npy', mmap_mode='r')[t+np.arange(1, 25), -1]
                          for r, t in ids])
    else:
        clear = np.load(ROOT/'cache'/dataset/'clear.npy', mmap_mode='r')[origins[:, None]+np.arange(1, 25), -1]
    for name in BASELINES:
        folder = d/name
        if name in ['Persistence', 'Clear-sky']:
            arrays[name], indices[name] = {}, {}
            for r in np.unique(regions):
                p = folder/f'region_{r:02d}'
                index = np.load(p/'origins.npy')
                prediction = np.load(p/'prediction.npy', mmap_mode='r')
                truth = np.load(p/'truth.npy', mmap_mode='r')
                m = np.load(p/'mask.npy', mmap_mode='r')
                node = -1 if dataset == 'SolarCube' else 866 if dataset == 'CN-Solar' else 0
                for j, t in enumerate(index):
                    key = int(r), int(t)
                    if key in id_set:
                        arrays[name][key] = (np.array(prediction[j, :, node]), np.array(truth[j, :, node]), np.array(m[j, :, node]))
        elif dataset == 'SolarCube':
            index = np.load(folder/'sample_ids.npy')
            prediction = np.load(folder/'prediction.npy', mmap_mode='r')
            truth = np.load(folder/'truth.npy', mmap_mode='r')
            m = np.load(folder/'mask.npy', mmap_mode='r')
            lookup = {tuple(v): j for j, v in enumerate(index)}
            arrays[name] = {key: (np.array(prediction[lookup[key], :, -1]), np.array(truth[lookup[key], :, -1]),
                                  np.array(m[lookup[key], :, -1])) for key in ids if key in lookup}
        else:
            path = folder/'test_predictions.npz'
            z = np.load(path)
            if dataset == 'CN-Solar':
                prediction = column(path, 'field_ghi_prediction', 866)
                truth = column(path, 'field_ghi_truth', 866)
                m = column(path, 'field_ghi_mask', 866)
            else:
                prediction, truth, m = z['prediction'][:, :, 0], z['truth'][:, :, 0], z['mask'][:, :, 0]
            arrays[name] = {(0, int(t)): (prediction[j], truth[j], m[j]) for j, t in enumerate(z['origins'])}
    keep = np.array([i for i, key in enumerate(ids) if all(key in arrays[n] for n in BASELINES)], int)
    assert len(keep)
    valid = mask[keep].copy()
    base = {}
    for name in BASELINES:
        v, y, m = map(np.stack, zip(*(arrays[name][ids[i]] for i in keep)))
        common = valid & m.astype(bool)
        assert np.allclose(yy[keep][common], y[common], atol=1e-3)
        valid &= m.astype(bool) & np.isfinite(v)
        base[name] = v
    base['V9-RAMP-DIFFUSION-TemporalRamp deterministic'] = source['point_ghi_deterministic'][keep]
    base['V9-RAMP-DIFFUSION-TemporalRamp ensemble mean'] = source['point_ghi_samples'][keep].mean(1)
    valid &= np.isfinite(yy[keep])
    for v in base.values():
        valid &= np.isfinite(v)
    result = dict(dataset=dataset, total_new_model_test_windows=len(yy), common_test_windows=len(keep),
                  protocol='all common test windows, not selected illustration; masked daytime pairs; >=100W/m2/h and departure from known-clear ramp>=100W/m2/h; same direction for matched hits; +/-1h greedy one-to-one event matching',
                  models={name: scores(v, yy[keep], valid, clear[keep]) for name, v in base.items()})
    (out/'ramp_comparison.json').write_text(json.dumps(result, indent=2))
    return result
