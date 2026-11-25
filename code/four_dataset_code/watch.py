"""Finite completion watcher: exports actual previews/final curves and audits."""
import argparse
import csv
import json
from pathlib import Path
import tarfile
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import ROOT, B
from ramp_metrics import compare

DS = ['CN-Solar', 'SolarCube', 'NSRDB', 'SolarAlabama']
BASE = ['Persistence', 'Clear-sky', 'SolarFormer', 'SolarSeer', 'GraphCast']
OUT = ROOT/'visuals'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': ['DejaVu Sans', 'sans-serif'], 'font.size': 9,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})
contract = json.loads((ROOT/'case_contract.json').read_text())


def snapshot():
    status, examples, signatures = {}, {}, []
    for ds in DS:
        folder = ROOT/'results'/ds
        state = folder/'status.json'
        try:
            status[ds] = json.loads(state.read_text()) if state.exists() else {'state': 'preparing_or_queued'}
        except (OSError, json.JSONDecodeError):
            status[ds] = {'state': 'updating'}
        done = folder/'DONE.json'
        final = done.exists()
        source = folder/('test_predictions.npz' if final else 'preview.npz')
        if not source.exists():
            signatures.append((ds, None))
            continue
        z = np.load(source)
        if final:
            matching = z['origins'] == contract[ds]['origin']
            if 'regions' in z:
                matching &= z['regions'] == 9
            idx = np.flatnonzero(matching)
            assert len(idx) == 1
            i = idx[0]
            det, samples, y, mask = [z[k][i] for k in ['point_ghi_deterministic', 'point_ghi_samples', 'point_ghi_truth', 'point_mask']]
        else:
            det, samples, y, mask = [z[k] for k in ['point_ghi_deterministic', 'point_ghi_samples', 'point_ghi_truth', 'point_mask']]
        old = np.load(B/'smf_four_dataset_irradiance_20261111'/f'{ds}_selected_curves.npz')
        assert np.allclose(old['Truth'][mask], y[mask], atol=1e-3)
        config = json.loads((folder/'config.json').read_text())
        examples[ds] = dict(prediction=det, samples=samples, truth=y, mask=mask,
                            baselines={n: old[n] for n in BASE}, epoch=int(z['checkpoint_epoch']),
                            final=final, config=config)
        signatures.append((ds, final, source.stat().st_mtime_ns))
    temp = OUT/'training_status.tmp'
    temp.write_text(json.dumps(status, indent=2))
    temp.replace(OUT/'training_status.json')
    return status, examples, signatures


def panel(ax, ds, example):
    x = np.arange(1, 25)
    mask = example['mask'].astype(bool)
    c = contract[ds]
    lo, hi = np.quantile(example['samples'], [.1, .9], axis=0)
    ax.fill_between(x, lo, hi, color='#c43c39', alpha=.12, label='8-member 10-90% range')
    ax.plot(x, np.where(mask, example['truth'], np.nan), color='black', lw=2.4, marker='o', ms=3, label='Truth')
    colors = ['#8c8c8c', '#d99928', '#367cb7', '#32966c', '#8b68aa']
    for color, (name, values) in zip(colors, example['baselines'].items()):
        ax.plot(x, np.where(mask, values, np.nan), color=color, ls='--', lw=1., label=name)
    ax.plot(x, np.where(mask, example['prediction'], np.nan), color='#c43c39', lw=2.4,
            marker='s', ms=3, label='V9 TemporalRamp deterministic')
    ax.plot(x, np.where(mask, example['samples'].mean(0), np.nan), color='#a55623', ls=':', lw=1.4,
            label='V9 diffusion ensemble mean')
    stage = 'final validation-selected' if example['final'] else 'INTERIM - training not finished'
    ax.set_title(f"{ds} | {c['site']} | epoch {example['epoch']} | {stage}\n{c['target']} | origin {c['origin_utc']}", fontsize=9, pad=90)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.02), ncol=3, fontsize=7)
    ax.set_xlabel('Forecast lead (h)')
    ax.set_ylabel('Irradiance (W/m^2)')
    ax.set_xticks(np.arange(1, 25))
    ax.set_xlim(1, 24)
    ax.grid(alpha=.18)


def export(status, examples):
    csvrows = []
    for ds, example in examples.items():
        fig, ax = plt.subplots(figsize=(12, 6.5), layout='constrained')
        panel(ax, ds, example)
        fig.text(.5, -.02, 'New TemporalRamp adaptation of V9-RAMP-DIFFUSION. Original predictions only, no truth mixing. Uncertainty range is not calibrated.', ha='center', fontsize=9)
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(OUT/f'{ds}_V9_RAMP_DIFFUSION_hourly.{ext}', dpi=300, bbox_inches='tight')
        plt.close(fig)
        for h in range(24):
            csvrows.append(dict(dataset=ds, lead_hour=h+1, checkpoint_epoch=example['epoch'],
                                final=example['final'], Truth=float(example['truth'][h]),
                                V9_deterministic=float(example['prediction'][h]),
                                V9_ensemble_mean=float(example['samples'][:, h].mean()),
                                **{n: float(v[h]) for n, v in example['baselines'].items()}))
    if len(examples) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(18, 14), layout='constrained')
        for ax, ds in zip(axes.ravel(), DS):
            panel(ax, ds, examples[ds])
        fig.text(.5, -.01, 'V9-RAMP-DIFFUSION-TemporalRamp-v1 on all four datasets; deterministic + sampled residual diffusion. No future-truth blending.', ha='center', fontsize=10)
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(OUT/f'four_datasets_V9_RAMP_DIFFUSION.{ext}', dpi=300, bbox_inches='tight')
        plt.close(fig)
    if csvrows:
        with (OUT/'hourly_curves.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(csvrows[0]))
            w.writeheader()
            w.writerows(csvrows)
    metadata = {ds: dict(final=e['final'], epoch=e['epoch'], model=e['config']['model'],
                         source=contract[ds], initialization=e['config'].get('initialization', 'trained from scratch, seed42'))
                for ds, e in examples.items()}
    (OUT/'plot_provenance.json').write_text(json.dumps(metadata, indent=2))
    html = '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>V9 four-dataset training</title><style>body{font-family:Arial;margin:24px}img{width:100%}p{line-height:1.6}a{margin-right:18px}</style>'
    html += '<h1>V9-RAMP-DIFFUSION-TemporalRamp-v1 · four datasets</h1><p>新版本：观测历史交叉注意力未来解码器＋变化量、曲率、上升/下降事件损失。使用原始预测，不混入未来真实值。INTERIM图为训练中检查点，不是最终结果。图中不标MAE/RMSE；完整测试Ramp指标完成后另列。CN为格点，SolarAlabama为CERES代理值。</p>'
    for ds in DS:
        html += f'<h2>{ds}</h2><pre>{json.dumps(status[ds], ensure_ascii=False, indent=2)}</pre>'
        if ds in examples:
            stem = f'{ds}_V9_RAMP_DIFFUSION_hourly'
            html += f'<a href="{stem}.pdf">PDF</a><a href="{stem}.svg">SVG</a><img src="{stem}.png">'
        else:
            html += '<p>尚无该版本的训练后曲线；不会用旧模型或未来真值替代。</p>'
    if len(examples) == 4:
        html += '<h2>Four-dataset overview</h2><a href="four_datasets_V9_RAMP_DIFFUSION.pdf">PDF</a><img src="four_datasets_V9_RAMP_DIFFUSION.png">'
    html += '<p><a href="hourly_curves.csv">逐小时数据</a><a href="plot_provenance.json">版本与来源</a><a href="ramp_comparison.json">完整测试Ramp指标（完成后）</a></p>'
    (OUT/'index.html').write_text(html, encoding='utf-8')
    (OUT/'README.txt').write_text('All plots use actual new TemporalRamp-v1 model output, with explicit interim/final checkpoint labels. No truth blending, no V2/V6 renaming. Dataset adapters and model extension documented in configs. SolarCube warm-starts original train/val-only V9 backbone; other3train from scratch. Dense computational grid16 is not native120 field training. Sparse target cells are masked source-node aggregates. GHI/product/proxy sources differ. Ensemble10to90% range uses8members, not calibrated confidence. Fixed examples from previous figure contract; never used to fit/select weights. All-test ramp metrics must determine improvement, not picture aesthetics.', encoding='utf-8')
    archive = ROOT/'visuals.tmp.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for p in OUT.iterdir():
            tar.add(p, arcname=p.name)
    archive.replace(ROOT/'visuals.tar.gz')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--once', action='store_true')
    a = ap.parse_args()
    last = None
    deadline = time.time()+7*24*3600
    while time.time() < deadline:
        status, examples, signature = snapshot()
        if signature != last:
            export(status, examples)
            last = signature
        if all((ROOT/'results'/ds/'DONE.json').exists() for ds in DS):
            audits = [compare(ds) for ds in DS]
            (OUT/'ramp_comparison.json').write_text(json.dumps(audits, indent=2))
            records = [dict(dataset=audit['dataset'], model=name, **score)
                       for audit in audits for name, score in audit['models'].items()]
            with (OUT/'ramp_comparison.csv').open('w') as f:
                w = csv.DictWriter(f, fieldnames=list(records[0]))
                w.writeheader()
                w.writerows(records)
            export(status, examples)
            (ROOT/'VISUALS_DONE.json').write_text(json.dumps({'complete': True, 'datasets': DS, 'time': time.time()}, indent=2))
            print('FOUR_DATASET_EXPORT_COMPLETE', flush=True)
            return
        if a.once:
            return
        time.sleep(30)
    raise TimeoutError('Finite seven-day job watcher expired; inspect queue and logs')


if __name__ == '__main__':
    main()
