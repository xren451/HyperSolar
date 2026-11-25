"""A finite queue on one explicitly selected GPU; never stops other processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--gpu', required=True)
p.add_argument('--datasets', nargs='+', required=True)
p.add_argument('--batch', type=int, default=8)
a = p.parse_args()
root = Path('/mnt/wangyishan_share/wangdongdong/smf_v9_four_datasets_20261111')
env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4')
for dataset in a.datasets:
    path = root/'queue'/f'{dataset}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    def status(state, **kwargs):
        path.write_text(json.dumps(dict(dataset=dataset, gpu=a.gpu, state=state, time=time.time(), **kwargs), indent=2))
    cmd = [sys.executable, str(Path(__file__).with_name('train.py')), '--dataset', dataset, '--batch', str(a.batch)]
    status('preparing_and_smoke_testing')
    with (root/'queue'/f'{dataset}.log').open('a', buffering=1) as log:
        run = subprocess.run(cmd+['--smoke'], env=env, stdout=log, stderr=subprocess.STDOUT)
        if run.returncode:
            status('smoke_failed', returncode=run.returncode)
            raise SystemExit(run.returncode)
        status('full_training')
        run = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        if run.returncode:
            status('training_failed', returncode=run.returncode)
            raise SystemExit(run.returncode)
    status('complete')
