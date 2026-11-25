# HyperSolar

HyperSolar is a research codebase for solar irradiance forecasting with a ramp-conditioned hypergraph model and a conditional residual diffusion model. It contains a four-dataset adaptation workflow and a separate high-cost 128 x 128 SolarCube experiment.

## Repository layout

```text
.
├── README.md
├── case_contract.json
└── code/
    ├── four_dataset_code/   # V9 temporal-ramp, hypergraph, and diffusion training
    └── solarcube_grid128/   # 128 x 128 model grid, native-120 supervision
```

## Method overview

The four-dataset model builds on the V9 Ramp-HG backbone. It encodes observed satellite/radiative products, NWP fields, and in-situ history; uses a temporal-ramp decoder conditioned only on observed station history; predicts field and selected-point clear-sky index (CSI); reconstructs irradiance using clear-sky irradiance; and models forecast residual uncertainty with a conditional DDPM. Training is end-to-end with field/point and CSI supervision plus a diffusion objective.

The code distinguishes observation sources and target semantics. For example, the SolarAlabama adapter uses a CERES irradiance proxy, not plant power or station-measured GHI. Refer to `case_contract.json` and each dataset's generated `READY.json`/`config.json` before comparing results.

## Training workflow

1. Prepare and audit dataset timestamps, coordinates, masks, and feature availability.
2. Fit input normalization on the training split only; keep chronological train/validation/test windows disjoint.
3. Run the smoke/leakage/gradient check (`--smoke`) before a full run.
4. Train with AdamW, gradient clipping, and validation-based early stopping. The four-dataset defaults are seed 42, 72-hour history, 24-hour horizon, 60 maximum epochs, patience 8, and batch size 8.
5. Restore the best validation checkpoint, run the held-out test evaluation once, and retain predictions, truth, per-lead metrics, configuration, and training history.

The SolarCube 128-grid experiment is separate: it uses a 128 x 128 computational grid, but source labels remain on the native 120 x 120 grid. Inputs are resampled to 128; model predictions are aggregated to native 120 for supervision and reported scores. No synthetic 128 x 128 labels are generated.

## Run the four-dataset workflow

The scripts currently contain paths specific to the original Linux server under `/mnt/wangyishan_share`. Before running on another host, configure those paths in `code/four_dataset_code/data.py` and `launch.py`, and make the prepared data available. Python dependencies include PyTorch, NumPy, SciPy, pandas, and pvlib; use a CUDA-compatible PyTorch build for training.

Smoke test:

```bash
cd code/four_dataset_code
python train.py --dataset SolarCube --batch 8 --smoke
```

Single dataset:

```bash
python train.py --dataset SolarCube --batch 8 --epochs 60 --patience 8
```

Sequential queue on a selected GPU:

```bash
python launch.py --gpu 0 --datasets SolarCube CN-Solar NSRDB SolarAlabama --batch 8
```

The launcher smoke-tests each dataset before training it and stops on errors. Check `DONE.json` and `metrics.json` for each completed dataset. `case_contract.json` records the fixed illustrative examples and their target/provenance notes.

## Run the 128 x 128 SolarCube experiment

This entry point also uses server-specific paths and requires the audited native SolarCube/ERA5 caches. Adapt the paths in `code/solarcube_grid128/train_v9_ramp_hg128.py` and `native_data.py` first.

```bash
cd code/solarcube_grid128
python train_v9_ramp_hg128.py --out /path/to/output_directory
```

## Outputs

The four-dataset workflow writes `config.json`, `train.jsonl`, `training_history.json`, `best.pt`, `test_predictions.npz`, `metrics.json`, and `DONE.json` under the server-side results directory. The 128-grid workflow writes `train.log`, `best.pt`, and `last.pt`; test archives and metrics are written only after successful early stopping and evaluation.

## Data and result limitations

- The target is irradiance, not electrical power.
- Dataset targets differ in source and meaning. Do not combine scores across datasets or workflows without matching target, split, temporal alignment, sampling, and metric definitions.
- NSRDB timestamp correction and SolarCube field-grid construction are documented in the adapters; audit them before reuse.
- No datasets, caches, or model checkpoints are included in this repository.
- The 128-grid SolarCube training entry point is computationally expensive; begin with a validated smoke run and monitor storage, GPU memory, and runtime.

## Provenance

The training code was copied from the author's server 192 (`192.168.60.1`) on 2026-11-25. The source tree includes original V9 dependencies used by the adapters. Historical monitoring scripts and generated Python bytecode are excluded from the Git commit.
