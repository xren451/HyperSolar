# HyperSolar

HyperSolar is a research codebase for field-and-point solar irradiance forecasting. It collects the V1–V9 model families, dataset loaders, training entry points, and a separate high-cost 128×128 SolarCube configuration. The code predicts irradiance (GHI/CSI), not electrical power.

## Model versions

| Version | Main design | Main source files |
|---|---|---|
| V1 | Deterministic SMF field forecast with a target-residual diffusion branch and selected-point decoder. | `code/models/v1_v2_station_models.py`, `code/models/field_point_versions.py` |
| V2 | Predict an intermediate radiative/cloud state, then decode field and point irradiance; diffuse state residuals. | Same model files; `train_v2_*.py` |
| V3 | Vision-Swin raster encoder screen. | `code/models/encoder_registry.py`; encoder-screen trainer |
| V4 | Residual-convolution plus ViT raster encoder. | `code/models/encoder_registry.py`; encoder-screen trainer |
| V5 | Residual-convolution + Swin local attention + ViT global interaction encoder. | `code/models/encoder_registry.py`; encoder-screen trainer |
| V6 | Validation-fitted clear-sky/SMF forecast gate. | `code/models/v6_clear_sky_gate.py` |
| V7 | Branch7 field/point CSI forecast with regional and local cloud/ramp features and bidirectional refinement. | `code/models/train_v7_branch7.py` |
| V8 | V3-style deterministic encoder with residual diffusion participating in message refinement. | `code/models/train_v8_diffusion_message.py` |
| V9 | Hypergraph future-state information exchange plus conditional residual diffusion; Ramp-HG adds observed-history ramp conditioning. | `code/models/v9_cloudstate_hypergraph.py`, `code/models/v9_ramp_hyperedge.py`, `code/models/train_v9_*.py` |

V3–V5 are encoder variants in the shared encoder registry rather than three unrelated end-to-end architectures. Check each entry point/config before comparing results: protocols, target products, grid resolution and availability of field supervision differ between experiment branches.

## Repository layout

```text
.
├── README.md
├── case_contract.json
└── code/
    ├── four_dataset_code/   # dataset adapters, V9 model, training and queue entry points
    ├── models/              # V1–V9 model definitions and version-specific trainers
    └── solarcube_grid128/   # V9-Ramp-HG 128×128 model-grid experiment
```

## Data loader

The four-dataset loader is in `code/four_dataset_code/data.py`. It creates chronological train/validation/test windows, aligns feature and target masks, builds 16×16 spatial inputs from the dataset's source nodes, derives future clear-sky geometry from timestamps and coordinates, and fits normalization statistics on training data only. SolarCube has a dedicated cache/preparation adapter; see `prepare_cube.py`.

Configure local data locations through environment variables; no machine-specific directory is embedded:

```bash
HYPERSOLAR_DATA_DIR=/path/to/data
HYPERSOLAR_CACHE_DIR=/path/to/cache
HYPERSOLAR_CNSOLAR_ARCHIVE=/path/to/cn_solar.npz
HYPERSOLAR_NSRDB_ARCHIVE=/path/to/nsrdb.npz
HYPERSOLAR_SOLARALABAMA_ARCHIVE=/path/to/solar_alabama.npz
HYPERSOLAR_SOLARCUBE_SITELIST=/path/to/solarcube_sitelist.csv
```

The archive contracts and feature names are specified in `data.py`. SolarAlabama uses a CERES irradiance proxy in this adapter; it is not station-measured GHI or plant power. NSRDB and CN-Solar targets also follow their archive/product definitions. Do not treat these target sources as interchangeable.

## Training workflow

1. Configure data/archive paths and inspect the dataset contract.
2. Run a smoke test to check tensor shapes, finite losses, gradients, and future-label isolation.
3. Train chronologically with AdamW, gradient clipping, and validation-based early stopping.
4. Restore the best validation checkpoint and evaluate the held-out test split once.
5. Preserve config, training history/log, checkpoint, test truth/prediction archive, metrics, and plots.

Four-dataset defaults are seed 42, 72-hour history, 24-hour horizon, 60 maximum epochs, patience 8, and batch size 8. Smoke test and single-dataset examples:

```bash
cd code/four_dataset_code
python train.py --dataset SolarCube --batch 8 --smoke
python train.py --dataset SolarCube --batch 8 --epochs 60 --patience 8
python launch.py --gpu 0 --datasets SolarCube CN-Solar NSRDB SolarAlabama --batch 8
```

The queue launcher smoke-tests each dataset before full training and exits on failure. A run is complete only when its `DONE.json` and `metrics.json` are written.

## SolarCube 128×128 experiment

`code/solarcube_grid128/` is a separate V9-Ramp-HG configuration. Inputs are mapped to a 128×128 computational grid, while the observed target labels remain on the native 120×120 grid. Predictions are aggregated back to native resolution for training loss and reported scores; no synthetic 128×128 ground-truth labels are created. Configure paths using environment variables before running the entry point:

```bash
HYPERSOLAR_REGISTRY_DATA=/path/to/solarcube_inputs \
HYPERSOLAR_NATIVE120_DATA=/path/to/native120_cache \
HYPERSOLAR_SOLARCUBE_PREPARED=/path/to/solarcube_prepared \
HYPERSOLAR_SOLARCUBE_SITELIST=/path/to/site_list.csv \
python code/solarcube_grid128/train_v9_ramp_hg128.py --out /path/to/output
```

## Requirements and outputs

The training code requires Python, PyTorch with CUDA for GPU training, NumPy, SciPy, pandas, and pvlib. Results are written to the configured cache/output directories and include model configuration, logs/history, best checkpoint, test predictions/truth, and metrics. No dataset archives, cache files, or trained weights are distributed in this repository.

## Reproducibility notes

- Keep train/validation/test splits chronological and fit normalization on training data only.
- Use the same target product, valid-data mask, spatial grid, temporal alignment, and test origins for a fair comparison.
- The 128×128 SolarCube model grid is a computational mesh, not a claim of 128×128 observational resolution.
- V3–V5 encoder screens and V1/V2/V6–V9 full models are different experiment families; cite the exact config and checkpoint used for any reported score.
