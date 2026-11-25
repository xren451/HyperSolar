# V9-Ramp-HG 128-grid SolarCube experiment

Model grid: 128x128. SolarCube 2018 all 19 regions; 72-hour history; 24-hour forecast; chronological 70/15/15 split; stride 3; seed 42; batch 1; patience 8 (restore best checkpoint).

Source inputs/labels are the audited native 120x120 SolarCube cache. Raster inputs are bilinearly resampled from 120 to 128 for model encoding. Deterministic field outputs are produced at 128, then area-aggregated to 120 for training losses and all reported test metrics against native labels. Residual diffusion is supervised on the native 120 field/point CSI residual and its field residual is bilinearly mapped to the 128 forecast grid. No 128x128 ground-truth pixels are fabricated.

V9 Ramp-HG keeps the ResNet-Swin-ViT encoder, observed-history ramp-conditioned local hyperedge, global and same-cell hyperedges, field/point CSI decoders, and conditional residual DDPM. Sparse incidence is retained. GPU4 is selected by CUDA_VISIBLE_DEVICES=4; other GPU4 processes are not stopped.
