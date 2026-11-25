"""Render V9-Ramp-HG Grid128 results in the established SMF-Results 9x24 style.

Runs on server 192 after the Grid128 test archive has been written. The main
comparison uses the common observed SolarCube native-120 verification grid;
an auxiliary panel displays the model's 128x128 mesh beside truth upsampled
for visual alignment only.
"""
from pathlib import Path
import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

BASE = Path(os.environ.get("HYPERSOLAR_DATA_DIR", "data"))
REG = Path(os.environ.get("HYPERSOLAR_REGISTRY_DATA", BASE))
NATIVE = Path(os.environ["HYPERSOLAR_NATIVE120_DATA"])
RUN = Path(os.environ["HYPERSOLAR_RUN_DIR"])
CASES = Path(os.environ.get("HYPERSOLAR_CASES_DIR", "cases"))
OUT = RUN / "figures_9x24"
OUT.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(4)
sys_path = [str(REG), str(REG / "code/v9_ramp_hg128_20261110")]
import sys
sys.path[:0] = sys_path
from native_data import NativeWindows

dataset_args = (
    str(REG / "SolarCube_2018_full19_hourly"),
    os.environ["HYPERSOLAR_SOLARCUBE_PREPARED"],
    os.environ["HYPERSOLAR_SOLARCUBE_SITELIST"],
    str(REG / "SolarCube_2018_era5_8x8_v7"),
    "test",
)
ds = NativeWindows(*dataset_args, stride=3, native=str(NATIVE))
row_to_idx = {tuple(map(int, row)): i for i, row in enumerate(ds.rows)}

METHODS = [
    ("Truth hourly mean", "truth_hourly"),
    ("Persistence", "Persistence"),
    ("Clear-sky · known geometry", "Clear-sky"),
    ("Baguan-Solar", None),
    ("GraphCast · resampled", "GraphCast_(resampled)"),
    ("SolarFormer · resampled", "SolarFormer_(resampled)"),
    ("SMF V5 · resampled", "SMF_V5_(resampled)"),
    ("V9 CloudState · resampled", "V9_CloudState_(resampled)"),
    ("V9-Ramp-HG · Grid128", "V9_Ramp_HG_Grid128"),
]
CASE_ROWS = [(1, 10, 8450), (2, 8, 8438), (3, 8, 7691)]
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})


def retrieve_grid128(case_no, region, origin):
    idx = row_to_idx[(region - 1, origin)]
    batch_idx, offset = divmod(idx, 4)
    p = RUN / "test_predictions" / f"{batch_idx:05d}.npz"
    with np.load(p) as z:
        pred120 = z["field_ensemble_native120"][offset].mean(axis=0).reshape(24, 120, 120)
        pred128 = z["field_prediction_128_mean"][offset].astype(np.float32)
    return pred120, pred128


lead_metrics = []
mean_metrics = []
for case, region, origin in CASE_ROWS:
    with np.load(CASES / f"case{case}_data.npz") as z:
        a = {k: z[k].copy() for k in z.files if k != "truth_snapshot"}
    # Replace the former native-grid V9-Ramp row with this experiment's
    # ensemble-mean Grid128 forecast, area-aggregated onto the truth grid.
    pred120, pred128 = retrieve_grid128(case, region, origin)
    a["V9_Ramp_HG_Grid128"] = pred120
    y = a["truth_hourly"]
    common = np.isfinite(y)
    for name, key in METHODS[1:]:
        if key is not None:
            common &= np.isfinite(a[key])

    fig, ax = plt.subplots(9, 24, figsize=(48, 18), layout="constrained")
    image = None
    for i, (name, key) in enumerate(METHODS):
        for h in range(24):
            aa = ax[i, h]
            if key is None:
                aa.set_facecolor("#eeeeee")
                aa.text(.5, .5, "Unavailable\nmissing inputs", ha="center",
                        va="center", fontsize=7)
            else:
                pred = a[key][h]
                display = np.where(common[h], pred, np.nan)
                image = aa.imshow(display, vmin=0, vmax=1200,
                                  cmap="RdYlBu_r", interpolation="nearest")
                if i > 0:
                    e = (pred - y[h])[common[h]]
                    mae = float(np.abs(e).mean()) if e.size else None
                    rmse = float(np.sqrt(np.square(e).mean())) if e.size else None
                    aa.set_title(f"MAE {mae:.1f}\nRMSE {rmse:.1f}" if e.size else "No common score",
                                 fontsize=7)
                    lead_metrics.append({"case": case, "region": region,
                                         "method": name, "lead_hour": h + 1,
                                         "MAE_Wm2": mae, "RMSE_Wm2": rmse,
                                         "valid_pixels": int(common[h].sum())})
            if i == 0:
                aa.set_title(f"+{h+1:02d}h", fontsize=10, fontweight="bold")
            aa.set_xticks([])
            aa.set_yticks([])
        ax[i, 0].set_ylabel(name, fontsize=10)
    fig.colorbar(image, ax=ax.ravel().tolist(), shrink=.35,
                 label="Solar radiation (W/m²)", extend="max")
    fig.suptitle(
        f"SolarCube region{region:02d} · case{case} · all 24 hourly leads · 9×24\n"
        "Native-120 truth and common-grid scores; V9-Ramp-HG Grid128 forecast area-aggregated to 120×120\n"
        "ERA5 and satellite inputs interpolated to 128 do not add observed resolution; Baguan-Solar unavailable",
        fontsize=13)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"case{case}_region{region:02d}_9x24.{ext}",
                    dpi=180 if ext == "png" else 120, bbox_inches="tight")
    plt.close(fig)

    records = [m for m in lead_metrics if m["case"] == case]
    for method, _ in METHODS[1:]:
        rr = [v for v in records if v["method"] == method]
        vals = [v["MAE_Wm2"] for v in rr if v["MAE_Wm2"] is not None]
        rmses = [v["RMSE_Wm2"] for v in rr if v["RMSE_Wm2"] is not None]
        mean_metrics.append({"case": case, "region": region, "method": method,
                            "valid_leads": len(vals),
                            "MAE_mean_valid_leads_Wm2": float(np.mean(vals)) if vals else None,
                            "RMSE_mean_valid_leads_Wm2": float(np.mean(rmses)) if rmses else None})

    # One row for the fine mesh forecast, one for the observed truth interpolated
    # only to align the display. The scored truth remains native120 above.
    from torch.nn.functional import interpolate
    truth128 = interpolate(torch.from_numpy(y[:, None].astype(np.float32)),
                           size=(128, 128), mode="bilinear", align_corners=False)[:, 0].numpy()
    fig2, ax2 = plt.subplots(2, 24, figsize=(48, 6), layout="constrained")
    for h in range(24):
        ax2[0, h].imshow(truth128[h], vmin=0, vmax=1200, cmap="RdYlBu_r", interpolation="nearest")
        ax2[1, h].imshow(pred128[h], vmin=0, vmax=1200, cmap="RdYlBu_r", interpolation="nearest")
        ax2[0, h].set_title(f"+{h+1:02d}h", fontsize=9, fontweight="bold")
        for row in (0, 1):
            ax2[row, h].set_xticks([])
            ax2[row, h].set_yticks([])
    ax2[0, 0].set_ylabel("Truth upsampled\n120→128", fontsize=10)
    ax2[1, 0].set_ylabel("V9-Ramp-HG\nGrid128", fontsize=10)
    fig2.colorbar(ax2[1, -1].images[0], ax=ax2.ravel().tolist(), shrink=.6,
                  label="Solar radiation (W/m²)")
    fig2.suptitle("Grid128 visualization only; truth was bilinearly displayed from native120 and is not a 128-resolution observation", fontsize=12)
    for ext in ("png", "pdf", "svg"):
        fig2.savefig(OUT / f"case{case}_region{region:02d}_grid128_leads.{ext}",
                     dpi=180 if ext == "png" else 120, bbox_inches="tight")
    plt.close(fig2)
    print(f"CASE_DONE {case}", flush=True)


for name, rows in (("per_lead_metrics.csv", lead_metrics),
                    ("case_mean24_metrics.csv", mean_metrics)):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
(OUT / "README.txt").write_text(
    "SMF-Results layout: 9 rows × 24 hourly leads, one column per +1…+24 h. "
    "The comparison panels use the common native120 truth grid; the new 128 forecast "
    "is area-aggregated to native120 for fair scoring. Separate 2×24 panels show "
    "Grid128 predictions and truth bilinearly displayed at 128 solely for visual alignment. "
    "No 128-resolution observations or labels are claimed. The selected cases are illustrative, "
    "not the full-test aggregate; report full-test metrics from metrics.json. Baguan-Solar "
    "remains unavailable and is not substituted.", encoding="utf-8")
print("FIGURES_COMPLETE", flush=True)
