"""Append lead-averaged result summaries to per-run READMEs."""
from pathlib import Path
import numpy as np


def append_fieldpoint_readme(out: Path, result: dict) -> None:
    readme = out / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.exists() else f"# {out.name}\n"
    marker = "## Test-set summary"
    if marker in text:
        return
    lines = ["", "", marker, "",
             f"Best epoch: {result['best_epoch']}; epochs run: {result['epochs_ran']}; "
             f"early stopped: {result['stopped_early']}; training seconds: {result['train_seconds']:.1f}.", ""]
    for label, key in (("Sparse-field", "field_metrics_by_lead"), ("Selected-point", "point_metrics_by_lead")):
        rows = result[key]
        values = {m: float(np.mean([r[m] for r in rows]))
                  for m in ("MAE", "RMSE", "CRPS", "PI50_coverage", "PI80_coverage", "PI90_coverage", "PI90_width")}
        lines.append(label + " equal-weight mean over 24 leads: " +
                     ", ".join(f"{m}={v:.4f}" for m, v in values.items()) + ".")
    lines += ["", "`test_predictions.npz` contains field and point ensembles, exact test truth, masks, timestamps, "
              "coordinates and selected-site identity. Per-lead probabilistic and deterministic metrics are in `test_metrics.json`.", ""]
    readme.write_text(text + "\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    import json
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    a = p.parse_args()
    root = Path(a.run_dir)
    result = json.loads((root / "test_metrics.json").read_text(encoding="utf-8"))
    append_fieldpoint_readme(root, result)
