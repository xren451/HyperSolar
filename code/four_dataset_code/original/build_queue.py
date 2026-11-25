#!/usr/bin/env python3
"""Build a transparent, dependency-aware screening queue from registry.json.

This only materializes run specifications. The dispatcher must refuse jobs whose
dataset adapter or architecture implementation has not passed its data-contract
and smoke checks; this prevents relabeling an older V1/V2 trainer as a new model.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REGISTRY = json.loads((ROOT / "registry.json").read_text(encoding="utf-8"))
DATASETS = list(REGISTRY["datasets"])
variants = {v["id"]: v for v in REGISTRY["variants"]}

jobs = []
for dataset in DATASETS:
    for vid, v in variants.items():
        if v.get("scope") and dataset not in v["scope"]:
            continue
        if v["phase"] == "frozen_reference":
            state = "reuse_reference_power_protocol_not_irradiance_comparable"
        elif vid == "SMF-V2":
            state = "running_full_irradiance_control"
        elif v["phase"] in ("position_screen", "cloud_screen"):
            state = "blocked_on_encoder_screen_winner"
        else:
            state = "blocked_on_implementation_and_data_gate"
        jobs.append({
            "run_id": f"{dataset}__{vid.replace('-', '_')}__seed42",
            "dataset": dataset,
            "variant": vid,
            "phase": v["phase"],
            "state": state,
            "early_stopping": REGISTRY["early_stopping"],
            "max_epochs": REGISTRY["max_epochs"],
            "output": f"results/{dataset}/{vid.replace('-', '_')}_seed42",
            "requires_data_gate": REGISTRY["datasets"][dataset]["protocol_gate"],
        })

for job in jobs:
    if job["variant"] == "SMF-V2":
        job["output"] = {
            "SolarCube": "results/SolarCube/SMF_V2_full19_irradiance_seed42_geoMapped_cached16",
            "CN-Solar": "results/CN-Solar/SMF_V2_seed42",
            "NSRDB": "results/NSRDB/SMF_V2_irradiance_seed42",
            "SolarAlabama": "results/SolarAlabama/SMF_V2_CERES_irradiance_seed42",
        }[job["dataset"]]

queue = {"benchmark_id": REGISTRY["benchmark_id"], "host": REGISTRY["server"],
         "created": "2026-11-05", "jobs": jobs}
(ROOT / "queue.json").write_text(json.dumps(queue, indent=2), encoding="utf-8")
print(f"Wrote {len(jobs)} run specifications to {ROOT / 'queue.json'}")
