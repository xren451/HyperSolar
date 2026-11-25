#!/usr/bin/env python3
"""Build a full-19-region hourly SolarCube irradiance cache without RAM copies.

Quarter-hour inputs are grouped by source index, matching the official
SolarCube loader convention. The dense field label is satellite ``ssr``;
the separate point label is the corresponding tile's smoothed in-situ series.
"""
from __future__ import annotations
import argparse, csv, json, zipfile
from pathlib import Path
import numpy as np

CHANNELS = ["sza", "cm", "ir133", "ssr", "vis047", "vis086"]

def build_region(src: Path, out: Path, ground: np.ndarray, ground_count: np.ndarray,
                 tile: int, grid_size=64, chunk_hours=128):
    folder = src / f"region_{tile:02d}"
    values = np.load(folder / "values.npy", mmap_mode="r")  # [quarter,C,H,W]
    valid = np.load(folder / "mask.npy", mmap_mode="r")
    if values.shape != (35040, 6, grid_size, grid_size) or valid.shape != values.shape:
        raise ValueError(f"region {tile}: unexpected values/mask shape {values.shape}/{valid.shape}")
    out.mkdir(parents=True, exist_ok=False)
    n_hour = values.shape[0] // 4
    feat = np.lib.format.open_memmap(out / "features.npy", mode="w+", dtype=np.float16,
                                     shape=(n_hour, grid_size, grid_size, 6))
    field = np.lib.format.open_memmap(out / "field_ssr.npy", mode="w+", dtype=np.float16,
                                      shape=(n_hour, grid_size, grid_size))
    fm = np.lib.format.open_memmap(out / "field_mask.npy", mode="w+", dtype=np.bool_,
                                   shape=(n_hour, grid_size, grid_size))
    state = np.lib.format.open_memmap(out / "cloud_state.npy", mode="w+", dtype=np.float16,
                                      shape=(n_hour, grid_size, grid_size))
    stm = np.lib.format.open_memmap(out / "cloud_state_mask.npy", mode="w+", dtype=np.bool_,
                                    shape=(n_hour, grid_size, grid_size))
    for h0 in range(0, n_hour, chunk_hours):
        h1 = min(n_hour, h0 + chunk_hours)
        q0, q1 = 4*h0, 4*h1
        x = np.asarray(values[q0:q1], dtype=np.float32)
        m = np.asarray(valid[q0:q1], dtype=bool) & np.isfinite(x)
        x = x.reshape(h1-h0, 4, 6, grid_size, grid_size)
        m = m.reshape(h1-h0, 4, 6, grid_size, grid_size)
        cnt = m.sum(axis=1)
        sums = np.where(m, x, 0).sum(axis=1)
        mean = np.divide(sums, cnt, out=np.full_like(sums, np.nan), where=cnt > 0)
        feat[h0:h1] = mean.transpose(0, 2, 3, 1).astype(np.float16)
        # Keep hourly SSR only if at least half of its four quarter-hour frames
        # are valid. Nighttime zero irradiance remains a valid physical label.
        ssr_count = cnt[:, 3]
        sm = ssr_count >= 2
        field[h0:h1] = np.where(sm, mean[:, 3], 0).astype(np.float16)
        fm[h0:h1] = sm
        cm_count = cnt[:, 1]
        cm_mask = cm_count >= 2
        state[h0:h1] = np.where(cm_mask, mean[:, 1], 0).astype(np.float16)
        stm[h0:h1] = cm_mask
    feat.flush(); field.flush(); fm.flush(); state.flush(); stm.flush()

    gp = np.asarray(ground[:, tile-1], dtype=np.float32).reshape(n_hour, 4)
    gc = np.asarray(ground_count[:, tile-1], dtype=np.int16).reshape(n_hour, 4)
    gm = np.isfinite(gp) & (gc >= 12)
    nvalid = gm.sum(axis=1)
    point_mask = nvalid >= 3
    point = np.divide(np.where(gm, gp, 0).sum(axis=1), nvalid,
                      out=np.zeros(n_hour, dtype=np.float32), where=nvalid > 0)
    np.save(out / "point_insitu.npy", point.astype(np.float32))
    np.save(out / "point_mask.npy", point_mask)
    meta = {"region": tile, "source_frames": 35040, "hourly_steps": n_hour,
            "channels": CHANNELS, "feature_layout": "T,H,W,C", "field_target": "satellite SSR W m-2",
            "point_target": "collocated in-situ station irradiance W m-2; separate label from field SSR",
            "source_index_alignment": "four consecutive UTC quarter-hour indices per hourly bin; station smoothed quarter bins are paired by official SolarCube index order",
            "field_min_valid_quarters": 2, "point_min_valid_quarters": 3,
            "point_min_valid_minutes_per_quarter": 12,
            "valid_field_fraction": float(fm.mean()), "valid_cloud_state_fraction": float(stm.mean()),
            "valid_point_hours": int(point_mask.sum()), "selected_station_grid_index": [grid_size//2, grid_size//2],
            "selected_station_label_mean_train": float(point[:int(n_hour*.70)][point_mask[:int(n_hour*.70)]].mean()),
            "selected_station_label_std_train": float(point[:int(n_hour*.70)][point_mask[:int(n_hour*.70)]].std()),
            "hourly_time_first": "2018-01-01T00:00:00Z", "hourly_time_last": "2018-12-31T23:00:00Z"}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prepared",type=Path,required=True); p.add_argument("--raw",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(f"refusing to overwrite {a.output}")
    ready=json.loads((a.prepared/"READY.json").read_text())
    if not ready.get("complete") or len(ready.get("regions",[])) != 19:
        raise ValueError("full-19 READY.json check failed")
    with zipfile.ZipFile(a.raw/"missing_index.zip") as zf:
        names=zf.namelist(); expected={f"index/{i}_2018_missing_index.csv" for i in range(1,20)}
        if not expected.issubset(names): raise ValueError("missing per-region timestamp indices")
        for tile in range(1,20):
            rows=list(csv.DictReader(zf.read(f"index/{tile}_2018_missing_index.csv").decode("utf-8").splitlines()))
            if len(rows)!=35040 or rows[0]["utc_time"]!="2018-01-01 00:00:00" or rows[-1]["utc_time"]!="2018-12-31 23:45:00":
                raise ValueError(f"region {tile}: official UTC index incomplete or unexpected")
            if tile==1: times=[r["utc_time"] for r in rows]
            elif [r["utc_time"] for r in rows] != times: raise ValueError("region UTC indices differ")
    ground=np.load(a.prepared/"ground_quarter.npy",mmap_mode="r")
    gc=np.load(a.prepared/"ground_quarter_count.npy",mmap_mode="r")
    if ground.shape != (35040,19) or gc.shape != ground.shape: raise ValueError("full station archive invalid")
    regions=range(1,2) if a.smoke else range(1,20)
    a.output.mkdir(parents=True,exist_ok=False)
    done=[]
    try:
        for tile in regions:
            build_region(a.prepared,a.output/f"region_{tile:02d}",ground,gc,tile)
            done.append(tile)
            print(json.dumps({"event":"region_ready","region":tile}),flush=True)
        if a.smoke: return
        ready_out={"complete":True,"regions":done,"steps_per_region":8760,
                   "input_frame_count_per_region":35040,"split":"chronological full-year 70/15/15; windows confined within split",
                   "history_hours":72,"horizon_hours":24,"field_label":"satellite SSR W m-2",
                   "point_label":"separate ground/in-situ irradiance W m-2","official_index_verified":True}
        (a.output/"READY.json").write_text(json.dumps(ready_out,indent=2),encoding="utf-8")
    except Exception:
        raise

if __name__=="__main__": main()
