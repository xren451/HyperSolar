#!/usr/bin/env python3
"""Unpack/regrid aligned monthly ERA5 snapshots for SolarCube V7.

The source files are ZIP archives named .nc; only instantaneous ERA5 fields
are used. ERA5 history is an input, never future reanalysis. Missing regions
and missing months remain explicitly masked.
"""
import argparse, json, zipfile, tempfile, os
from pathlib import Path
import numpy as np
import netCDF4
import torch
import torch.nn.functional as F

VARS = ["t2m", "d2m", "sp", "u10", "v10", "u100", "v100", "tcc"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--regions", type=int, default=19); ap.add_argument("--year", type=int, default=2018)
    a=ap.parse_args(); src=Path(a.source); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    report={"source":str(src),"year":a.year,"variables":VARS,"target_grid":[8,8],"regions":[],"method":"monthly ERA5 UTC instantaneous fields, bilinear resized to 8x8; missing months/regions zero-filled with valid mask 0"}
    for r in range(1,a.regions+1):
        rd=src/f"SolarCube_region_{r:02d}"; missing=[]
        # Keep float32: surface pressure exceeds the finite float16 range.
        raw=np.zeros((8760,8,8,len(VARS)),np.float32); vm=np.zeros(8760,bool)
        for month in range(1,13):
            f=rd/f"era5_{a.year}-{month:02d}.nc"
            if not f.exists(): missing.append(month); continue
            with zipfile.ZipFile(f) as zf:
                instant=next(n for n in zf.namelist() if "instant" in n)
                with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
                    tmp.write(zf.read(instant)); tmp.flush()
                    with netCDF4.Dataset(tmp.name) as ds:
                        ntime=len(ds.dimensions["valid_time"]); arrays=[]
                        for v in VARS:
                            x=np.asarray(ds.variables[v][:],dtype=np.float32); x=np.where(np.isfinite(x),x,0)
                            arrays.append(F.interpolate(torch.from_numpy(x[:,None]),size=(8,8),mode="bilinear",align_corners=False)[:,0].numpy())
                        grid=np.stack(arrays,-1).astype(np.float32)
                        tv=ds.variables["valid_time"]
                        dates=netCDF4.num2date(tv[:],tv.units,calendar=getattr(tv,"calendar","standard"),
                                               only_use_cftime_datetimes=False)
                        vt=np.asarray([np.datetime64(x.replace(tzinfo=None),"s") for x in dates])
                        hours=((vt-np.datetime64(f"{a.year}-01-01T00:00:00"))/np.timedelta64(1,"h")).astype(int)
                        ok=(hours>=0)&(hours<8760); raw[hours[ok]]=grid[ok]; vm[hours[ok]]=True
        dest=out/f"region_{r:02d}"; dest.mkdir(parents=True,exist_ok=True)
        np.save(dest/"era5_8x8.npy",raw); np.save(dest/"valid.npy",vm)
        report["regions"].append({"region":r,"valid_hours":int(vm.sum()),"missing_months":missing,"file":str(dest)})
        print(json.dumps(report["regions"][-1]),flush=True)
    (out/"READY.json").write_text(json.dumps(report,indent=2))

if __name__=="__main__": main()
