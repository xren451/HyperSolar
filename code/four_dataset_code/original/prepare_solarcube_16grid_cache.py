#!/usr/bin/env python3
"""Precompute 16x16 hourly SolarCube inputs/targets once for all windows."""
import json
from pathlib import Path
import numpy as np

def reduce_features(x, block=4):
    # Input [T,64,64,C], output [T,16,16,C]; ignore per-channel missing values.
    t,_,_,c=x.shape; z=x.reshape(t,16,block,16,block,c).astype(np.float32)
    valid=np.isfinite(z); count=valid.sum((2,4)); total=np.where(valid,z,0).sum((2,4))
    return np.divide(total,count,out=np.full_like(total,np.nan),where=count>0)

def reduce_target(y,m,block=4):
    t,_,_=y.shape; yy=y.reshape(t,16,block,16,block).astype(np.float32); mm=m.reshape(t,16,block,16,block).astype(bool)
    count=mm.sum((2,4)); total=np.where(mm,yy,0).sum((2,4)); mean=np.divide(total,count,out=np.zeros_like(total),where=count>0)
    # Preserve sparse land-only coverage (notably SolarCube region 11): every
    # 16x16 label is an average of its actually observed pixels, and the mask
    # remains false only where all sixteen native pixels are invalid.
    return mean,count>=1

def main(root:Path):
    ready=root/"READY.json"
    meta=json.loads(ready.read_text())
    if not meta.get("complete") or len(meta.get("regions",[]))!=19: raise ValueError("full19 cache not ready")
    rows=[]
    for r in range(1,20):
        d=root/f"region_{r:02d}"
        outfiles=[d/"features16.npy",d/"field_ssr16.npy",d/"field_mask16.npy"]
        exists=[p.exists() for p in outfiles]
        if any(exists) and not all(exists): raise FileExistsError(f"partial cache in {d}")
        x=np.load(d/"features.npy",mmap_mode="r"); y=np.load(d/"field_ssr.npy",mmap_mode="r"); m=np.load(d/"field_mask.npy",mmap_mode="r")
        mode="r+" if all(exists) else "w+"
        xf=np.lib.format.open_memmap(outfiles[0],mode=mode,dtype=np.float16,shape=(8760,16,16,6))
        yf=np.lib.format.open_memmap(outfiles[1],mode=mode,dtype=np.float16,shape=(8760,16,16))
        mf=np.lib.format.open_memmap(outfiles[2],mode=mode,dtype=np.bool_,shape=(8760,16,16))
        for t0 in range(0,8760,128):
            t1=min(8760,t0+128)
            xf[t0:t1]=reduce_features(np.asarray(x[t0:t1],np.float32)).astype(np.float16)
            yy,mm=reduce_target(np.asarray(y[t0:t1],np.float32),np.asarray(m[t0:t1],bool))
            yf[t0:t1]=np.where(mm,yy,0).astype(np.float16);mf[t0:t1]=mm
        xf.flush();yf.flush();mf.flush()
        row={"region":r,"grid_shape":[16,16],"valid_field_fraction":float(mf.mean()),"features16_bytes":int(outfiles[0].stat().st_size)}
        rows.append(row);print(json.dumps(row),flush=True)
    (root/"READY16.json").write_text(json.dumps({"complete":True,"regions":rows,"aggregation":"64x64 hourly values area-aggregated 4x4 to 16x16 once; each coarse label averages all valid native pixels, and mask is true when at least one native pixel is valid"},indent=2))

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();main(Path(a.root))
