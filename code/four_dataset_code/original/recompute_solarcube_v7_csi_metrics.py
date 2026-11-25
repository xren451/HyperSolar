#!/usr/bin/env python3
"""Post-test CSI metric mask audit for the V7 SolarCube archive."""
import argparse,json
from pathlib import Path
import numpy as np

def cs(z):
    c=np.maximum(np.cos(np.deg2rad(z)),0); return 1098*c*np.exp(-0.059/np.maximum(c,1e-4))

def metric_rows(ens,y,m):
    rows=[]; mu=ens.mean(1)
    for h in range(y.shape[1]):
        mm=m[:,h].astype(bool)
        if not mm.any(): continue
        e=mu[:,h]-y[:,h]; x=ens[:,:,h]; crps=np.abs(x-y[:,None,h]).mean(1)-.5*np.abs(x[:,None]-x[:,:,None]).mean((1,2))
        lo,hi=np.quantile(x,[.05,.95],axis=1)
        rows.append({
            "lead_hour":h+1,"n":int(mm.sum()),
            "MAE":float(np.abs(e[mm]).mean()),
            "RMSE":float(np.sqrt(np.mean(e[mm]**2))),
            "CRPS":float(crps[mm].mean()),
            "PI90_coverage":float(((y[:,h]>=lo)&(y[:,h]<=hi))[mm].mean()),
            "PI90_width":float((hi-lo)[mm].mean()),
            "spread_skill":float(x.std(1)[mm].mean()/max(np.sqrt(np.mean(e[mm]**2)),1e-8))})
    return rows

def interp_sza(x,cf,rf):
    h,w=x.shape; xx=np.clip(cf*(w-1),0,w-1); yy=np.clip(rf*(h-1),0,h-1); x0=int(np.floor(xx));x1=min(x0+1,w-1);y0=int(np.floor(yy));y1=min(y0+1,h-1); dx=xx-x0;dy=yy-y0
    return (1-dy)*((1-dx)*x[y0,x0]+dx*x[y0,x1])+dy*((1-dx)*x[y1,x0]+dx*x[y1,x1])

def main():
    p=argparse.ArgumentParser();p.add_argument("--result",required=True);p.add_argument("--data",required=True);p.add_argument("--sitelist",required=True);a=p.parse_args()
    out=Path(a.result); droot=Path(a.data); z=np.load(out/"test_predictions.npz",mmap_mode="r")
    reg=np.asarray(z["region"],int); origin=np.asarray(z["origin"],int); n=len(origin); lead=np.arange(1,25)
    import csv
    with open(a.sitelist,newline="",encoding="utf-8-sig") as f: sites={int(q["tile_id"]):q for q in csv.DictReader(f)}
    fcm=np.zeros((n,24,256),bool);pcm=np.zeros((n,24),bool)
    for r in np.unique(reg):
        ix=np.flatnonzero(reg==r); feat=np.load(droot/f"region_{r:02d}"/"features16.npy",mmap_mode="r"); fm=np.load(droot/f"region_{r:02d}"/"field_mask16.npy",mmap_mode="r"); pm=np.load(droot/f"region_{r:02d}"/"point_mask.npy",mmap_mode="r")
        site=sites[int(r)]; rf=(float(site["lat_ulcnr"])-float(site["lats"]))/(float(site["lat_ulcnr"])-float(site["lat_lrcnr"])); cf=(float(site["lons"])-float(site["lon_ulcnr"]))/(float(site["lon_lrcnr"])-float(site["lon_ulcnr"]))
        for start in range(0,len(ix),256):
            ids=ix[start:start+256]; ti=origin[ids,None]+lead[None,:]; sza=np.asarray(feat[ti,:,:,0],np.float32); finite=np.isfinite(sza); c=cs(np.where(finite,sza,180)); fvalid=np.asarray(fm[ti],bool)&(c>20)&finite
            fcm[ids]=fvalid.reshape(len(ids),24,256)
            point_sza=np.stack([[interp_sza(sza[i,j],cf,rf) for j in range(24)] for i in range(len(ids))]); pcs=cs(point_sza);pcm[ids]=np.asarray(pm[ti],bool)&(pcs>20)&np.isfinite(point_sza)
    np.savez_compressed(out/"csi_valid_masks_posthoc.npz",field_csi_mask=fcm,point_csi_mask=pcm)
    res={"field_CSI_by_lead":metric_rows(np.asarray(z["field_csi_ensemble"]),np.asarray(z["field_csi_truth"]),fcm),"point_CSI_by_lead":metric_rows(np.asarray(z["point_csi_ensemble"]),np.asarray(z["point_csi_truth"]),pcm),"validity":"field/point label mask AND Haurwitz clear-sky >20 W m-2; reconstructed from archived region/hour/SZA after test evaluation"}
    (out/"metrics_csi_daylight_corrected.json").write_text(json.dumps(res,indent=2))
    print(json.dumps({"samples":n,"field_valid":int(fcm.sum()),"point_valid":int(pcm.sum()),"output":str(out/"metrics_csi_daylight_corrected.json")}))

if __name__=="__main__":main()
