#!/usr/bin/env python3
"""Full-19 SolarCube irradiance field + in-situ selected-point V2 run.

This adapter uses all hourly frames and regions. To keep spatial attention
tractable, native 64x64 fields are area-aggregated to a 16x16 grid; no region
or date subset is taken. The point head is supervised from in-situ station
GHI, independently of the satellite SSR field label.
"""
from __future__ import annotations
import argparse, csv, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from models_field_point import Version2FieldPoint
from models import FlatResidualDDPM

class SolarCubeWindows(Dataset):
    def __init__(self, root: Path, prepared: Path, sitelist: Path, split: str, hist=72, horizon=24):
        self.root=root; self.prepared=prepared; self.hist=hist; self.horizon=horizon
        self.n=8760; cuts=[0,int(self.n*.70),int(self.n*.85),self.n]
        cache16=json.loads((root/"READY16.json").read_text())
        if not cache16.get("complete") or len(cache16.get("regions",[]))!=19:
            raise ValueError("Full 19-region 16x16 cache is not complete")
        a,b={"train":(cuts[0],cuts[1]),"val":(cuts[1],cuts[2]),"test":(cuts[2],cuts[3])}[split]
        origins=np.arange(max(a+hist-1, hist-1), b-horizon, dtype=np.int32)
        self.rows=[]; self.arrays=[]; self.target_stats=[]; self.point_stats=[]; self.point_positions=[]
        prep=json.loads((prepared/"READY.json").read_text())
        self.source_regions=prep["regions"]
        with sitelist.open(newline="",encoding="utf-8-sig") as sf:
            sites={int(row["tile_id"]):row for row in csv.DictReader(sf)}
        for region in range(1,20):
            d=root/f"region_{region:02d}"
            feat=np.load(d/"features16.npy",mmap_mode="r")
            field=np.load(d/"field_ssr16.npy",mmap_mode="r")
            fm=np.load(d/"field_mask16.npy",mmap_mode="r")
            point=np.load(d/"point_insitu.npy",mmap_mode="r")
            pm=np.load(d/"point_mask.npy",mmap_mode="r")
            info=json.loads((d/"metadata.json").read_text())
            tstats=self.source_regions[region-1]["train_only_stats"]
            self.arrays.append((feat,field,fm,point,pm))
            self.target_stats.append((float(tstats[3]["mean"]),max(float(tstats[3]["std"]),1e-5)))
            valid_train=np.asarray(pm[:cuts[1]],bool)
            point_train=np.asarray(point[:cuts[1]],np.float32)[valid_train]
            self.point_stats.append((float(point_train.mean()),max(float(point_train.std()),1e-5)))
            site=sites[region]
            row_frac=(float(site["lat_ulcnr"])-float(site["lats"]))/(float(site["lat_ulcnr"])-float(site["lat_lrcnr"]))
            col_frac=(float(site["lons"])-float(site["lon_ulcnr"]))/(float(site["lon_lrcnr"])-float(site["lon_ulcnr"]))
            if not (0<=row_frac<=1 and 0<=col_frac<=1): raise ValueError(f"station outside SolarCube tile {region}")
            # grid_sample consumes coordinates in (x=column,y=row) order.
            self.point_positions.append((col_frac,row_frac))
            valid_at=np.asarray(fm).reshape(self.n,-1).any(1) | np.asarray(pm,bool)
            prefix=np.concatenate([[0],np.cumsum(valid_at,dtype=np.int64)])
            has_future=(prefix[origins+horizon+1]-prefix[origins+1])>0
            self.rows.extend((region-1,int(t)) for t in origins[has_future])
        if not self.rows: raise ValueError(f"No SolarCube windows for {split}")
        self.split=split; self.cuts=cuts

    def __len__(self): return len(self.rows)

    def __getitem__(self,i):
        r,t=self.rows[i]; feat,field,fm,point,pm=self.arrays[r]
        lo=t-self.hist+1; hi=t+1; fut=np.arange(t+1,t+1+self.horizon)
        x=np.asarray(feat[lo:hi],dtype=np.float32)
        # source train-only SolarCube normalization, then stabilize missing channels.
        stats=self.source_regions[r]["train_only_stats"]
        means=np.asarray([s["mean"] for s in stats],np.float32)
        stds=np.maximum(np.asarray([s["std"] for s in stats],np.float32),1e-5)
        x=np.where(np.isfinite(x),(x-means)/stds,0).astype(np.float32)
        fy=np.asarray(field[fut],np.float32); fm2=np.asarray(fm[fut],bool)
        mu,sd=self.target_stats[r]; fy=np.where(fm2,(fy-mu)/sd,0).astype(np.float32)
        # Cloud-mask fraction is the supervised intermediate state; it is derived
        # from the hourly CM feature, with its own finite-value mask.
        cm=np.asarray(feat[fut,:,:,1],dtype=np.float32)
        cm2=cm
        sm=np.isfinite(cm2); state=np.where(sm,cm2,0).astype(np.float32)
        py=np.asarray(point[fut],np.float32); pvalid=np.asarray(pm[fut],bool)
        pmean,psd=self.point_stats[r]; py=np.where(pvalid,(py-pmean)/psd,0).astype(np.float32)
        # normalized local coordinates for 16x16 tokens
        coord=np.stack(np.meshgrid(np.linspace(-1,1,16),np.linspace(-1,1,16),indexing="ij"),-1).reshape(-1,2).astype(np.float32)
        times=np.arange(lo,hi); ft=fut
        def tf(idx):
            day=idx//24; hour=idx%24; doy=day%365
            return np.stack([np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),
                             np.sin(2*np.pi*doy/365),np.cos(2*np.pi*doy/365),
                             hour/23,doy/364],-1).astype(np.float32)
        return {"x":torch.from_numpy(x.reshape(self.hist,256,6)),
                "time_hist":torch.from_numpy(tf(times)),"time_future":torch.from_numpy(tf(ft)),
                "y":torch.from_numpy(fy.reshape(self.horizon,256)),
                "target_mask":torch.from_numpy(fm2.reshape(self.horizon,256)),
                "state":torch.from_numpy(state.reshape(self.horizon,256)),
                "state_mask":torch.from_numpy(sm.reshape(self.horizon,256)),
                "point_y":torch.from_numpy(py),"point_mask":torch.from_numpy(pvalid),
                "field_mean":torch.tensor(mu,dtype=torch.float32),"field_std":torch.tensor(sd,dtype=torch.float32),
                "point_mean":torch.tensor(pmean,dtype=torch.float32),"point_std":torch.tensor(psd,dtype=torch.float32),
                "point_position":torch.tensor(self.point_positions[r],dtype=torch.float32),
                "region":torch.tensor(r+1),"origin":torch.tensor(t)}

class SolarCubeV2(Version2FieldPoint):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.station_point= torch.nn.Sequential(torch.nn.Linear(4,self.det.width),torch.nn.GELU(),
                   torch.nn.Linear(self.det.width,self.det.width//2),torch.nn.GELU(),torch.nn.Linear(self.det.width//2,1))
        self.diff=FlatResidualDDPM(self.horizon*self.sites,width=128,steps=8)
    def _point(self,state,z,b):
        # Bilinear readout at each tile's actual station coordinate, derived
        # from station lat/lon and tile bounds rather than assuming tile center.
        pos=b["point_position"]; grid=(2*pos-1).view(-1,1,1,2)
        bs,leads,nodes=state.shape
        sm=state.reshape(bs*leads,1,16,16)
        sg=grid[:,None].expand(-1,leads,-1,-1,-1).reshape(bs*leads,1,1,2)
        s=torch.nn.functional.grid_sample(sm,sg,mode="bilinear",padding_mode="border",align_corners=True).reshape(bs,leads)
        zm=z.transpose(1,2).reshape(bs,z.shape[-1],16,16)
        z0=torch.nn.functional.grid_sample(zm,grid,mode="bilinear",padding_mode="border",align_corners=True).reshape(bs,z.shape[-1]).mean(-1)[:,None].expand_as(s)
        g=state.mean(-1); zg=z.mean((1,2))[:,None].expand_as(s)
        return self.station_point(torch.stack([s,g,z0,zg],-1))[...,0]
    def loss(self,b):
        _,state,z=self.deterministic(b); field=self.decode_field(state,z,b)
        tm=b["target_mask"].bool(); pm=b["point_mask"].bool(); sm=b["state_mask"].bool()
        fl=(field-b["y"]).square()[tm].mean() if tm.any() else field.sum()*0
        point=self._point(state,z,b); pl=(point-b["point_y"]).square()[pm].mean() if pm.any() else field.sum()*0
        sl=(state-b["state"]).square()[sm].mean() if sm.any() else field.sum()*0
        dl=self.diff.loss(b["state"]-state.detach(),sm) if sm.any() else field.sum()*0
        return fl+pl+sl+.05*dl,{"field":fl,"point":pl,"state":sl,"diffusion":dl}
    @torch.no_grad()
    def forecast(self,b,members=4):
        _,state,z=self.deterministic(b); resid=self.diff.sample(len(state),state.device,state.shape,members)
        states=state[:,None]+resid
        fs=[]; ps=[]
        for k in range(members):
            sk=states[:,k]; fs.append(self.decode_field(sk,z,b)); ps.append(self._point(sk,z,b))
        fm=self.decode_field(state,z,b); pm=self._point(state,z,b)
        return torch.stack(fs,1),torch.stack(ps,1),fm,pm

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",required=True); p.add_argument("--prepared",required=True); p.add_argument("--sitelist",required=True); p.add_argument("--out",required=True)
    p.add_argument("--epochs",type=int,default=60); p.add_argument("--patience",type=int,default=8); p.add_argument("--batch",type=int,default=8); p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    ds={s:SolarCubeWindows(Path(a.data),Path(a.prepared),Path(a.sitelist),s) for s in ("train","val","test")}
    dl={s:DataLoader(ds[s],batch_size=a.batch,shuffle=s=="train",num_workers=0,pin_memory=True) for s in ds}
    coords=np.stack(np.meshgrid(np.linspace(-1,1,16),np.linspace(-1,1,16),indexing="ij"),-1).reshape(-1,2).astype(np.float32)
    model=SolarCubeV2(6,256,coords,selected_site=136,width=64,heads=4,ablation="full").cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
    config={"dataset":"SolarCube","scope":"all 19 regions, all hourly 2018 records; native 64x64 labels area-aggregated to 16x16 field grid once during cache preparation",
      "history_hours":72,"horizon_hours":24,"field_target":"satellite SSR W m-2","point_target":"separate collocated in-situ irradiance W m-2",
      "state_target":"future CM cloud-mask fraction","station_mapping":"per-region station latitude/longitude transformed using tile bounds; bilinear readout on 16x16 grid; no center assumption",
      "station_list":str(Path(a.sitelist).resolve()),"split":"chronological 70/15/15, windows confined within split","early_stopping":{"patience":a.patience,"restore_best":True},
      "optimizer":"AdamW lr=1e-4 wd=1e-5","batch":a.batch,"seed":a.seed,"diffusion_steps":8,"source":"official SolarCube UTC index and prepared full19 arrays"}
    (out/"config.json").write_text(json.dumps(config,indent=2)); (out/"README.md").write_text(
      "# SolarCube full19 V2 field + point irradiance\n\n"+json.dumps(config,indent=2)+"\n\nCode: train_solarcube_v2_fieldpoint.py, models_field_point.py, models.py. Logs, best checkpoint, history, test truths/predictions, metrics, and lead figures are written here.\n")
    log=(out/"train.log").open("w"); start=time.time(); best=float("inf"); bad=0; hist=[]
    log.write(json.dumps({"event":"started","train_windows":len(ds["train"]),"val_windows":len(ds["val"]),"test_windows":len(ds["test"]),"device":"cuda"})+"\n"); log.flush()
    for ep in range(1,a.epochs+1):
        model.train(); tr=[]
        for b in dl["train"]:
            b={k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in b.items()}; opt.zero_grad(set_to_none=True); loss,_=model.loss(b); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tr.append(float(loss.detach()))
        model.eval(); va=[]
        with torch.no_grad():
            for b in dl["val"]:
                b={k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in b.items()}; va.append(float(model.loss(b)[0]))
        row={"epoch":ep,"train_loss":float(np.mean(tr)),"val_loss":float(np.mean(va)),"elapsed_sec":time.time()-start}; hist.append(row); log.write(json.dumps(row)+"\n");log.flush()
        if row["val_loss"]<best-1e-7: best=row["val_loss"];bad=0;torch.save(model.state_dict(),out/"best.pt")
        else: bad+=1
        if ep>=5 and bad>=a.patience: break
    model.load_state_dict(torch.load(out/"best.pt",map_location="cuda",weights_only=True)); model.eval()
    fields=[]; points=[]; fy=[]; fm=[]; py=[]; pm=[]; regs=[]; origins=[]
    with torch.no_grad():
      for b0 in dl["test"]:
        b={k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in b0.items()}; f,p0,_,_=model.forecast(b,4)
        fs=f.cpu().numpy(); ps0=p0.cpu().numpy(); fm0=b0["field_mean"].numpy(); fstd=b0["field_std"].numpy(); pm0=b0["point_mean"].numpy(); pstd=b0["point_std"].numpy()
        fields.append(fs*fstd[:,None,None,None]+fm0[:,None,None,None]); points.append(ps0*pstd[:,None,None]+pm0[:,None,None])
        fy.append(b0["y"].numpy()*fstd[:,None,None]+fm0[:,None,None]);fm.append(b0["target_mask"].numpy());py.append(b0["point_y"].numpy()*pstd[:,None]+pm0[:,None]);pm.append(b0["point_mask"].numpy());regs.append(b0["region"].numpy());origins.append(b0["origin"].numpy())
    fs=np.concatenate(fields); ps0=np.concatenate(points); fyt=np.concatenate(fy); fmt=np.concatenate(fm).astype(bool); pyt=np.concatenate(py); pmt=np.concatenate(pm).astype(bool)
    def metric_rows(pred,truth,mask,scope):
        rows=[]
        for h in range(truth.shape[1]):
            m=mask[:,h]
            if not m.any(): continue
            ens=pred[:,:,h]; obs=truth[:,h]; err=ens.mean(1)-obs; spread=ens.std(1)
            crps=np.abs(ens-obs[:,None]).mean(1)-.5*np.abs(ens[:,None]-ens[:,:,None]).mean((1,2))
            lo,hi=np.quantile(ens,[.05,.95],axis=1)
            rows.append({"lead_hour":h+1,"n":int(m.sum()),"MAE":float(np.abs(err[m]).mean()),"RMSE":float(np.sqrt((err[m]**2).mean())),"CRPS":float(crps[m].mean()),"PI90_coverage":float(((obs>=lo)&(obs<=hi))[m].mean()),"PI90_width":float((hi-lo)[m].mean()),"spread_skill_ratio":float(spread[m].mean()/max(np.sqrt((err[m]**2).mean()),1e-8))})
        return rows
    result={"dataset":"SolarCube","version":"SMF-V2-raster-adaptation","best_epoch":int(np.argmin([r["val_loss"] for r in hist])+1),"epochs_ran":len(hist),"train_seconds":time.time()-start,"peak_gpu_allocated_MiB":torch.cuda.max_memory_allocated()/1024**2,
      "field_spatial_scope":"all 19 complete regions; full native 64x64 label field is area-aggregated to 16x16 forecast nodes","field_target":"satellite SSR W m-2","point_target":"separate collocated in-situ irradiance W m-2","field_metrics_by_lead":metric_rows(fs,fyt,fmt,"field"),"point_metrics_by_lead":metric_rows(ps0[:,:, :,None],pyt[:,:,None],pmt[:,:,None],"point")}
    origin_all=np.concatenate(origins); timestamps=np.datetime64("2018-01-01T00:00:00","h")+origin_all[:,None].astype("timedelta64[h]")+np.arange(1,25,dtype="timedelta64[h]")
    np.savez_compressed(out/"test_predictions.npz",field_ensemble=fs,field_truth=fyt,field_mask=fmt,point_ensemble=ps0,point_truth=pyt,point_mask=pmt,region=np.concatenate(regs),origin=origin_all,timestamps_utc=timestamps)
    try:
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,4,figsize=(14,4))
        for ax,h in zip(axs,[0,5,11,23]):
            obs=np.where(fmt[0,h].reshape(16,16),fyt[0,h].reshape(16,16),np.nan)
            fc=fs[0,:,h].mean(0).reshape(16,16)
            im=ax.imshow(obs,cmap="viridis");ax.contour(fc,colors="white",linewidths=.6);ax.set_title(f"lead {h+1} h");fig.colorbar(im,ax=ax,fraction=.046)
        fig.tight_layout();fig.savefig(out/"lead_field_example.png",dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,4)); ax.plot(pyt[0],label="in-situ truth",color="black");ax.plot(ps0[0].mean(0),label="forecast",color="#1976d2");ax.fill_between(np.arange(24),np.quantile(ps0[0],.05,axis=0),np.quantile(ps0[0],.95,axis=0),alpha=.2);ax.set_xlabel("Lead hour");ax.set_ylabel("Irradiance (W m$^{-2}$)");ax.legend();fig.tight_layout();fig.savefig(out/"lead_point_example.png",dpi=180);plt.close(fig)
    except Exception as exc:
        (out/"PLOT_ERROR.txt").write_text(repr(exc))
    (out/"test_metrics.json").write_text(json.dumps(result,indent=2)); (out/"training_history.json").write_text(json.dumps(hist,indent=2)); log.write(json.dumps({"event":"finished","result":result})+"\n");log.close()

if __name__=="__main__": main()
