#!/usr/bin/env python3
"""CSI-joint SolarCube encoder screen: V3 Vision-Swin, V4 ResNet-ViT,
V5 ResNet-Swin-ViT. All variants share splits, data windows, CSI/GHI heads,
local point-history encoder and optimization; only the raster encoder differs.
These are deterministic encoder screens (no diffusion/cloud-state branch), so
deterministic CRPS equals MAE and prediction intervals are intentionally N/A.
"""
from __future__ import annotations
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_solarcube_v7_fieldpoint_csi import SolarCubeWindows,MultiScaleLocal,clear_sky,masked_mse
from models_registry import RasterEncoder

VARIANTS={"V3":"ENC-VISION-SWIN","V4":"ENC-RVIT","V5":"ENC-RES-SWIN-VIT"}

class EncoderCSI(nn.Module):
    def __init__(self,variant,width=64,heads=4):
        super().__init__()
        if variant not in VARIANTS: raise ValueError(f"Unknown version {variant}")
        self.variant=variant; self.encoder_name=VARIANTS[variant]; self.width=width
        self.encoder=RasterEncoder(channels=15,width=width,heads=heads,name=self.encoder_name,preserve_grid=True)
        self.clock=nn.Sequential(nn.Linear(6,width),nn.GELU(),nn.Linear(width,width))
        tl=nn.TransformerEncoderLayer(width,heads,2*width,.1,batch_first=True,norm_first=True,activation="gelu")
        self.temporal=nn.TransformerEncoder(tl,1)
        self.local=MultiScaleLocal(width)
        self.field_csi=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,1))
        self.point_fuse=nn.Sequential(nn.Linear(3*width,width),nn.GELU(),nn.Linear(width,width))
        self.point_csi=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,1))
    def encode(self,b):
        nw=F.interpolate(b["nwp"].flatten(0,1),size=(128,128),mode="bilinear",align_corners=False).view(*b["nwp"].shape[:2],9,128,128)
        sat=F.interpolate(b["sat"].flatten(0,1),size=(128,128),mode="bilinear",align_corners=False).view(*b["sat"].shape[:2],6,128,128)
        x=torch.cat([sat,nw],2)
        z=self.encoder(x) # B,T,D,8,8; preserve local spatial structure for window attention
        z=z+self.clock(b["time_hist"])[:,:,None,None,:].permute(0,1,4,2,3)
        B,T,D,H,W=z.shape
        tok=z.permute(0,3,4,1,2).reshape(B*H*W,T,D)
        tok=self.temporal(tok)
        latent=(tok[:,-1]+tok.mean(1)).view(B,H,W,D).permute(0,3,1,2)
        # Lift the 2x2 encoder map to the common 16x16 target grid.
        return F.interpolate(latent,size=(128,128),mode="bilinear",align_corners=False).flatten(2).transpose(1,2)
    def forward(self,b):
        grid=self.encode(b); B,N,D=grid.shape; tf=self.clock(b["time_future"])
        future=grid[:,None]+tf[:,:,None,:]
        fcsi=2.5*torch.sigmoid(self.field_csi(future)[...,0])
        local,_=self.local(b["point_hist"],b["point_hist_mask"])
        pos=b["position"]; xy=(2*pos-1).view(B,1,1,2)
        g=grid.transpose(1,2).reshape(B,D,128,128)
        site=F.grid_sample(g,xy,mode="bilinear",padding_mode="border",align_corners=True).reshape(B,D)
        pf=self.point_fuse(torch.cat([site,grid.mean(1),local],-1))[:,None]+tf
        pcsi=2.5*torch.sigmoid(self.point_csi(pf)[...,0])
        return fcsi,pcsi

def metric_rows(pred,y,mask):
    pred=np.asarray(pred); y=np.asarray(y); mask=np.asarray(mask,dtype=bool); rows=[]
    for h in range(y.shape[1]):
        m=mask[:,h]
        if not m.any(): continue
        e=pred[:,h]-y[:,h]
        mae=float(np.abs(e[m]).mean()); rmse=float(np.sqrt(np.square(e[m]).mean()))
        rows.append({"lead_hour":h+1,"n":int(m.sum()),"MAE":mae,"RMSE":rmse,"CRPS":mae,"PI90":"not_applicable_deterministic_encoder_screen"})
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--variant",choices=list(VARIANTS),required=True); ap.add_argument("--data",required=True); ap.add_argument("--prepared",required=True); ap.add_argument("--sitelist",required=True); ap.add_argument("--nwp",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--epochs",type=int,default=60); ap.add_argument("--patience",type=int,default=8); ap.add_argument("--batch",type=int,default=8); ap.add_argument("--stride",type=int,default=3); ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=False); random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    ds={s:SolarCubeWindows(a.data,a.prepared,a.sitelist,a.nwp,s,stride=a.stride) for s in ("train","val","test")}; dl={s:DataLoader(ds[s],batch_size=a.batch,shuffle=s=="train",num_workers=0,pin_memory=True) for s in ds}
    model=EncoderCSI(a.variant).cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5);best=float("inf");bad=0;hist=[];start=time.time();log=(out/"train.log").open("w",buffering=1)
    config={"version":a.variant,"registry_encoder":VARIANTS[a.variant],"dataset":"SolarCube 2018 all 19 regions","split":"chronological 70/15/15","history_hours":72,"horizon_hours":24,"window_stride_hours":a.stride,"inputs":"same satellite history, partial ERA5 history (masked), and past in-situ point history across V3-V5","outputs":"field and selected point CSI; physical GHI = CSI x future Haurwitz clear-sky GHI","loss":"field GHI + point GHI + 0.5*(field CSI + point CSI), masked by labels/daylight","probability":"deterministic encoder screen; CRPS=MAE; prediction intervals not applicable","early_stopping":{"patience":a.patience,"restore_best":True},"batch":a.batch,"seed":a.seed,"max_epochs":a.epochs}
    (out/"config.json").write_text(json.dumps(config,indent=2));(out/"README.md").write_text("# "+a.variant+" SolarCube CSI field/point screen\n\n"+json.dumps(config,indent=2)+"\n\nTest truths/predictions, checkpoint, history and lead figures are saved in this directory.\n")
    log.write(json.dumps({"event":"started","train_windows":len(ds["train"]),"val_windows":len(ds["val"]),"test_windows":len(ds["test"]),"parameters":sum(p.numel() for p in model.parameters()),"device":"cuda","early_stopping":config["early_stopping"]})+"\n")
    def mv(b): return {k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in b.items()}
    for ep in range(1,a.epochs+1):
        model.train();tr=[]
        for b0 in dl["train"]:
            b=mv(b0);opt.zero_grad(set_to_none=True);fc,pc=model(b);fg=fc*b["field_cs"];pg=pc*b["point_cs"]
            fl=masked_mse((fg-b["field_ghi"])/1000,torch.zeros_like(fg),b["field_mask"]);pl=masked_mse((pg-b["point_ghi"])/1000,torch.zeros_like(pg),b["point_mask"])
            cfl=masked_mse(fc,b["field_csi"],b["field_csi_mask"]);cpl=masked_mse(pc,b["point_csi"],b["point_csi_mask"]);loss=fl+pl+.5*(cfl+cpl)
            if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite loss epoch {ep}")
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();tr.append([float(loss.detach()),float(fl.detach()),float(pl.detach()),float(cfl.detach()),float(cpl.detach())])
        model.eval();va=[]
        with torch.no_grad():
            for b0 in dl["val"]:
                b=mv(b0);fc,pc=model(b);fg=fc*b["field_cs"];pg=pc*b["point_cs"]
                va.append(float(masked_mse((fg-b["field_ghi"])/1000,torch.zeros_like(fg),b["field_mask"])+masked_mse((pg-b["point_ghi"])/1000,torch.zeros_like(pg),b["point_mask"])+.5*(masked_mse(fc,b["field_csi"],b["field_csi_mask"])+masked_mse(pc,b["point_csi"],b["point_csi_mask"]))))
        v=float(np.mean(va));row={"epoch":ep,"train_loss_components":np.mean(tr,axis=0).tolist(),"val_joint_loss":v,"elapsed_sec":time.time()-start,"peak_gpu_allocated_MiB":torch.cuda.max_memory_allocated()/2**20};hist.append(row);log.write(json.dumps(row)+"\n")
        if v<best-1e-7:best=v;bad=0;best_ep=ep;torch.save(model.state_dict(),out/"best.pt")
        else:bad+=1
        if ep>=5 and bad>=a.patience:break
    model.load_state_dict(torch.load(out/"best.pt",map_location="cuda",weights_only=True));model.eval();fs=[];ps=[];fy=[];py=[];fm=[];pm=[];fcs=[];pcs=[];fct=[];pct=[];fcm=[];pcm=[];regs=[];orig=[]
    with torch.no_grad():
        for b0 in dl["test"]:
            b=mv(b0);fc,pc=model(b);fg=fc*b["field_cs"];pg=pc*b["point_cs"]
            fs.append(fg.cpu().numpy());ps.append(pg.cpu().numpy());fy.append(b0["field_ghi"].numpy());py.append(b0["point_ghi"].numpy());fm.append(b0["field_mask"].numpy());pm.append(b0["point_mask"].numpy());fcs.append(fc.cpu().numpy());pcs.append(pc.cpu().numpy());fct.append(b0["field_csi"].numpy());pct.append(b0["point_csi"].numpy());fcm.append(b0["field_csi_mask"].numpy());pcm.append(b0["point_csi_mask"].numpy());regs.append(b0["region"].numpy());orig.append(b0["origin"].numpy())
    fs=np.concatenate(fs);ps=np.concatenate(ps);fy=np.concatenate(fy);py=np.concatenate(py);fm=np.concatenate(fm);pm=np.concatenate(pm);fcs=np.concatenate(fcs);pcs=np.concatenate(pcs);fct=np.concatenate(fct);pct=np.concatenate(pct);fcm=np.concatenate(fcm);pcm=np.concatenate(pcm)
    result={"version":a.variant,"encoder":VARIANTS[a.variant],"best_epoch":best_ep,"epochs_ran":len(hist),"train_seconds":time.time()-start,"test_windows":len(fs),"parameters":sum(p.numel() for p in model.parameters()),"peak_gpu_allocated_MiB":torch.cuda.max_memory_allocated()/2**20,"field_GHI_by_lead":metric_rows(fs,fy,fm),"point_GHI_by_lead":metric_rows(ps,py,pm),"field_CSI_by_lead":metric_rows(fcs,fct,fcm),"point_CSI_by_lead":metric_rows(pcs,pct,pcm)}
    np.savez_compressed(out/"test_predictions.npz",field_ghi_prediction=fs,field_ghi_truth=fy,field_mask=fm,point_ghi_prediction=ps,point_ghi_truth=py,point_mask=pm,field_csi_prediction=fcs,field_csi_truth=fct,field_csi_mask=fcm,point_csi_prediction=pcs,point_csi_truth=pct,point_csi_mask=pcm,region=np.concatenate(regs),origin=np.concatenate(orig))
    (out/"metrics.json").write_text(json.dumps(result,indent=2));(out/"training_history.json").write_text(json.dumps(hist,indent=2));log.write(json.dumps({"event":"finished","result":result})+"\n");log.close()

if __name__=="__main__":main()
