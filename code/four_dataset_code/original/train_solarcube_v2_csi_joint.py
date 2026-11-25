#!/usr/bin/env python3
"""CSI-supervised SMF-V2 field + selected in-situ point SolarCube run.

This is a new CSI-joint run, kept separate from the already-running direct-GHI
V2 control. It reuses the verified full19 SolarCube windows/NWP cache contract,
adds separate bounded field/point CSI heads and keeps V2's cloud-state and
state-residual diffusion path. Both physical irradiance and CSI are evaluated.
"""
from __future__ import annotations
import argparse,csv,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_solarcube_v7_fieldpoint_csi import SolarCubeWindows,masked_mse,metrics
from train_solarcube_v2_fieldpoint import SolarCubeV2

class V2CSI(nn.Module):
    def __init__(self):
        super().__init__(); self.base=SolarCubeV2(6,256,np.stack(np.meshgrid(np.linspace(-1,1,16),np.linspace(-1,1,16),indexing="ij"),-1).reshape(-1,2).astype(np.float32),selected_site=136,width=64,heads=4,ablation="full")
        d=self.base.det.width
        self.fcsi=nn.Sequential(nn.Linear(d+1,d),nn.GELU(),nn.Linear(d,1))
        self.pcsi=nn.Sequential(nn.Linear(3*d+1,d),nn.GELU(),nn.Linear(d,1))
    def csi(self,z,state,b):
        tf=self.base.det.clock(b["time_future"])+self.base.det.lead.weight[None]
        ctx=z[:,None]+tf[:,:,None,:]
        f=2.5*torch.sigmoid(self.fcsi(torch.cat([ctx,state[...,None]],-1))[...,0])
        pos=b["position"]; grid=(2*pos-1).view(-1,1,1,2); B,L,N=state.shape
        zs=z.transpose(1,2).reshape(B,z.shape[-1],16,16); zsite=F.grid_sample(zs,grid,mode="bilinear",padding_mode="border",align_corners=True).reshape(B,-1)
        st=F.grid_sample(state.reshape(B*L,1,16,16),grid[:,None].expand(-1,L,-1,-1,-1).reshape(B*L,1,1,2),mode="bilinear",padding_mode="border",align_corners=True).reshape(B,L)
        pctx=torch.cat([zsite[:,None].expand(-1,L,-1),z.mean(1)[:,None].expand(-1,L,-1),tf,st[...,None]],-1)
        p=2.5*torch.sigmoid(self.pcsi(pctx)[...,0]); return f,p
    def forward_train(self,b):
        _,state,z=self.base.deterministic(b); f=self.base.decode_field(state,z,b); p=self.base._point(state,z,b); cf,cp=self.csi(z,state,b); return f,p,cf,cp,state

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",required=True); ap.add_argument("--prepared",required=True); ap.add_argument("--sitelist",required=True); ap.add_argument("--nwp",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--epochs",type=int,default=60); ap.add_argument("--patience",type=int,default=8); ap.add_argument("--batch",type=int,default=4); ap.add_argument("--stride",type=int,default=3); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--members",type=int,default=8)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=False); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    dev=torch.device("cuda"); ds={s:SolarCubeWindows(a.data,a.prepared,a.sitelist,a.nwp,s,stride=a.stride) for s in ("train","val","test")}; dl={s:DataLoader(ds[s],batch_size=a.batch,shuffle=s=="train",num_workers=0,pin_memory=True) for s in ds}
    model=V2CSI().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5); best=float("inf");bad=0; hist=[]; start=time.time(); log=(out/"train.log").open("w",buffering=1)
    cfg={"version":"SMF-V2-CSI-joint","dataset":"SolarCube full19 2018","history_hours":72,"horizon_hours":24,"stride_hours":a.stride,"targets":"field satellite SSR + separate in-situ point GHI, each with direct irradiance and bounded CSI head","CSI":"Haurwitz clear-sky denominator from future known SZA; valid only where clear sky >20 W/m2","state":"future CM cloud mask fraction","diffusion":"V2 supervised cloud-state residual DDPM, 8 reverse steps","early_stopping":{"patience":a.patience,"restore_best":True},"batch":a.batch,"seed":a.seed,"source_code":"train_solarcube_v2_csi_joint.py"}
    (out/"config.json").write_text(json.dumps(cfg,indent=2)); (out/"README.md").write_text("# SMF-V2 with joint field/point CSI supervision\n\n"+json.dumps(cfg,indent=2)+"\n\nTest truth/prediction arrays, logs, best checkpoint, per-lead MAE/RMSE/CRPS, CSI scores, and figures are saved here.\n")
    log.write(json.dumps({"event":"started","train_windows":len(ds["train"]),"val_windows":len(ds["val"]),"test_windows":len(ds["test"]),"parameters":sum(p.numel() for p in model.parameters()),"device":"cuda","early_stopping":cfg["early_stopping"]})+"\n")
    def mv(b): return {k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in b.items()}
    for ep in range(1,a.epochs+1):
        model.train(); tr=[]
        for b0 in dl["train"]:
            b=mv(b0); b["x"]=b["sat"].permute(0,1,3,4,2).reshape(len(b["sat"]),72,256,6); b["point_position"]=b["position"]
            # V2 target/state keys. The model learns standardized GHI; CSI heads
            # are trained against physically reconstructed clear-sky ratios.
            y=(b["field_ghi"]-b["field_mean"][:,None,None])/b["field_std"][:,None,None]
            py=(b["point_ghi"]-b["point_mean"][:,None])/b["point_std"][:,None]
            state_mask=b["state_mask"].bool(); target_mask=b["field_mask"].bool(); point_mask=b["point_mask"].bool()
            opt.zero_grad(set_to_none=True); f,p,cf,cp,s=model.forward_train(b)
            fl=masked_mse(f,y,target_mask); pl=masked_mse(p,py,point_mask); cfl=masked_mse(cf,b["field_csi"],b["field_csi_mask"]); cpl=masked_mse(cp,b["point_csi"],b["point_csi_mask"])
            sl=masked_mse(s,b["state"],state_mask); diff_loss=model.base.diff.loss(b["state"]-s.detach(),state_mask)
            loss=fl+pl+.5*(cfl+cpl)+sl+.05*diff_loss
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss epoch {ep}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); tr.append([float(loss.detach()),float(fl.detach()),float(pl.detach()),float(cfl.detach()),float(cpl.detach()),float(sl.detach()),float(diff_loss.detach())])
        model.eval(); va=[]
        with torch.no_grad():
            for b0 in dl["val"]:
                b=mv(b0); b["x"]=b["sat"].permute(0,1,3,4,2).reshape(len(b["sat"]),72,256,6); b["point_position"]=b["position"]
                y=(b["field_ghi"]-b["field_mean"][:,None,None])/b["field_std"][:,None,None]; py=(b["point_ghi"]-b["point_mean"][:,None])/b["point_std"][:,None]
                f,p,cf,cp,s=model.forward_train(b); v=masked_mse(f,y,b["field_mask"])+masked_mse(p,py,b["point_mask"])+.5*(masked_mse(cf,b["field_csi"],b["field_csi_mask"])+masked_mse(cp,b["point_csi"],b["point_csi_mask"]))
                va.append(float(v))
        v=float(np.mean(va)); row={"epoch":ep,"train_losses":np.mean(tr,axis=0).tolist(),"val_joint_loss":v,"elapsed_sec":time.time()-start,"peak_gpu_allocated_MiB":torch.cuda.max_memory_allocated()/2**20};hist.append(row);log.write(json.dumps(row)+"\n")
        if v<best-1e-7: best=v;bad=0;best_ep=ep;torch.save(model.state_dict(),out/"best.pt")
        else: bad+=1
        if ep>=5 and bad>=a.patience: break
    model.load_state_dict(torch.load(out/"best.pt",map_location=dev,weights_only=True)); model.eval(); fields=[];points=[];fy=[];py=[];fm=[];pm=[];fcs=[];pcs=[];fcst=[];pcst=[];fcm=[];pcm=[];regs=[];orig=[]
    with torch.no_grad():
        for b0 in dl["test"]:
            b=mv(b0); b["x"]=b["sat"].permute(0,1,3,4,2).reshape(len(b["sat"]),72,256,6);b["point_position"]=b["position"]
            _,s,z=model.base.deterministic(b); ensf,ensp,meanf,meanp=model.base.forecast(b,a.members); cf,cp=model.csi(z,s,b)
            fs=ensf*b["field_std"][:,None,None,None]+b["field_mean"][:,None,None,None]; ps=ensp*b["point_std"][:,None,None]+b["point_mean"][:,None,None]
            fields.append(fs.cpu().numpy());points.append(ps.cpu().numpy());fy.append(b0["field_ghi"].numpy());py.append(b0["point_ghi"].numpy());fm.append(b0["field_mask"].numpy());pm.append(b0["point_mask"].numpy());fcs.append(cf.cpu().numpy());pcs.append(cp.cpu().numpy());fcst.append(b0["field_csi"].numpy());pcst.append(b0["point_csi"].numpy());fcm.append(b0["field_csi_mask"].numpy());pcm.append(b0["point_csi_mask"].numpy());regs.append(b0["region"].numpy());orig.append(b0["origin"].numpy())
    fields=np.concatenate(fields);points=np.concatenate(points);fy=np.concatenate(fy);py=np.concatenate(py);fm=np.concatenate(fm);pm=np.concatenate(pm);fcs=np.concatenate(fcs);pcs=np.concatenate(pcs);fcst=np.concatenate(fcst);pcst=np.concatenate(pcst);fcm=np.concatenate(fcm);pcm=np.concatenate(pcm)
    result={"version":"SMF-V2-CSI-joint","best_epoch":best_ep,"epochs_ran":len(hist),"stopped_epoch":ep,"train_seconds":time.time()-start,"field_GHI_by_lead":metrics(torch.tensor(fields),torch.tensor(fy),torch.tensor(fm)),"point_GHI_by_lead":metrics(torch.tensor(points),torch.tensor(py),torch.tensor(pm)),"field_CSI_by_lead":metrics(torch.tensor(fcs[:,None]),torch.tensor(fcst),torch.tensor(fcm)),"point_CSI_by_lead":metrics(torch.tensor(pcs[:,None]),torch.tensor(pcst),torch.tensor(pcm)),"peak_gpu_allocated_MiB":torch.cuda.max_memory_allocated()/2**20}
    np.savez_compressed(out/"test_predictions.npz",field_ghi_ensemble=fields,field_ghi_truth=fy,field_mask=fm,point_ghi_ensemble=points,point_ghi_truth=py,point_mask=pm,field_csi_prediction=fcs,field_csi_truth=fcst,field_csi_mask=fcm,point_csi_prediction=pcs,point_csi_truth=pcst,point_csi_mask=pcm,region=np.concatenate(regs),origin=np.concatenate(orig))
    (out/"metrics.json").write_text(json.dumps(result,indent=2));(out/"training_history.json").write_text(json.dumps(hist,indent=2));log.write(json.dumps({"event":"finished","result":result})+"\n");log.close()

if __name__=="__main__": main()
