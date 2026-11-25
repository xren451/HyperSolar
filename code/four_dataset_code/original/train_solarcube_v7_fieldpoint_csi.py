#!/usr/bin/env python3
"""Branch7: joint field/point GHI + CSI forecasting on full SolarCube.

Regional branch: historical satellite raster + UTC ERA5 history -> residual
CNN + Swin windows + global ViT + sparse grid GNN + temporal attention.
Local branch: causal in-situ history, multi-scale temporal convolutions and
explicit ramp statistics. A bounded CSI forecast reconstructs irradiance via
future clear-sky GHI. One fixed bidirectional refinement round couples field,
point and residual-diffusion forecasts; it is fixed at inference (no target-
loss-dependent stopping). COD is optional in the data contract but absent from
the audited SolarCube inputs, so this run logs COD as unavailable.
"""
from __future__ import annotations
import argparse,csv,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

def clear_sky(sza):
    c=torch.cos(torch.deg2rad(sza)).clamp_min(0)
    return 1098.0*c*torch.exp(-0.059/c.clamp_min(1e-4))

class SwinWindow(nn.Module):
    def __init__(self,d,heads=4,win=4,shift=0):
        super().__init__(); self.win=win; self.shift=shift
        self.n1=nn.LayerNorm(d); self.attn=nn.MultiheadAttention(d,heads,batch_first=True)
        self.n2=nn.LayerNorm(d); self.mlp=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
    def forward(self,x): # B,T,D,H,W
        b,t,d,h,w=x.shape; s=self.shift
        if s: x=torch.roll(x,(-s,-s),(-2,-1))
        ph=(-h)%self.win; pw=(-w)%self.win; x=F.pad(x,(0,pw,0,ph)); hp,wp=x.shape[-2:]
        z=x.permute(0,1,3,4,2).reshape(b*t,hp//self.win,self.win,wp//self.win,self.win,d)
        z=z.permute(0,1,3,2,4,5).reshape(-1,self.win*self.win,d)
        q=self.n1(z); z=z+self.attn(q,q,q,need_weights=False)[0]; z=z+self.mlp(self.n2(z))
        z=z.reshape(b*t,hp//self.win,wp//self.win,self.win,self.win,d).permute(0,1,3,2,4,5)
        z=z.reshape(b,t,hp,wp,d)[:,:,:h,:w].permute(0,1,4,2,3)
        return torch.roll(z,(s,s),(-2,-1)) if s else z

class Residual2D(nn.Module):
    def __init__(self,d):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(d,d,3,padding=1),nn.GroupNorm(8,d),nn.GELU(),nn.Conv2d(d,d,3,padding=1),nn.GroupNorm(8,d))
    def forward(self,x): return F.gelu(x+self.net(x))

class SparseGridGNN(nn.Module):
    def __init__(self,d,side=16):
        super().__init__(); self.msg=nn.Sequential(nn.Linear(2*d+2,d),nn.GELU(),nn.Linear(d,d)); self.norm=nn.LayerNorm(d)
        row,col=torch.meshgrid(torch.arange(side),torch.arange(side),indexing="ij"); coords=torch.stack([row.flatten(),col.flatten()],-1).float()
        src=[]; dst=[]
        for i in range(side):
            for j in range(side):
                u=i*side+j
                for di,dj in [(1,0),(-1,0),(0,1),(0,-1)]:
                    ii,jj=i+di,j+dj
                    if 0<=ii<side and 0<=jj<side: src.append(ii*side+jj); dst.append(u)
        self.register_buffer("src",torch.tensor(src,dtype=torch.long)); self.register_buffer("dst",torch.tensor(dst,dtype=torch.long))
        self.register_buffer("coords",coords/max(side-1,1))
    def forward(self,x): # B,L,N,D
        rel=self.coords[self.src]-self.coords[self.dst]
        m=self.msg(torch.cat([x[...,self.src,:],x[...,self.dst,:],rel.view(*([1]*(x.ndim-2)),-1,2).expand(*x.shape[:-2],-1,2)],-1))
        agg=torch.zeros_like(x); agg.index_add_(-2,self.dst,m)
        deg=torch.bincount(self.dst,minlength=x.shape[-2]).to(x.dtype).clamp_min(1)
        return self.norm(x+agg/deg.view(*([1]*(x.ndim-2)),-1,1))

class MultiScaleLocal(nn.Module):
    def __init__(self,d):
        super().__init__(); self.c3=nn.Conv1d(1,d,3,padding=1); self.c9=nn.Conv1d(1,d,9,padding=4); self.c21=nn.Conv1d(1,d,21,padding=10)
        self.ramp=nn.Sequential(nn.Linear(6,d),nn.GELU(),nn.Linear(d,d)); self.out=nn.Sequential(nn.Linear(4*d,d),nn.GELU(),nn.Linear(d,d))
    def forward(self,h,mask): # B,H physical normalized irradiance
        m=mask.bool()&torch.isfinite(h); v=torch.where(m,h,torch.zeros_like(h)); den=m.sum(1).clamp_min(1)
        dy=h[:,1:]-h[:,:-1]; dm=m[:,1:]&m[:,:-1]; dv=torch.where(dm,dy,0)
        mean=v.sum(1)/den; slope=dv.sum(1)/dm.sum(1).clamp_min(1)
        up=dv.masked_fill(~dm,-torch.inf).amax(1); dn=dv.masked_fill(~dm,torch.inf).amin(1)
        up=torch.where(torch.isfinite(up),up,0); dn=torch.where(torch.isfinite(dn),dn,0)
        ramp=torch.stack([mean,slope,up,dn,dm.float().mean(1),m.float().mean(1)],-1)
        z=v[:,None]; a=F.gelu(self.c3(z)).mean(-1); b=F.gelu(self.c9(z)).mean(-1); c=F.gelu(self.c21(z)).mean(-1)
        return self.out(torch.cat([a,b,c,self.ramp(ramp)],-1)),ramp

class ConditionalResidualDiffusion(nn.Module):
    """Small conditional DDPM over CSI residuals around a learned residual mean."""
    def __init__(self,dim,cond=64,width=256,steps=8):
        super().__init__(); self.dim=dim; self.steps=steps
        self.time=nn.Embedding(steps,width); self.net=nn.Sequential(nn.Linear(dim+cond+width,width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,dim))
        beta=torch.linspace(1e-4,.02,steps); self.register_buffer("beta",beta); self.register_buffer("abar",(1-beta).cumprod(0))
    def loss(self,res,mask,cond):
        b=res.shape[0]; x=res.flatten(1); m=mask.expand_as(res).flatten(1).float(); k=torch.randint(self.steps,(b,),device=x.device); eps=torch.randn_like(x); a=self.abar[k,None]
        noisy=a.sqrt()*torch.where(m.bool(),x,torch.zeros_like(x))+(1-a).sqrt()*eps
        pred=self.net(torch.cat([noisy,cond,self.time(k)],-1)); return ((pred-eps).square()*m).sum()/m.sum().clamp_min(1)
    @torch.no_grad()
    def sample(self,cond,shape,members=4):
        b=cond.shape[0]; cond=cond[:,None].expand(-1,members,-1).reshape(b*members,-1); x=torch.randn(b*members,self.dim,device=cond.device)
        for k in reversed(range(self.steps)):
            kk=torch.full((len(x),),k,device=x.device,dtype=torch.long); eps=self.net(torch.cat([x,cond,self.time(kk)],-1)); aa=self.abar[k]
            x=(x-self.beta[k]/(1-aa).sqrt()*eps)/(1-self.beta[k]).sqrt()
            if k: x=x+self.beta[k].sqrt()*torch.randn_like(x)
        return x.view(b,members,*shape[1:])

class SolarCubeWindows(Dataset):
    def __init__(self,root,prepared,sitelist,nwp_root,split,hist=72,horizon=24,stride=3):
        self.root=Path(root); self.prepared=Path(prepared); self.nwp_root=Path(nwp_root); self.hist=hist; self.horizon=horizon; self.n=8760
        cuts=[0,int(self.n*.70),int(self.n*.85),self.n]; a,b={"train":(cuts[0],cuts[1]),"val":(cuts[1],cuts[2]),"test":(cuts[2],cuts[3])}[split]
        self.rows=[]; self.arr=[]; self.stats=[]; self.pstats=[]; self.positions=[]; self.nwp=[]; self.nstats=[]; self.nvalid=[]
        cache=json.loads((self.root/"READY16.json").read_text()); prep=json.loads((self.prepared/"READY.json").read_text()); self.source_regions=prep["regions"]
        if not cache.get("complete") or len(cache.get("regions",[]))!=19: raise ValueError("Require full SolarCube 19-region 16-grid cache")
        with Path(sitelist).open(newline="",encoding="utf-8-sig") as f: sites={int(x["tile_id"]):x for x in csv.DictReader(f)}
        origins=np.arange(max(a+hist-1,hist-1),b-horizon,stride,dtype=np.int32)
        for r in range(1,20):
            d=self.root/f"region_{r:02d}"; feat=np.load(d/"features16.npy",mmap_mode="r"); field=np.load(d/"field_ssr16.npy",mmap_mode="r"); fm=np.load(d/"field_mask16.npy",mmap_mode="r"); point=np.load(d/"point_insitu.npy",mmap_mode="r"); pm=np.load(d/"point_mask.npy",mmap_mode="r")
            st=self.source_regions[r-1]["train_only_stats"]; means=np.array([x["mean"] for x in st],np.float32); stds=np.maximum([x["std"] for x in st],1e-4)
            # target normalization is retained for stable GHI losses; CSI supervision is physical-ratio based.
            mu=float(np.nanmean(np.asarray(field[:cuts[1]],np.float32)[np.asarray(fm[:cuts[1]],bool)])); sd=float(np.nanstd(np.asarray(field[:cuts[1]],np.float32)[np.asarray(fm[:cuts[1]],bool)])); sd=max(sd,1e-4)
            pv=np.asarray(pm[:cuts[1]],bool); pvals=np.asarray(point[:cuts[1]],np.float32)[pv]; pmean=float(pvals.mean()) if len(pvals) else 0.; psd=max(float(pvals.std()) if len(pvals) else 1.,1e-4)
            self.arr.append((feat,field,fm,point,pm)); self.stats.append((means,stds,mu,sd)); self.pstats.append((pmean,psd))
            site=sites[r]; rf=(float(site["lat_ulcnr"])-float(site["lats"]))/(float(site["lat_ulcnr"])-float(site["lat_lrcnr"])); cf=(float(site["lons"])-float(site["lon_ulcnr"]))/(float(site["lon_lrcnr"])-float(site["lon_ulcnr"]))
            if not(0<=rf<=1 and 0<=cf<=1): raise ValueError(f"station outside region {r}")
            self.positions.append((cf,rf))
            nd=self.nwp_root/f"region_{r:02d}"; nf=nd/"era5_8x8.npy"; vm=nd/"valid.npy"
            if nf.exists() and vm.exists():
                na=np.load(nf,mmap_mode="r"); nv=np.load(vm,mmap_mode="r"); train=np.asarray(nv[:cuts[1]],bool)
                vals=np.asarray(na[:cuts[1]],np.float32)[train]; nmean=vals.mean((0,1,2)) if vals.size else np.zeros(8,np.float32); nsd=vals.std((0,1,2)) if vals.size else np.ones(8,np.float32); nsd=np.maximum(nsd,1e-3)
                self.nwp.append(na); self.nvalid.append(nv); self.nstats.append((nmean,nsd))
            else: self.nwp.append(None); self.nvalid.append(None); self.nstats.append((np.zeros(8,np.float32),np.ones(8,np.float32)))
            valid=np.asarray(fm).reshape(self.n,-1).any(1)|np.asarray(pm,bool)
            pref=np.concatenate([[0],np.cumsum(valid,dtype=np.int64)]); ok=(pref[origins+horizon+1]-pref[origins+1])>0
            self.rows.extend((r-1,int(t)) for t in origins[ok])
        self.split=split; self.cuts=cuts
    def __len__(self): return len(self.rows)
    @staticmethod
    def timefeat(idx):
        day=idx//24; hour=idx%24; doy=day%365
        return np.stack([np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),np.sin(2*np.pi*doy/365),np.cos(2*np.pi*doy/365),hour/23,doy/364],-1).astype(np.float32)
    def __getitem__(self,i):
        r,t=self.rows[i]; feat,field,fm,point,pm=self.arr[r]; lo=t-self.hist+1; fut=np.arange(t+1,t+1+self.horizon); stats=self.stats[r]; means,stds,mu,sd=stats
        x=np.asarray(feat[lo:t+1],np.float32); rawx=x.copy(); x=np.where(np.isfinite(x),(x-means)/stds,0).astype(np.float32)
        # ERA5 fields are historical only. Upsampling from coarse 8x8 does not
        # add information, it only aligns feature grids for fusion.
        if self.nwp[r] is not None:
            nv=np.asarray(self.nvalid[r][lo:t+1],bool); nw=np.asarray(self.nwp[r][lo:t+1],np.float32); nm,ns=self.nstats[r]; nw=np.where(np.isfinite(nw),(nw-nm)/ns,0); nw*=nv[:,None,None,None]; nvalid=np.broadcast_to(nv[:,None,None,None],(len(nv),8,8,1)).astype(np.float32)
        else:
            nw=np.zeros((self.hist,8,8,8),np.float32); nvalid=np.zeros((self.hist,8,8,1),np.float32)
        fy=np.asarray(field[fut],np.float32); fmask=np.asarray(fm[fut],bool)&np.isfinite(fy); py=np.asarray(point[fut],np.float32); p_mask=np.asarray(pm[fut],bool)&np.isfinite(py)
        sza=np.asarray(feat[fut,:,:,0],np.float32); sza_valid=np.isfinite(sza); sza=np.where(sza_valid,sza,180.)
        # CM is satellite cloud-mask fraction; supervised as a future state, not used as future input.
        cm=np.asarray(feat[fut,:,:,1],np.float32); cm_mask=np.isfinite(cm); cm=np.where(cm_mask,cm,0)
        pmean,psd=self.pstats[r]; ph=np.asarray(point[lo:t+1],np.float32); phm=np.asarray(pm[lo:t+1],bool)&np.isfinite(ph); ph=np.where(phm,(ph-pmean)/psd,0).astype(np.float32)
        sza_t=torch.from_numpy(sza.reshape(self.horizon,1,16,16)); cs=clear_sky(sza_t).numpy().reshape(self.horizon,256)
        pos=np.array(self.positions[r],np.float32); gx=2*pos[0]-1; gy=2*pos[1]-1
        psza=np.asarray(torch.nn.functional.grid_sample(torch.from_numpy(sza[:,None]),torch.tensor([[[[gx,gy]]]]).expand(self.horizon,-1,-1,-1),mode="bilinear",padding_mode="border",align_corners=True).numpy().reshape(self.horizon),np.float32)
        pcs=clear_sky(torch.from_numpy(psza)).numpy()
        fy_flat=fy.reshape(self.horizon,256); fm_flat=fmask.reshape(self.horizon,256)
        csi=np.clip(fy_flat/np.maximum(cs,1e-4),0,2.5); pcsi=np.clip(py/np.maximum(pcs,1e-4),0,2.5)
        csi_mask=fm_flat&(cs>20)&sza_valid.reshape(self.horizon,256); pcsi_mask=p_mask&(pcs>20)
        coord=np.stack(np.meshgrid(np.linspace(-1,1,16),np.linspace(-1,1,16),indexing="ij"),-1).reshape(-1,2).astype(np.float32)
        return {"sat":torch.from_numpy(x.transpose(0,3,1,2)),"nwp":torch.from_numpy(np.concatenate([nw,nvalid],-1).transpose(0,3,1,2)),
          "time_hist":torch.from_numpy(self.timefeat(np.arange(lo,t+1))),"time_future":torch.from_numpy(self.timefeat(fut)),"point_hist":torch.from_numpy(ph),"point_hist_mask":torch.from_numpy(phm),
          "field_ghi":torch.from_numpy(fy_flat),"field_mask":torch.from_numpy(fm_flat),"field_cs":torch.from_numpy(cs),"field_csi":torch.from_numpy(csi),"field_csi_mask":torch.from_numpy(csi_mask.reshape(self.horizon,256)),
          "field_mean":torch.tensor(mu,dtype=torch.float32),"field_std":torch.tensor(sd,dtype=torch.float32),"point_mean":torch.tensor(pmean,dtype=torch.float32),"point_std":torch.tensor(psd,dtype=torch.float32),
          "point_ghi":torch.from_numpy(py),"point_mask":torch.from_numpy(p_mask),"point_cs":torch.from_numpy(pcs),"point_csi":torch.from_numpy(pcsi),"point_csi_mask":torch.from_numpy(pcsi_mask),
          "state":torch.from_numpy(cm.reshape(self.horizon,256)),"state_mask":torch.from_numpy(cm_mask.reshape(self.horizon,256)),"position":torch.from_numpy(pos),"region":torch.tensor(r+1),"origin":torch.tensor(t)}

class Branch7(nn.Module):
    def __init__(self,hist=72,width=32,heads=4,diff_steps=8):
        super().__init__(); self.horizon=24; self.width=width
        # ResNet visual stem and residual blocks over 6 satellite + 8 ERA5 + availability.
        self.stem=nn.Sequential(nn.Conv2d(15,width,3,padding=1),nn.GroupNorm(8,width),nn.GELU(),Residual2D(width),Residual2D(width))
        self.swin=nn.ModuleList([SwinWindow(width,heads,4,0),SwinWindow(width,heads,4,2)])
        layer=nn.TransformerEncoderLayer(width,heads,2*width,.1,batch_first=True,norm_first=True,activation="gelu")
        self.vit=nn.TransformerEncoder(layer,1)
        temp=nn.TransformerEncoderLayer(width,heads,2*width,.1,batch_first=True,norm_first=True,activation="gelu")
        self.temporal=nn.TransformerEncoder(temp,1); self.gnn1=SparseGridGNN(width); self.gnn2=SparseGridGNN(width)
        self.clock=nn.Sequential(nn.Linear(6,width),nn.GELU(),nn.Linear(width,width)); self.lead=nn.Embedding(24,width)
        self.state_head=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,1))
        self.field_csi_head=nn.Sequential(nn.Linear(width+1,width),nn.GELU(),nn.Linear(width,1))
        self.local=MultiScaleLocal(width); self.point_fuse=nn.Sequential(nn.Linear(3*width,width),nn.GELU(),nn.Linear(width,width))
        self.point_csi_head=nn.Sequential(nn.Linear(width+1,width),nn.GELU(),nn.Linear(width,1))
        self.field_resid=nn.Linear(width,1); self.point_resid=nn.Linear(width,1)
        self.diff=ConditionalResidualDiffusion(24*257,cond=2*width,width=256,steps=diff_steps)
        self.field_msg=nn.Sequential(nn.Linear(width+3,width),nn.GELU(),nn.Linear(width,width)); self.point_msg=nn.Sequential(nn.Linear(width+3,width),nn.GELU(),nn.Linear(width,width))
        self.refine_field=nn.Sequential(nn.Linear(width+1,width),nn.GELU(),nn.Linear(width,1)); self.refine_point=nn.Sequential(nn.Linear(width+1,width),nn.GELU(),nn.Linear(width,1))
    def encode(self,b):
        sat=b["sat"]; nwp=F.interpolate(b["nwp"].flatten(0,1),size=(16,16),mode="bilinear",align_corners=False).view(*b["nwp"].shape[:2],9,16,16)
        inp=torch.cat([sat,nwp],2); B,H,C,Y,X=inp.shape; z=self.stem(inp.flatten(0,1)).view(B,H,self.width,Y,X)
        for m in self.swin: z=m(z)
        # global ViT interaction on 8x8 pooled tokens, projected back to the grid.
        pooled=F.avg_pool2d(z.flatten(0,1),2).view(B,H,self.width,8,8).permute(0,1,3,4,2).reshape(B*H,64,self.width)
        gv=self.vit(pooled).view(B,H,8,8,self.width).permute(0,1,4,2,3).flatten(0,1)
        z=z+F.interpolate(gv,size=(16,16),mode="bilinear",align_corners=False).view(B,H,self.width,16,16)
        tok=z.permute(0,3,4,1,2).reshape(B*256,H,self.width); tok=self.temporal(tok)
        reg=(tok[:,-1]+tok.mean(1)).view(B,16,16,self.width).reshape(B,256,self.width)
        return self.gnn2(self.gnn1(reg[:,None]))[:,0] # sparse regional state
    def decode(self,reg,local,b,init_field=None,init_point=None,diff_hint=None):
        B=reg.shape[0]; tf=self.clock(b["time_future"])+self.lead.weight[None]
        r=reg[:,None].expand(-1,24,-1,-1)+tf[:,:,None,:]
        state=torch.sigmoid(self.state_head(r)[...,0])
        fcsi=2.5*torch.sigmoid(self.field_csi_head(torch.cat([r,state[...,None]],-1))[...,0])
        # bilinear gather at the real in-situ station position within each tile.
        pos=b["position"]; grid=(2*pos-1).view(B,1,1,2)
        local_grid=F.grid_sample(reg.transpose(1,2).reshape(B,self.width,16,16),grid,mode="bilinear",padding_mode="border",align_corners=True).reshape(B,self.width)
        pf=self.point_fuse(torch.cat([local,local_grid,reg.mean(1)],-1))[:,None].expand(-1,24,-1)+tf
        pcsi=2.5*torch.sigmoid(self.point_csi_head(torch.cat([pf,fcsi.mean(-1,keepdim=True)],-1))[...,0])
        # residual means around the initial deterministic CSI forecast
        fres=torch.tanh(self.field_resid(r)[...,0])*0.25; pres=torch.tanh(self.point_resid(pf)[...,0])*0.25
        if diff_hint is not None:
            fres=fres+diff_hint[:,:,:256]; pres=pres+diff_hint[:,:,256]
        if init_field is not None:
            # one fixed bidirectional point<->regional update round
            fctx=torch.stack([init_field,pcsi[:,:,None].expand_as(fcsi),fres],-1)
            r2=r+self.field_msg(torch.cat([r,fctx],-1)); pctx=torch.stack([init_point,fcsi.mean(-1),pres],-1)
            pf2=pf+self.point_msg(torch.cat([pf,pctx],-1))
            fcsi=2.5*torch.sigmoid(self.refine_field(torch.cat([r2,state[...,None]],-1))[...,0]+torch.logit((fcsi/2.5).clamp(1e-4,1-1e-4)))
            pcsi=2.5*torch.sigmoid(self.refine_point(torch.cat([pf2,pcsi[...,None]],-1))[...,0]+torch.logit((pcsi/2.5).clamp(1e-4,1-1e-4)))
            r=r2; pf=pf2
        cond=torch.cat([reg.mean(1),local],-1)
        return fcsi,pcsi,state,fres,pres,cond,r,pf
    def forward(self,b,ensemble=0):
        reg=self.encode(b); local,ramp=self.local(b["point_hist"],b["point_hist_mask"])
        f0,p0,state,fr,pr,cond,r,pf=self.decode(reg,local,b)
        hint=torch.cat([fr,pr[...,None]],-1)
        if self.training:
            fc,pc,state,fr,pr,cond,r,pf=self.decode(reg,local,b,init_field=f0,init_point=p0,diff_hint=hint)
            res=torch.cat([b["field_csi"]-f0,b["point_csi"][...,None]-p0[...,None]],-1)
            m=torch.cat([b["field_csi_mask"],b["point_csi_mask"][...,None]],-1)
            dloss=self.diff.loss(res.detach()-hint,m,cond)
            hmask=m.float(); hloss=((hint-res.detach()).square()*hmask).sum()/hmask.sum().clamp_min(1)
            return fc,pc,state,dloss,hloss
        fc,pc,state,fr,pr,cond,r,pf=self.decode(reg,local,b,init_field=f0,init_point=p0,diff_hint=hint)
        if ensemble:
            eps=self.diff.sample(cond,(len(fc),24,257),ensemble); fens=(fc[:,None]+eps[:,:,:,:256]).clamp(0,2.5); pens=(pc[:,None]+eps[:,:,:,256]).clamp(0,2.5)
        else: fens=fc[:,None]; pens=pc[:,None]
        return fc,pc,state,fens,pens

def masked_mse(x,y,m): return ((x-y).square()*m.float()).sum()/m.float().sum().clamp_min(1)
def metrics(ens,y,m):
    mean=ens.mean(1); rows=[]
    for lead in range(y.shape[1]):
        mm=m[:,lead].bool()
        if not mm.any(): continue
        e=mean[:,lead]-y[:,lead]; members=ens[:,:,lead]
        crps=(members-y[:,None,lead]).abs().mean(1)-.5*(members[:,None]-members[:,:,None]).abs().mean((1,2))
        lo,hi=torch.quantile(members,torch.tensor([.05,.95],device=members.device),dim=1)
        rows.append({"lead_hour":lead+1,"n":int(mm.sum()),"MAE":float(e.abs()[mm].mean()),"RMSE":float(e[mm].square().mean().sqrt()),"CRPS":float(crps[mm].mean()),"PI90_coverage":float(((y[:,lead]>=lo)&(y[:,lead]<=hi))[mm].float().mean()),"PI90_width":float((hi-lo)[mm].mean()),"spread_skill":float(members.std(1)[mm].mean()/e[mm].square().mean().sqrt().clamp_min(1e-8))})
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",required=True); p.add_argument("--prepared",required=True); p.add_argument("--sitelist",required=True); p.add_argument("--nwp",required=True); p.add_argument("--out",required=True)
    p.add_argument("--epochs",type=int,default=60); p.add_argument("--patience",type=int,default=8); p.add_argument("--batch",type=int,default=8); p.add_argument("--stride",type=int,default=3); p.add_argument("--workers",type=int,default=0); p.add_argument("--seed",type=int,default=42); p.add_argument("--members",type=int,default=8); p.add_argument("--device",default="cuda")
    a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=False); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    dev=torch.device(a.device); ds={s:SolarCubeWindows(a.data,a.prepared,a.sitelist,a.nwp,s,stride=a.stride) for s in ("train","val","test")}; dl={s:DataLoader(ds[s],batch_size=a.batch,shuffle=s=="train",num_workers=a.workers,pin_memory=dev.type=="cuda",persistent_workers=a.workers>0) for s in ds}
    model=Branch7().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=8e-5,weight_decay=1e-5); best=float("inf"); bad=0; hist=[]; start=time.time(); log=(out/"train.log").open("w",buffering=1)
    nwp_ready=Path(a.nwp)/"READY.json"; nwp_cov=json.loads(nwp_ready.read_text()) if nwp_ready.exists() else {"status":"cache readiness missing"}
    cfg={"version":"V7-Branch7-CSI-joint","dataset":"SolarCube 2018 all 19 regions, full year","split":"chronological 70/15/15; split-confined windows","window_stride_hours":a.stride,"history_hours":72,"forecast_hours":24,"clear_sky":"Haurwitz GHI=1098*cos(SZA)*exp(-0.059/cos(SZA)); denominator >20 W m-2; CSI clipped [0,2.5]","targets":{"field":"satellite SSR, W m-2","point":"separate collocated in-situ GHI, W m-2","CSI":"both field and point GHI / clear-sky GHI"},"inputs":{"satellite":"past 6 channels only: sza, cm, ir133, historical ssr, vis047, vis086","NWP":"past ERA5 t2m,d2m,sp,u10,v10,u100,v100,tcc; 8x8 cache upsampled for fusion; explicit valid mask","NWP_cache_coverage":nwp_cov.get("regions",[]),"in_situ":"past station GHI only; multi-scale kernels 3/9/21 and ramp signed magnitude/up/down","COD":"unavailable in audited SolarCube input tree; omitted, no synthetic COD"},"backbone":"ResNet residual raster stem + 16x16 shifted Swin windows + global ViT at 8x8 + sparse 4-neighbor GNN + temporal attention","refinement":"one fixed bidirectional field-point message round conditioned on initial forecasts and learned residual-diffusion mean; fixed at inference","diffusion":{"space":"joint field+point CSI residual","steps":8,"members":a.members},"early_stopping":{"patience":a.patience,"restore_best":True},"batch":a.batch,"seed":a.seed,"max_epochs":a.epochs,"source_code":"train_solarcube_v7_fieldpoint_csi.py"}
    (out/"config.json").write_text(json.dumps(cfg,indent=2)); (out/"README.md").write_text("# V7 Branch7 CSI joint field-point forecast\n\n"+json.dumps(cfg,indent=2)+"\n\nArtifacts: complete JSONL train.log, history, best checkpoint, config, test truth/prediction archive, per-lead GHI and CSI metrics, lead plots. ERA5 coverage is recorded by the NWP cache READY.json; unavailable cells are mask-zero.\n")
    log.write(json.dumps({"event":"started","utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"train_windows":len(ds["train"]),"val_windows":len(ds["val"]),"test_windows":len(ds["test"]),"parameters":sum(p.numel() for p in model.parameters()),"device":str(dev),"early_stopping":cfg["early_stopping"]})+"\n")
    def move(b): return {k:v.to(dev,non_blocking=True) if torch.is_tensor(v) else v for k,v in b.items()}
    for ep in range(1,a.epochs+1):
        model.train(); tr=[]
        for b0 in dl["train"]:
            b=move(b0); opt.zero_grad(set_to_none=True); fc,pc,state,dloss,hloss=model(b)
            fghi=fc*b["field_cs"]; pghi=pc*b["point_cs"]
            fl=masked_mse((fghi-b["field_ghi"])/1000,torch.zeros_like(fghi),b["field_mask"])
            pl=masked_mse((pghi-b["point_ghi"])/1000,torch.zeros_like(pghi),b["point_mask"])
            cfl=masked_mse(fc,b["field_csi"],b["field_csi_mask"]); cpl=masked_mse(pc,b["point_csi"],b["point_csi_mask"])
            sl=masked_mse(state,b["state"],b["state_mask"])
            loss=fl+pl+.5*(cfl+cpl)+.1*sl+.03*dloss+.05*hloss
            if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite training loss at epoch {ep}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); tr.append([float(loss.detach()),float(fl.detach()),float(pl.detach()),float(cfl.detach()),float(cpl.detach()),float(sl.detach()),float(dloss.detach())])
        model.eval(); va=[]
        with torch.no_grad():
            for b0 in dl["val"]:
                b=move(b0); fc,pc,state,_,_=model(b); fghi=fc*b["field_cs"]; pghi=pc*b["point_cs"]
                val=masked_mse((fghi-b["field_ghi"])/1000,torch.zeros_like(fghi),b["field_mask"])+masked_mse((pghi-b["point_ghi"])/1000,torch.zeros_like(pghi),b["point_mask"])+.5*(masked_mse(fc,b["field_csi"],b["field_csi_mask"])+masked_mse(pc,b["point_csi"],b["point_csi_mask"]))
                va.append(float(val))
        v=float(np.mean(va)); row={"epoch":ep,"train_loss":np.mean(tr,axis=0).tolist(),"val_joint_loss":v,"elapsed_sec":time.time()-start,"peak_gpu_allocated_MiB":torch.cuda.max_memory_allocated(dev)/2**20 if dev.type=="cuda" else 0}; hist.append(row); log.write(json.dumps(row)+"\n")
        if v<best-1e-7: best=v;bad=0;torch.save(model.state_dict(),out/"best.pt"); best_ep=ep
        else: bad+=1
        if ep>=5 and bad>=a.patience: break
    model.load_state_dict(torch.load(out/"best.pt",map_location=dev,weights_only=True)); model.eval(); allp={k:[] for k in ["fg","fy","fm","fcs","fc","fcm","pg","py","pm","pcs","pc","pcm","reg","origin"]}
    with torch.no_grad():
        for b0 in dl["test"]:
            b=move(b0); fc,pc,state,fe,pe=model(b,a.members); fens=fe*b["field_cs"][:,None]; pens=pe*b["point_cs"][:,None]
            for key,val in [("fg",fens),("fy",b["field_ghi"]),("fm",b["field_mask"]),("fcs",fe),("fc",b["field_csi"]),("fcm",b["field_csi_mask"]),("pg",pens),("py",b["point_ghi"]),("pm",b["point_mask"]),("pcs",pe),("pc",b["point_csi"]),("pcm",b["point_csi_mask"]),("reg",b["region"]),("origin",b["origin"])]: allp[key].append(val.detach().cpu().numpy())
    ar={k:np.concatenate(v) for k,v in allp.items()}; fmet=metrics(torch.tensor(ar["fg"]),torch.tensor(ar["fy"]),torch.tensor(ar["fm"])); pmet=metrics(torch.tensor(ar["pg"]),torch.tensor(ar["py"]),torch.tensor(ar["pm"]))
    fcmet=metrics(torch.tensor(ar["fcs"]),torch.tensor(ar["fc"]),torch.tensor(ar["fcm"])); pcmet=metrics(torch.tensor(ar["pcs"]),torch.tensor(ar["pc"]),torch.tensor(ar["pcm"]))
    result={"version":"V7-Branch7-CSI-joint","dataset":"SolarCube full19/full-year 2018","best_epoch":best_ep,"epochs_ran":len(hist),"stopped_epoch":ep,"train_seconds":time.time()-start,"test_windows":len(ar["origin"]),"parameters":sum(p.numel() for p in model.parameters()),"peak_gpu_allocated_MiB":torch.cuda.max_memory_allocated(dev)/2**20 if dev.type=="cuda" else 0,"field_GHI_by_lead":fmet,"point_GHI_by_lead":pmet,"field_CSI_by_lead":fcmet,"point_CSI_by_lead":pcmet,"notes":"COD unavailable; ERA5 partial coverage explicitly masked. One fixed refinement round."}
    np.savez_compressed(out/"test_predictions.npz",field_ghi_ensemble=ar["fg"],field_ghi_truth=ar["fy"],field_mask=ar["fm"],field_csi_ensemble=ar["fcs"],field_csi_truth=ar["fc"],field_csi_mask=ar["fcm"],point_ghi_ensemble=ar["pg"],point_ghi_truth=ar["py"],point_mask=ar["pm"],point_csi_ensemble=ar["pcs"],point_csi_truth=ar["pc"],point_csi_mask=ar["pcm"],region=ar["reg"],origin=ar["origin"])
    (out/"metrics.json").write_text(json.dumps(result,indent=2)); (out/"training_history.json").write_text(json.dumps(hist,indent=2)); log.write(json.dumps({"event":"finished","result":result})+"\n"); log.close()
    try:
        import matplotlib.pyplot as plt
        f=ar["fg"].mean(1); y=ar["fy"]; mask=ar["fm"].astype(bool); fig,axs=plt.subplots(1,4,figsize=(16,4))
        for ax,h in zip(axs,[0,5,11,23]): ax.imshow(np.where(mask[0,h],y[0,h].reshape(16,16),np.nan)); ax.contour(f[0,h].reshape(16,16),colors="white",linewidths=.6); ax.set_title(f"lead {h+1} h")
        fig.tight_layout(); fig.savefig(out/"field_forecast_examples.png",dpi=180); plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4)); ax.plot(ar["py"][0],label="truth"); ax.plot(ar["pg"][0].mean(0),label="forecast"); ax.fill_between(np.arange(24),np.quantile(ar["pg"][0],.05,axis=0),np.quantile(ar["pg"][0],.95,axis=0),alpha=.2); ax.legend(); ax.set_xlabel("Lead hour"); ax.set_ylabel("GHI W m-2"); fig.tight_layout(); fig.savefig(out/"point_forecast_example.png",dpi=180); plt.close(fig)
    except Exception as e: (out/"PLOT_ERROR.txt").write_text(repr(e))

if __name__=="__main__": main()
