import pathlib,numpy as np,torch
from train_solarcube_v7_fieldpoint_csi import SolarCubeWindows,clear_sky
class FeatureArray:
 def __init__(self,p):self.a=[np.load(p/(c+'.npy'),mmap_mode='r') for c in ['sza','cm','ir133','ssr','vis047','vis086']]
 def __getitem__(self,key):
  if isinstance(key,tuple):return self.a[key[-1]][key[:-1]]
  return np.stack([a[key] for a in self.a],-1)
class ValidArray:
 def __init__(self,a):self.a=a
 def __getitem__(self,key):return np.isfinite(self.a[key])
class NativeWindows(SolarCubeWindows):
 def __init__(self,*args,native,**kw):
  super().__init__(*args,**kw)
  for r in range(19):
   p=pathlib.Path(native)/('region_%02d'%(r+1));feat=FeatureArray(p);field=feat.a[3];_,_,_,point,pm=self.arr[r];self.arr[r]=(feat,field,ValidArray(field),point,pm)
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
        sza_t=torch.from_numpy(sza.reshape(self.horizon,1,120,120)); cs=clear_sky(sza_t).numpy().reshape(self.horizon,14400)
        pos=np.array(self.positions[r],np.float32); gx=2*pos[0]-1; gy=2*pos[1]-1
        psza=np.asarray(torch.nn.functional.grid_sample(torch.from_numpy(sza[:,None]),torch.tensor([[[[gx,gy]]]]).expand(self.horizon,-1,-1,-1),mode="bilinear",padding_mode="border",align_corners=True).numpy().reshape(self.horizon),np.float32)
        pcs=clear_sky(torch.from_numpy(psza)).numpy()
        fy_flat=fy.reshape(self.horizon,14400); fm_flat=fmask.reshape(self.horizon,14400)
        csi=np.clip(fy_flat/np.maximum(cs,1e-4),0,2.5); pcsi=np.clip(py/np.maximum(pcs,1e-4),0,2.5)
        csi_mask=fm_flat&(cs>20)&sza_valid.reshape(self.horizon,14400); pcsi_mask=p_mask&(pcs>20)
        coord=np.stack(np.meshgrid(np.linspace(-1,1,120),np.linspace(-1,1,120),indexing="ij"),-1).reshape(-1,2).astype(np.float32)
        result = {"sat":torch.from_numpy(x.transpose(0,3,1,2)),"nwp":torch.from_numpy(np.concatenate([nw,nvalid],-1).transpose(0,3,1,2)),
          "time_hist":torch.from_numpy(self.timefeat(np.arange(lo,t+1))),"time_future":torch.from_numpy(self.timefeat(fut)),"point_hist":torch.from_numpy(ph),"point_hist_mask":torch.from_numpy(phm),
          "field_ghi":torch.from_numpy(fy_flat),"field_mask":torch.from_numpy(fm_flat),"field_cs":torch.from_numpy(cs),"field_csi":torch.from_numpy(csi),"field_csi_mask":torch.from_numpy(csi_mask.reshape(self.horizon,14400)),
          "field_mean":torch.tensor(mu,dtype=torch.float32),"field_std":torch.tensor(sd,dtype=torch.float32),"point_mean":torch.tensor(pmean,dtype=torch.float32),"point_std":torch.tensor(psd,dtype=torch.float32),
          "point_ghi":torch.from_numpy(py),"point_mask":torch.from_numpy(p_mask),"point_cs":torch.from_numpy(pcs),"point_csi":torch.from_numpy(pcsi),"point_csi_mask":torch.from_numpy(pcsi_mask),
          "state":torch.from_numpy(cm.reshape(self.horizon,14400)),"state_mask":torch.from_numpy(cm_mask.reshape(self.horizon,14400)),"position":torch.from_numpy(pos),"region":torch.tensor(r+1),"origin":torch.tensor(t)}

        for key in ['field_ghi','point_ghi','field_csi','point_csi','state']:
            result[key]=torch.nan_to_num(result[key],nan=0.,posinf=0.,neginf=0.)
        return result

