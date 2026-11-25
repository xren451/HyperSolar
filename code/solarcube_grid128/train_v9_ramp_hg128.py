import os,sys,pathlib,json,time,random,hashlib,argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
HERE=pathlib.Path(__file__).parent
REG=pathlib.Path('/mnt/wangyishan_share/wangdongdong/smf_registry_variants_20261105')
sys.path.insert(0,str(REG));sys.path.insert(0,str(HERE))
from native_data import NativeWindows
from v9_ramp_hg128_model import V9RampHyperedgeCSI
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--out',required=True);a=ap.parse_args()
 out=pathlib.Path(a.out);out.mkdir(exist_ok=False,parents=True);log=(out/'train.log').open('w',buffering=1)
 def record(d):log.write(json.dumps(d)+'\n');print(json.dumps(d),flush=True)
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 torch.set_num_threads(4)
 cfg=dict(version='V9-RampHG-Grid128-native120-supervision',model_grid=[128,128],label_grid=[120,120],spatial_preprocessing_downsampling=False,history=72,horizon=24,stride=3,seed=42,all_regions=19,split='chronological70/15/15',epochs=60,patience=8,batch=4,lr=1e-4,local_hyperedges='128-grid sparse hypergraph; 32x32 local patches; targets/metrics aggregated to native120',encoder='V5 ResNet-Swin-ViT encoder at128; 120 native inputs bilinear-resampled; 64x64 latent grid; 128 future field queries; native120 labels',normalization='reuse earlier training-only input statistics, no validation/test fitting',diffusion='joint native14400 field nodes plus one point, eight diffusion steps',precision='float32; checkpoint raster frames in groups of2',source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')})
 (out/'config.json').write_text(json.dumps(cfg,indent=2));record(dict(event='initializing',gpu=torch.cuda.get_device_name()))
 args=(str(REG/'data/SolarCube_2018_full19_hourly'),'/mnt/wangyishan_share/wangdongdong/smf_full_20261025/solarcube/prepared','/mnt/wangyishan_share/paper_solar_datasets/solarcube/raw/SolarCube_sitelist.csv',str(REG/'data/SolarCube_2018_era5_8x8_v7'))
 ds={s:NativeWindows(*args,s,stride=3,native='/mnt/wangyishan_share/wangdongdong/smf_native120_20261109') for s in ['train','val','test']}
 dl={s:DataLoader(d,batch_size=4,shuffle=s=='train',num_workers=0) for s,d in ds.items()}
 model=V9RampHyperedgeCSI().cuda();encoder=model.base.encoder;original=encoder.forward
 def chunked(x):
  return torch.cat([checkpoint(original,x[:,i:i+2],use_reentrant=False) if model.training else original(x[:,i:i+2]) for i in range(0,x.shape[1],2)],dim=1)
 encoder.forward=chunked
 opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
 def move(b):return {k:v.cuda() for k,v in b.items()}
 start=time.time();b=move(next(iter(dl['train'])));model.train();o=model(b);loss=model.joint_loss(o,b)
 assert b['sat'].shape[-2:]==(120,120) and o['field_csi'].shape==(4,24,16384)
 if not torch.isfinite(loss['loss']):raise RuntimeError('nonfinite smoke loss')
 loss['loss'].backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 opt.step();opt.zero_grad(set_to_none=True)
 record(dict(event='real_native120_forward_backward_passed',loss=float(loss['loss']),shape=list(o['field_csi'].shape),seconds=time.time()-start,peak_MiB=torch.cuda.max_memory_allocated()/2**20,windows={s:len(d) for s,d in ds.items()}))
 # Reinitialize after the smoke update so full training starts from the exact declared seed.
 if a.smoke:(out/'SMOKE_OK.json').write_text(json.dumps({'passed':True}));return
 del b,o,loss,model,opt,encoder,original
 torch.manual_seed(42);np.random.seed(42);torch.cuda.manual_seed_all(42)
 model=V9RampHyperedgeCSI().cuda();encoder=model.base.encoder;original=encoder.forward
 def chunked(x):
  return torch.cat([checkpoint(original,x[:,i:i+2],use_reentrant=False) if model.training else original(x[:,i:i+2]) for i in range(0,x.shape[1],2)],dim=1)
 encoder.forward=chunked
 opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
 model.train()
 # The full run now starts from freshly seeded parameters and a fresh optimizer.
 best=float('inf');bad=0;step=0
 for ep in range(1,61):
  model.train();total=0.;count=0
  for raw in dl['train']:
   b=move(raw);opt.zero_grad(set_to_none=True);o=model(b);ls=model.joint_loss(o,b)
   if not torch.isfinite(ls['loss']):raise RuntimeError('nonfinite training loss')
   ls['loss'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();total+=float(ls['loss']);count+=1;step+=1
   if count%20==0:record(dict(event='train_progress',epoch=ep,step=step,epoch_steps=count,loss=total/count,elapsed_seconds=time.time()-start))
  model.eval();v=0.;vn=0
  with torch.no_grad():
   for raw in dl['val']:
    b=move(raw);o=model(b);ls=model.joint_loss(o,b);v+=float(ls['deterministic']);vn+=1
  val=v/max(vn,1);record(dict(event='epoch',epoch=ep,train_loss=total/count,val_deterministic=val,elapsed_seconds=time.time()-start))
  torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=ep),out/'last.pt')
  if val<best:best=val;bad=0;torch.save(model.state_dict(),out/'best.pt')
  else:bad+=1
  if ep>=5 and bad>=8:break
 model.load_state_dict(torch.load(out/'best.pt',weights_only=True));model.eval();archive=out/'test_predictions';archive.mkdir();acc={s:np.zeros(4) for s in ['field','point']}
 with torch.no_grad():
  for i,raw in enumerate(dl['test']):
   b=move(raw);o=model(b);s=model.forecast_samples(o,8)
   bsz,members,horizon,_=s.shape
   cs128=torch.nn.functional.interpolate(b['field_cs'].reshape(bsz*horizon,1,120,120),size=(128,128),mode='bilinear',align_corners=False).reshape(bsz,horizon,16384)
   f128=s[...,:16384]*cs128[:,None]
   f=torch.nn.functional.interpolate(f128.reshape(bsz*members*horizon,1,128,128),size=(120,120),mode='area').reshape(bsz,members,horizon,14400)
   p=s[...,16384]*b['point_cs'][:,None]
   det128=(o['field_csi']*cs128).reshape(bsz,horizon,128,128)
   det120=torch.nn.functional.interpolate(det128.reshape(bsz*horizon,1,128,128),size=(120,120),mode='area').reshape(bsz,horizon,14400)
   arrays={k:raw[k].numpy() for k in ['field_ghi','point_ghi','field_mask','point_mask','region','origin']}
   arrays.update(field_ensemble_native120=f.cpu().numpy(),field_prediction_128_mean=f128.mean(1).reshape(bsz,horizon,128,128).cpu().numpy().astype(np.float16),
                 point_ensemble=p.cpu().numpy(),field_deterministic_native120=det120.cpu().numpy(),
                 field_deterministic_128=det128.cpu().numpy().astype(np.float16),point_deterministic=(o['point_csi']*b['point_cs']).cpu().numpy())
   np.savez_compressed(archive/f'{i:05d}.npz',**arrays)
   for scope,ens in [('field',f),('point',p)]:
    y=b[scope+'_ghi'];m=b[scope+'_mask'];err=ens.mean(1)-y;crps=(ens-y[:,None]).abs().mean(1)-sum((ens[:,j]-ens[:,k]).abs() for j in range(8) for k in range(8))/128
    acc[scope]+=np.array([float(err[m].abs().sum()),float(err[m].square().sum()),float(crps[m].sum()),int(m.sum())])
 metrics={s:dict(MAE=float(v[0]/v[3]),RMSE=float((v[1]/v[3])**.5),CRPS=float(v[2]/v[3]),n=int(v[3]),BSS=None) for s,v in acc.items()}
 (out/'metrics.json').write_text(json.dumps(metrics,indent=2));record(dict(event='complete',metrics=metrics))
if __name__=='__main__':main()

