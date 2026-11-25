#!/usr/bin/env python3
"""Recover test evaluation for a trained V3-V5 encoder checkpoint."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_solarcube_v7_fieldpoint_csi import SolarCubeWindows
from train_solarcube_encoders_csi import EncoderCSI,VARIANTS

def metric_rows(pred,y,mask):
    pred=np.asarray(pred);y=np.asarray(y);mask=np.asarray(mask,dtype=bool);rows=[]
    for h in range(y.shape[1]):
        m=mask[:,h]&np.isfinite(y[:,h])&np.isfinite(pred[:,h])
        e=pred[:,h][m]-y[:,h][m]
        if e.size:rows.append({'lead_hour':h+1,'n':int(e.size),'MAE':float(np.mean(np.abs(e))),'RMSE':float(np.sqrt(np.mean(e*e))),'CRPS':float(np.mean(np.abs(e))),'PI90':'not_applicable_deterministic_encoder_screen'})
    return rows
def mean_rows(rows):return {k:float(np.mean([x[k] for x in rows])) for k in ('MAE','RMSE','CRPS')}
def main():
    ap=argparse.ArgumentParser()
    for name in ('variant','data','prepared','sitelist','nwp','out'):ap.add_argument('--'+name,required=True)
    ap.add_argument('--batch',type=int,default=4);ap.add_argument('--stride',type=int,default=3);ap.add_argument('--device',default='cuda')
    a=ap.parse_args();out=Path(a.out);dev=torch.device(a.device);torch.set_num_threads(4)
    ds=SolarCubeWindows(a.data,a.prepared,a.sitelist,a.nwp,'test',stride=a.stride)
    dl=DataLoader(ds,batch_size=a.batch,shuffle=False,num_workers=0,pin_memory=dev.type=='cuda')
    model=EncoderCSI(a.variant).to(dev);model.load_state_dict(torch.load(out/'best.pt',map_location=dev,weights_only=True));model.eval()
    preds={k:[] for k in ('fg','pg','fc','pc','fy','py','fm','pm','fct','pct','fcm','pcm','reg','origin')};start=time.time()
    with torch.no_grad():
        for b0 in dl:
            b={k:v.to(dev,non_blocking=True) if torch.is_tensor(v) else v for k,v in b0.items()}
            fc,pc=model(b);fg=fc*b['field_cs'];pg=pc*b['point_cs']
            for k,v in [('fg',fg),('pg',pg),('fc',fc),('pc',pc),('fy',b0['field_ghi']),('py',b0['point_ghi']),('fm',b0['field_mask']),('pm',b0['point_mask']),('fct',b0['field_csi']),('pct',b0['point_csi']),('fcm',b0['field_csi_mask']),('pcm',b0['point_csi_mask']),('reg',b0['region']),('origin',b0['origin'])]:preds[k].append(v.detach().cpu().numpy())
    ar={k:np.concatenate(v) for k,v in preds.items()}
    hist=[]
    for line in (out/'train.log').read_text(errors='replace').splitlines():
        try:
            q=json.loads(line)
            if 'epoch' in q:hist.append(q)
        except Exception:pass
    best=min(hist,key=lambda x:x['val_joint_loss'])
    f=metric_rows(ar['fg'],ar['fy'],ar['fm']);p=metric_rows(ar['pg'],ar['py'],ar['pm']);fc=metric_rows(ar['fc'],ar['fct'],ar['fcm']);pc=metric_rows(ar['pc'],ar['pct'],ar['pcm'])
    eval_peak=torch.cuda.max_memory_allocated(dev)/2**20 if dev.type=='cuda' else 0
    train_peak=max((float(x.get('peak_gpu_allocated_MiB',0)) for x in hist),default=0)
    result={'version':a.variant,'encoder':VARIANTS[a.variant],'dataset':'SolarCube 2018 all19/full-year','protocol':'chronological 70/15/15; history72; horizon24; stride3','best_epoch':best['epoch'],'epochs_ran':len(hist),'best_val_joint_loss':best['val_joint_loss'],'training_elapsed_sec':hist[-1]['elapsed_sec'],'test_evaluation_sec':time.time()-start,'test_windows':len(ar['origin']),'parameters':sum(x.numel() for x in model.parameters()),'peak_gpu_allocated_MiB':max(train_peak,eval_peak),'training_peak_gpu_allocated_MiB':train_peak,'test_evaluation_peak_gpu_allocated_MiB':eval_peak,'field_GHI_mean24':mean_rows(f),'point_GHI_mean24':mean_rows(p),'field_CSI_mean24':mean_rows(fc),'point_CSI_mean24':mean_rows(pc),'field_GHI_by_lead':f,'point_GHI_by_lead':p,'field_CSI_by_lead':fc,'point_CSI_by_lead':pc,'evaluation_note':'Training reached its early-stopping endpoint and best.pt was restored. Test metrics/predictions recovered after fixing a NumPy boolean-mask adapter bug; no training was repeated. Deterministic CRPS=MAE; prediction intervals are not applicable.'}
    np.savez_compressed(out/'test_predictions.npz',field_ghi_prediction=ar['fg'],field_ghi_truth=ar['fy'],field_mask=ar['fm'],point_ghi_prediction=ar['pg'],point_ghi_truth=ar['py'],point_mask=ar['pm'],field_csi_prediction=ar['fc'],field_csi_truth=ar['fct'],field_csi_mask=ar['fcm'],point_csi_prediction=ar['pc'],point_csi_truth=ar['pct'],point_csi_mask=ar['pcm'],region=ar['reg'],origin=ar['origin'])
    (out/'metrics.json').write_text(json.dumps(result,indent=2));(out/'training_history.json').write_text(json.dumps(hist,indent=2));(out/'DONE.json').write_text(json.dumps({'status':'evaluation_recovered','best_epoch':best['epoch'],'test_evaluation_sec':result['test_evaluation_sec']},indent=2));(out/'README.md').write_text('# V3 Vision-Swin SolarCube CSI field/point\n\n'+json.dumps(result,indent=2)+'\n\nFull test truths/predictions, metrics, training history, best checkpoint, config and logs are retained here.\n')
    print(json.dumps({k:result[k] for k in ('version','best_epoch','epochs_ran','training_elapsed_sec','test_evaluation_sec','test_windows','peak_gpu_allocated_MiB','field_GHI_mean24','point_GHI_mean24','field_CSI_mean24','point_CSI_mean24')},indent=2))
if __name__=='__main__':main()
