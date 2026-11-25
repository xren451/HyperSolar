#!/usr/bin/env python3
"""Same-window SolarCube V7 vs deterministic Haurwitz clear-sky baseline."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def cs_from_sza(sza):
    c=np.maximum(np.cos(np.deg2rad(sza)),0.0)
    return 1098.0*c*np.exp(-0.059/np.maximum(c,1e-4))

def metric(yhat,y,mask):
    rows=[]
    for h in range(y.shape[1]):
        m=mask[:,h].astype(bool)&np.isfinite(yhat[:,h])&np.isfinite(y[:,h])
        e=yhat[:,h][m]-y[:,h][m]
        if e.size: rows.append({"lead_hour":h+1,"n":int(e.size),"MAE":float(np.mean(np.abs(e))),"RMSE":float(np.sqrt(np.mean(e*e)))})
    return rows

def avg(rows,keys=("MAE","RMSE")):
    return {k:float(np.mean([r[k] for r in rows])) for k in keys if rows and k in rows[0]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--pred',required=True);p.add_argument('--data',required=True);p.add_argument('--sitelist',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    z=np.load(a.pred,allow_pickle=True)
    region=z['region'].astype(int); origin=z['origin'].astype(int); n=len(origin); leads=np.arange(1,25,dtype=np.int32)
    field_cs=np.empty((n,24,256),np.float32);point_cs=np.empty((n,24),np.float32)
    with open(a.sitelist,newline='',encoding='utf-8-sig') as f: sites={int(x['tile_id']):x for x in csv.DictReader(f)}
    root=Path(a.data)
    for r in np.unique(region):
        ii=np.flatnonzero(region==r);feat=np.load(root/f'region_{r:02d}'/'features16.npy',mmap_mode='r')
        tt=origin[ii,None]+leads[None,:]
        sza=np.asarray(feat[tt,:,:,0],np.float32)
        field_cs[ii]=cs_from_sza(sza).reshape(len(ii),24,256)
        site=sites[int(r)];rf=(float(site['lat_ulcnr'])-float(site['lats']))/(float(site['lat_ulcnr'])-float(site['lat_lrcnr']));cf=(float(site['lons'])-float(site['lon_ulcnr']))/(float(site['lon_lrcnr'])-float(site['lon_ulcnr']))
        y=rf*15;x=cf*15;y0=int(np.floor(y));x0=int(np.floor(x));y1=min(y0+1,15);x1=min(x0+1,15);wy=y-y0;wx=x-x0
        psza=(1-wy)*(1-wx)*sza[:,:,y0,x0]+(1-wy)*wx*sza[:,:,y0,x1]+wy*(1-wx)*sza[:,:,y1,x0]+wy*wx*sza[:,:,y1,x1]
        point_cs[ii]=cs_from_sza(psza)
    fm=z['field_mask'].astype(bool);pm=z['point_mask'].astype(bool);fy=z['field_ghi_truth'];py=z['point_ghi_truth']
    fmean=z['field_ghi_ensemble'].mean(axis=1);pmean=z['point_ghi_ensemble'].mean(axis=1)
    # Daylight-valid CSI uses the same solar-zenith denominator threshold as training.
    fcsi_mask=fm&(field_cs>20);pcsi_mask=pm&(point_cs>20)
    f_csi_true=np.divide(fy,field_cs,out=np.zeros_like(fy),where=field_cs>20).clip(0,2.5)
    p_csi_true=np.divide(py,point_cs,out=np.zeros_like(py),where=point_cs>20).clip(0,2.5)
    fv7=metric(fmean,fy,fm);fclear=metric(field_cs,fy,fm);pv7=metric(pmean,py,pm);pclear=metric(point_cs,py,pm)
    f_csi_pred=z['field_csi_ensemble'].mean(axis=1);p_csi_pred=z['point_csi_ensemble'].mean(axis=1)
    fc_v7=metric(f_csi_pred,f_csi_true,fcsi_mask);fc_clear=metric(np.ones_like(f_csi_true),f_csi_true,fcsi_mask)
    pc_v7=metric(p_csi_pred,p_csi_true,pcsi_mask);pc_clear=metric(np.ones_like(p_csi_true),p_csi_true,pcsi_mask)
    out={"dataset":"SolarCube 2018 full19","protocol":"exact V7 test origins; chronological 70/15/15; 72h history; 24h horizon; stride3","clear_sky":"Haurwitz; GHI_clear=1098*cos(SZA)*exp(-0.059/cos(SZA)); identical test masks; CSI=1 baseline evaluated only where GHI_clear>20 W/m2","test_origins":n,"field_GHI":{"V7":avg(fv7),"Haurwitz_clear_sky":avg(fclear),"V7_by_lead":fv7,"clear_sky_by_lead":fclear},"point_GHI":{"V7":avg(pv7),"Haurwitz_clear_sky":avg(pclear),"V7_by_lead":pv7,"clear_sky_by_lead":pclear},"daylight_CSI":{"field_V7":avg(fc_v7),"field_CSI1":avg(fc_clear),"point_V7":avg(pc_v7),"point_CSI1":avg(pc_clear),"field_V7_by_lead":fc_v7,"field_CSI1_by_lead":fc_clear,"point_V7_by_lead":pc_v7,"point_CSI1_by_lead":pc_clear},"decision_note":"This is test-set baseline comparison only; any V7+clear-sky gate must fit on validation forecasts, never on these test labels."}
    Path(a.out).write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ('field_GHI','point_GHI','daylight_CSI')},indent=2));print('FIELD',out['field_GHI']['V7'],out['field_GHI']['Haurwitz_clear_sky']);print('POINT',out['point_GHI']['V7'],out['point_GHI']['Haurwitz_clear_sky']);print('CSI',out['daylight_CSI']['field_V7'],out['daylight_CSI']['field_CSI1'],out['daylight_CSI']['point_V7'],out['daylight_CSI']['point_CSI1'])
if __name__=='__main__':main()
