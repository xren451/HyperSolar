import torch
from torch.utils.data import DataLoader
from train_solarcube_v7_fieldpoint_csi import SolarCubeWindows
from train_solarcube_encoders_csi import EncoderCSI

ROOT="/mnt/wangyishan_share/wangdongdong/smf_registry_variants_20261105"
DATA=f"{ROOT}/data/SolarCube_2018_full19_hourly"
PREP="/mnt/wangyishan_share/wangdongdong/smf_full_20261025/solarcube/prepared"
SITELIST="/mnt/wangyishan_share/paper_solar_datasets/solarcube/raw/SolarCube_sitelist.csv"
NWP=f"{ROOT}/data/SolarCube_2018_era5_8x8_v7"
torch.set_num_threads(2)
ds=SolarCubeWindows(DATA,PREP,SITELIST,NWP,"train",stride=3)
b=next(iter(DataLoader(ds,batch_size=1,num_workers=0)))
print("batch shapes", {k:tuple(v.shape) for k,v in b.items() if torch.is_tensor(v)})
batch_cpu=b
for v in ("V3","V4","V5"):
    m=EncoderCSI(v).cuda().train()
    b={k:x.cuda() if torch.is_tensor(x) else x for k,x in batch_cpu.items()}
    fc,pc=m(b); fg=fc*b["field_cs"];pg=pc*b["point_cs"]
    loss=((fg-b["field_ghi"]).square()*b["field_mask"]).sum()/b["field_mask"].sum().clamp_min(1)/1e6
    loss=loss+((pg-b["point_ghi"]).square()*b["point_mask"]).sum()/b["point_mask"].sum().clamp_min(1)/1e6
    loss=loss+0.5*( (fc-b["field_csi"]).square()*b["field_csi_mask"] ).sum()/b["field_csi_mask"].sum().clamp_min(1)
    loss=loss+0.5*( (pc-b["point_csi"]).square()*b["point_csi_mask"] ).sum()/b["point_csi_mask"].sum().clamp_min(1)
    loss.backward(); print(v,tuple(fc.shape),tuple(pc.shape),"finite",bool(torch.isfinite(loss)),"loss",float(loss.detach()))
    del m,b,fc,pc,fg,pg,loss;torch.cuda.empty_cache()
