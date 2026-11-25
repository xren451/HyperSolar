import torch
from train_solarcube_encoders_csi import EncoderCSI

torch.set_num_threads(2)
for version in ("V3", "V4", "V5"):
    model = EncoderCSI(version).cuda().train()
    b = {
        "sat": torch.randn(1,72,6,16,16,device="cuda"),
        "nwp": torch.randn(1,72,9,8,8,device="cuda"),
        "time_hist": torch.randn(1,72,6,device="cuda"),
        "time_future": torch.randn(1,24,6,device="cuda"),
        "point_hist": torch.randn(1,72,device="cuda"),
        "point_hist_mask": torch.ones(1,72,device="cuda"),
        "position": torch.rand(1,2,device="cuda"),
    }
    fc, pc = model(b)
    loss = fc.square().mean() + pc.square().mean()
    loss.backward()
    print(version, tuple(fc.shape), tuple(pc.shape), float(loss), sum(p.numel() for p in model.parameters()))
    del model, b, fc, pc, loss
    torch.cuda.empty_cache()
