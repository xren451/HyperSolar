"""Synthetic shape/gradient smoke check for V9CloudStateHG (not a benchmark)."""
import torch
from v9_cloudstate_hypergraph import V9CloudStateHG

torch.manual_seed(0)
model = V9CloudStateHG().cpu().train()
b, t, h = 1, 72, 24
batch = {
    "sat": torch.randn(b, t, 6, 16, 16),
    "nwp": torch.randn(b, t, 9, 8, 8),
    "time_hist": torch.randn(b, t, 6),
    "time_future": torch.randn(b, h, 6),
    "point_hist": torch.randn(b, t),
    "point_hist_mask": torch.ones(b, t, dtype=torch.bool),
    "position": torch.tensor([[0.4, 0.6]]),
}
out = model(batch)
assert out["field_csi"].shape == (b, h, 256)
assert out["point_csi"].shape == (b, h)
assert out["cloud_field"].shape == (b, h, 256, 64)
targets = {
    "field_csi": torch.rand(b, h, 256),
    "point_csi": torch.rand(b, h),
    "field_csi_mask": torch.ones(b, h, 256, dtype=torch.bool),
    "point_csi_mask": torch.ones(b, h, dtype=torch.bool),
    "field_cs": torch.rand(b, h, 256) * 900,
    "point_cs": torch.rand(b, h) * 900,
    "field_ghi": torch.rand(b, h, 256) * 800,
    "point_ghi": torch.rand(b, h) * 800,
    "field_mask": torch.ones(b, h, 256, dtype=torch.bool),
    "point_mask": torch.ones(b, h, dtype=torch.bool),
}
losses = model.joint_loss(out, targets)
losses["loss"].backward()
assert torch.isfinite(losses["loss"])
assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
model.eval()
with torch.no_grad():
    infer = model(batch)
    samples = model.forecast_samples(infer, members=2)
assert samples.shape == (b, 2, h, 257)
assert torch.isfinite(samples).all()
print({"forward": "ok", "backward": "ok", "diffusion_samples": list(samples.shape),
       "loss": float(losses["loss"].detach())})
