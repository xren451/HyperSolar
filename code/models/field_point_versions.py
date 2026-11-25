"""Data-faithful station-supported field + selected-point model adaptations.

These datasets provide irregular station vectors, not a dense gridded GHI
label. The `field` head therefore forecasts all station locations jointly;
the point head separately decodes one selected station while conditioning on
its sampled station-field prediction and the shared spatial context.
"""
import torch
from torch import nn

from models import FlatResidualDDPM, SMFDeterministic


class FieldPointBase(nn.Module):
    def __init__(self, features, sites, coords, selected_site=0, horizon=24,
                 width=64, heads=4, ablation="full"):
        super().__init__()
        if not 0 <= selected_site < sites:
            raise ValueError(f"selected site {selected_site} outside [0, {sites})")
        self.sites, self.selected_site, self.horizon = sites, selected_site, horizon
        self.ablation = ablation
        self.use_field_task = ablation != "point_only"
        self.use_point_task = ablation != "field_only"
        self.det = SMFDeterministic(
            features, sites, coords, horizon=horizon, width=width, heads=heads,
            use_gnn=ablation != "no_gnn",
            use_spatial_transformer=ablation not in ("no_spatial_transformer", "no_transformer"),
            use_temporal_transformer=ablation not in ("no_temporal_transformer", "no_transformer"),
        )
        self.point_decoder = nn.Sequential(
            nn.Linear(2, width), nn.GELU(), nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 1)
        )
        self.use_diffusion = ablation != "no_diffusion"
        self.diff = FlatResidualDDPM(horizon * sites, steps=32)

    def point_from_field(self, field):
        # field is [..., lead, site]; point decoder sees sampled selected-site
        # value plus the all-station spatial context at that lead.
        sampled = field[..., self.selected_site]
        regional = field.mean(dim=-1)
        return sampled + self.point_decoder(torch.stack([sampled, regional], dim=-1))[..., 0]

    def _mask_for_task(self, mask):
        if self.ablation == "point_only":
            keep = torch.zeros(self.sites, dtype=torch.bool, device=mask.device)
            keep[self.selected_site] = True
            return mask & keep
        return mask

    def _task_losses(self, field, b):
        target_mask = b["target_mask"].bool()
        field_loss = (field - b["y"]).square()[target_mask].mean() if self.use_field_task else field.sum() * 0
        point = self.point_from_field(field)
        j = self.selected_site
        pmask = target_mask[..., j] if self.use_point_task else torch.zeros_like(target_mask[..., j])
        point_loss = (point - b["y"][..., j]).square()[pmask].mean() if pmask.any() else field.sum() * 0
        return field_loss, point_loss, point


class Version1FieldPoint(FieldPointBase):
    """Shared SMF station-field forecast, point decoder, GHI residual diffusion."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss(self, b):
        field = self.det(b["x"], b["time_hist"], b["time_future"])
        field_loss, point_loss, _ = self._task_losses(field, b)
        mask = self._mask_for_task(b["target_mask"].bool())
        residual_loss = self.diff.loss(b["y"] - field.detach(), mask) if self.use_diffusion else field_loss * 0
        total = field_loss + point_loss + 0.05 * residual_loss
        return total, {"field": field_loss, "point": point_loss, "diffusion": residual_loss}

    @torch.no_grad()
    def forecast(self, b, members=16):
        field_mean = self.det(b["x"], b["time_hist"], b["time_future"])
        if self.use_diffusion:
            residual = self.diff.sample(len(field_mean), field_mean.device, field_mean.shape, members)
            field_samples = field_mean[:, None] + residual
        else:
            field_samples = field_mean[:, None].expand(-1, members, -1, -1)
        point_samples = self.point_from_field(field_samples)
        return field_samples, point_samples, field_mean, self.point_from_field(field_mean)


class Version2FieldPoint(FieldPointBase):
    """Intermediate radiative-attenuation state, state residual diffusion,
    then separate station-field and selected-point target decoders.
    """
    def __init__(self, features, sites, coords, selected_site=0, horizon=24,
                 width=64, heads=4, ablation="full"):
        super().__init__(features, sites, coords, selected_site, horizon, width, heads, ablation)
        self.use_cloud_state = ablation != "no_cloud_state"
        self.use_state_supervision = ablation != "no_state_supervision"
        self.use_diffusion = ablation not in ("no_diffusion", "no_state_diffusion")
        self.state_head = nn.Sequential(nn.Linear(width + 1, width), nn.GELU(), nn.Linear(width, 1))
        self.field_decoder = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 1))
        self.point_state_decoder = nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 1))

    def deterministic(self, b):
        z = self.det.encode(b["x"], b["time_hist"])
        base = self.det(b["x"], b["time_hist"], b["time_future"])
        if not self.use_cloud_state:
            state = base
        else:
            lead = self.det.lead.weight[None, :, None, :] + self.det.clock(b["time_future"])[:, :, None]
            state = self.state_head(torch.cat([z[:, None].expand(-1, self.horizon, -1, -1) + lead,
                                                base[..., None]], dim=-1))[..., 0]
        return base, state, z

    def decode_field(self, state, z, b):
        if not self.use_cloud_state:
            return state
        local = z[:, None].expand(-1, state.shape[1], -1, -1).mean(-1)
        clock = b["time_future"][:, :, None, 0].expand_as(state)
        return self.field_decoder(torch.stack([state, local, clock], dim=-1))[..., 0]

    def decode_point(self, state, z, b):
        j = self.selected_site
        state_at_site = state[..., j]
        spatial = state.mean(dim=-1)
        z_point = z[:, j, :].mean(-1)[:, None].expand_as(state_at_site)
        point_input = torch.stack([state_at_site + z_point, spatial], dim=-1)
        return self.point_state_decoder(point_input)[..., 0]

    def loss(self, b):
        _, state, z = self.deterministic(b)
        field = self.decode_field(state, z, b)
        target_mask = b["target_mask"].bool()
        field_loss = (field - b["y"]).square()[target_mask].mean() if self.use_field_task else field.sum() * 0
        point = self.decode_point(state, z, b)
        j = self.selected_site
        point_mask = target_mask[..., j] if self.use_point_task else torch.zeros_like(target_mask[..., j])
        point_loss = (point - b["y"][..., j]).square()[point_mask].mean() if point_mask.any() else field.sum() * 0
        state_mask = b["state_mask"].bool()
        if self.ablation == "point_only":
            keep = torch.zeros(self.sites, dtype=torch.bool, device=state_mask.device)
            keep[self.selected_site] = True
            state_mask = state_mask & keep
        if self.use_cloud_state and self.use_state_supervision:
            sl = (state - b["state"]).square()[state_mask].mean() if state_mask.any() else state.sum() * 0
        else:
            sl = state.sum() * 0
        source = b["state"] if self.use_cloud_state else b["y"]
        source_mask = state_mask if self.use_cloud_state else self._mask_for_task(b["target_mask"].bool())
        diffusion = self.diff.loss(source - state.detach(), source_mask) if self.use_diffusion else field_loss * 0
        total = field_loss + point_loss + sl + 0.05 * diffusion
        return total, {"field": field_loss, "point": point_loss, "state": sl, "diffusion": diffusion}

    @torch.no_grad()
    def forecast(self, b, members=16):
        _, state, z = self.deterministic(b)
        if self.use_diffusion:
            residual = self.diff.sample(len(state), state.device, state.shape, members)
            state_samples = state[:, None] + residual
        else:
            state_samples = state[:, None].expand(-1, members, -1, -1)
        fields, points = [], []
        for member in range(members):
            state_member = state_samples[:, member]
            fields.append(self.decode_field(state_member, z, b))
            points.append(self.decode_point(state_member, z, b))
        field_samples = torch.stack(fields, dim=1)
        point_samples = torch.stack(points, dim=1)
        field_mean = self.decode_field(state, z, b)
        point_mean = self.decode_point(state, z, b)
        return field_samples, point_samples, field_mean, point_mean
