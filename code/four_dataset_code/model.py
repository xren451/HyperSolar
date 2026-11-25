"""Explicit TemporalRamp extension; do not relabel archived V9 checkpoints.

The non-autoregressive future decoder attends to observed point history only.
Future irradiance is consumed exclusively by joint_loss, never forward.
"""
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent/'original'))
from v9_ramp_hyperedge import V9RampHyperedgeCSI as ArchivedV9


class V9TemporalRamp(ArchivedV9):
    VERSION = 'V9-RAMP-DIFFUSION-TemporalRamp-v1'

    def __init__(self, width=64, heads=4, diffusion_steps=8):
        super().__init__(width, heads, diffusion_steps)
        self.observed_history = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))
        self.observed_position = nn.Embedding(72, width)
        self.lead_position = nn.Embedding(24, width)
        layer = nn.TransformerDecoderLayer(width, heads, 2*width, dropout=.1,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.horizon_decoder = nn.TransformerDecoder(layer, 2)
        self.ramp_residual = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                          nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.ramp_residual[-1].weight)
        nn.init.zeros_(self.ramp_residual[-1].bias)
        self.ramp_class = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 3))

    def forward(self, batch):
        out = super().forward(batch)
        h = batch['point_hist']
        valid = batch['point_hist_mask'].bool()
        change = F.pad(h[:, 1:]-h[:, :-1], (1, 0))
        adjacent = valid & F.pad(valid[:, :-1], (1, 0), value=False)
        features = torch.stack([h, torch.where(adjacent, change, 0), valid.float(), adjacent.float()], -1)
        memory = self.observed_history(features)
        memory = memory + self.observed_position(torch.arange(72, device=h.device))[None]
        query = out['cloud_point'] + self.lead_position(torch.arange(24, device=h.device))[None]
        # Even an entirely missing history has finite availability embeddings.
        # Avoid an all-masked attention row; valid flags remain explicit inputs.
        memory_padding = ~valid
        memory_padding = memory_padding.clone()
        memory_padding[~valid.any(1), -1] = False
        decoded = self.horizon_decoder(query, memory, memory_key_padding_mask=memory_padding)
        residual = .75*torch.tanh(self.ramp_residual(decoded)[..., 0])
        point = (out['point_csi']+residual).clamp(0, 2.5)
        out['base_point_csi'] = out['point_csi']
        out['point_csi'] = point
        out['cloud_point'] = decoded
        out['ramp_logits'] = self.ramp_class(decoded)
        out['diffusion_cond'] = torch.cat([out['cloud_field'].mean((1, 2)), decoded.mean(1),
                                            out['field_csi'].mean((1, 2))[:, None],
                                            point.mean(1, keepdim=True)], -1)
        return out

    def joint_loss(self, output, batch, diffusion_weight=.1):
        losses = super().joint_loss(output, batch, diffusion_weight)
        pred = output['point_csi']*batch['point_cs']
        truth = batch['point_ghi']
        valid = batch['point_mask'].bool()
        # Recover the last observed physical irradiance with train-only stats.
        last = batch['point_hist'][:, -1]*batch['point_std']+batch['point_mean']
        last_mask = batch['point_hist_mask'][:, -1].bool()
        pred_delta = torch.diff(torch.cat([last[:, None], pred], 1), dim=1)
        true_delta = torch.diff(torch.cat([last[:, None], truth], 1), dim=1)
        pair = valid & torch.cat([last_mask[:, None], valid[:, :-1]], 1)
        daytime = (batch['point_cs'] > 20)
        pair = pair & daytime
        ramp = true_delta.abs() >= 100.
        weights = pair.float()*(1.+2.*ramp.float())
        delta = F.smooth_l1_loss(pred_delta/1000., true_delta/1000., reduction='none', beta=.05)
        delta_loss = (delta*weights).sum()/weights.sum().clamp_min(1)
        pc = torch.diff(pred_delta, dim=1)
        tc = torch.diff(true_delta, dim=1)
        cm = pair[:, 1:] & pair[:, :-1]
        curvature = F.smooth_l1_loss(pc/1000., tc/1000., reduction='none', beta=.05)
        curvature_loss = (curvature*cm).sum()/cm.sum().clamp_min(1)
        classes = torch.where(true_delta > 100, 2, torch.where(true_delta < -100, 0, 1))
        classification = F.cross_entropy(output['ramp_logits'].transpose(1, 2), classes, reduction='none')
        event_weights = pair.float()*(1.+2.*ramp.float())
        class_loss = (classification*event_weights).sum()/event_weights.sum().clamp_min(1)
        extra = .5*delta_loss+.25*curvature_loss+.02*class_loss
        losses['base_deterministic'] = losses['deterministic']
        losses['deterministic'] = losses['deterministic']+extra
        losses['loss'] = losses['loss']+extra
        losses.update(point_delta=delta_loss, point_curvature=curvature_loss, point_event=class_loss)
        return losses
