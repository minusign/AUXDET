"""ALCNet DLC algebra and optional conditional scale fusion for AuxDet.

Direction products/minimum follow open-alcnet model/contrast.py cal_pcm.
Replicate boundaries deliberately replace the reference's circular shifts.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _shift_replicate(x: Tensor, dy: int, dx: int) -> Tensor:
    """Sample x[..., y+dy, x+dx], clamping coordinates to the same image."""
    py, px = abs(dy), abs(dx)
    padded = F.pad(x, (px, px, py, py), mode='replicate')
    return padded[..., py + dy:py + dy + x.shape[-2],
                  px + dx:px + dx + x.shape[-1]]


def directional_dlc(x: Tensor, dilation: int) -> Tensor:
    """Minimum of four opposite-neighbor difference products (no ReLU).

    Paper Eq.(2) and official cal_pcm use min; Eq.(4) prints max. We use
    Eq.(1)/(2) and the executable reference, without interchanging operators.
    Half/bfloat16 products are promoted to float32 to avoid squaring overflow.
    """
    if x.ndim != 4 or not x.is_floating_point():
        raise ValueError('DLC expects a floating tensor [B,C,H,W]')
    if not isinstance(dilation, int) or isinstance(dilation, bool) or dilation <= 0:
        raise ValueError('dilation must be a positive integer')
    source = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    products = []
    for dy, dx in ((-dilation, -dilation), (-dilation, 0),
                   (-dilation, dilation), (0, -dilation)):
        negative = _shift_replicate(source, dy, dx)
        positive = _shift_replicate(source, -dy, -dx)
        products.append((source - negative) * (source - positive))
    return torch.stack(products, dim=2).amin(dim=2)


class LocalContrastMLC(nn.Module):
    """Shared DLC/residual wrapper with max or conditional scale fusion.

    visual_metadata uses one FiLM condition projection: [gamma,beta,bias].
    Both conditional modes share the local visual descriptor and logit head.
    Metadata is pure view/band/size encoding, not all_aux_fea.
    Statistics/maps are opt-in and retain only the most recent detached batch.
    """

    def __init__(self, channels: int, dilations: Sequence[int] = (1, 2, 3),
                 mode: str = 'original', metadata_dim: int = 96,
                 hidden_dim: int = 64, lambda_init: float = 0.1,
                 norm_groups: int = 32, collect_stats: bool = False,
                 store_weight_maps: bool = False) -> None:
        super().__init__()
        if mode not in ('original', 'visual', 'visual_metadata'):
            raise ValueError('Unknown MLC fusion mode')
        if min(channels, hidden_dim, metadata_dim, norm_groups) <= 0:
            raise ValueError('MLC channel dimensions and norm_groups must be positive')
        self.dilations = tuple(dilations)
        if (not self.dilations or any(not isinstance(d, int) or isinstance(d, bool)
                                     or d <= 0 for d in self.dilations)
                or len(set(self.dilations)) != len(self.dilations)):
            raise ValueError('dilations must be distinct positive integers')
        if not math.isfinite(lambda_init) or lambda_init <= 0:
            raise ValueError('lambda_init must be finite and positive')
        self.channels, self.metadata_dim, self.hidden_dim = channels, metadata_dim, hidden_dim
        self.mode = mode
        self.collect_stats, self.store_weight_maps = collect_stats, store_weight_maps
        self.visual_descriptor = self.visual_logits = self.metadata_condition = None
        if mode != 'original':
            self.visual_descriptor = nn.Sequential(
                nn.Conv2d(channels, hidden_dim, 1), nn.SiLU())
            self.visual_logits = nn.Conv2d(hidden_dim, len(self.dilations), 1)
        if mode == 'visual_metadata':
            self.metadata_condition = nn.Linear(metadata_dim,
                                                2 * hidden_dim + len(self.dilations))
            # Default AuxFPN init_cfg initializes Conv2d only, leaving these
            # Linear zeros intact. Runner checkpoint loading subsequently wins.
            nn.init.zeros_(self.metadata_condition.weight)
            nn.init.zeros_(self.metadata_condition.bias)
        groups = math.gcd(channels, norm_groups)
        self.residual_norm = nn.GroupNorm(groups, channels)
        self.residual_projection = nn.Conv2d(channels, channels, 1, bias=False)
        raw = lambda_init + math.log(-math.expm1(-lambda_init))
        self.lambda_raw = nn.Parameter(torch.tensor(raw, dtype=torch.float32))
        self._last_scale_stats = self._last_weight_maps = None

    def scale_weights(self, x: Tensor, metadata: Tensor | None = None) -> Tensor:
        """Return [B,K,H,W] softmax weights; max mode has no scalar weights."""
        if self.mode == 'original':
            raise ValueError('Original MLC uses channelwise max, not softmax weights')
        # A fixed local 3x3 average supplies neighborhood context to a light
        # shared pointwise descriptor. No BatchNorm or Dropout is introduced.
        pooled = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), 3, stride=1)
        local = self.visual_descriptor(pooled)
        logits = self.visual_logits(local)
        if self.mode == 'visual_metadata':
            if metadata is None or metadata.shape != (x.shape[0], self.metadata_dim):
                raise ValueError('visual_metadata requires matching pure metadata [B,M]')
            gamma, beta, bias = torch.split(
                self.metadata_condition(metadata),
                (self.hidden_dim, self.hidden_dim, len(self.dilations)), dim=1)
            # A = VisualLogit(local) + MetaBias + Interaction. Reusing the
            # same logit head makes this exactly FiLM(local), with gamma/beta
            # bounded for numerical stability. Its constant conv bias cancels.
            interaction = local * gamma.tanh()[..., None, None] + beta.tanh()[..., None, None]
            interaction_logits = F.conv2d(interaction, self.visual_logits.weight)
            logits = logits + interaction_logits + bias[..., None, None]
        return logits.float().softmax(dim=1)

    def contrast(self, x: Tensor, metadata: Tensor | None = None):
        responses = torch.stack([directional_dlc(x, d) for d in self.dilations], dim=1)
        if self.mode == 'original':
            return responses.amax(dim=1), None
        weights = self.scale_weights(x, metadata)
        return (responses * weights.unsqueeze(2)).sum(dim=1), weights

    def forward(self, x: Tensor, metadata: Tensor | None = None) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError('MLC input must match configured [B,C,H,W]')
        fused, weights = self.contrast(x, metadata)
        # Normalize products before casting back to the feature dtype; all
        # three modes share this normalization/projection/residual envelope.
        normalized = self.residual_norm(fused)
        residual = self.residual_projection(normalized.to(dtype=x.dtype))
        self._last_scale_stats = self._last_weight_maps = None
        if self.collect_stats:
            self._record_stats(weights, x.shape[0])
        if self.store_weight_maps and weights is not None:
            self._last_weight_maps = weights.detach().float().cpu()
        scale = F.softplus(self.lambda_raw).to(dtype=x.dtype)
        return x + scale * residual

    def _record_stats(self, weights, batch_size):
        if weights is None:
            self._last_scale_stats = [dict(fusion='max', dilations=list(self.dilations),
                                          mean=None, entropy=None)
                                      for _ in range(batch_size)]
            return
        detached = weights.detach().float()
        means = detached.mean(dim=(2, 3)).cpu().tolist()
        entropy = -(detached * detached.clamp_min(1e-8).log()).sum(dim=1)
        entropies = entropy.mean(dim=(1, 2)).cpu().tolist()
        self._last_scale_stats = [dict(fusion='softmax_weighted_sum',
                                      dilations=list(self.dilations), mean=mean,
                                      entropy=ent)
                                 for mean, ent in zip(means, entropies)]

    def get_scale_stats(self):
        """Detached per-image spatial means/entropy; original has no weights."""
        import copy
        return copy.deepcopy(self._last_scale_stats)

    def get_weight_maps(self):
        """Optional detached CPU [B,K,H,W] maps from the latest forward."""
        return None if self._last_weight_maps is None else self._last_weight_maps.clone()
