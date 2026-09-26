"""Spiral-aware feature sampling for AuxDet.

Algorithmic adaptation of the SFS module described in:
https://github.com/mengduann/NS-FPN

This implementation uses MMDetection/MMCV's existing low-level multi-scale
Deformable Attention operator and keeps the original SFS sampling semantics:
query-conditioned attention weights, fixed spiral offsets, and a shared
learnable offset residual.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

try:
    from mmcv.ops.multi_scale_deform_attn import (
        MultiScaleDeformableAttnFunction,
    )
except ImportError as exc:  # pragma: no cover - depends on runtime MMCV
    MultiScaleDeformableAttnFunction = None
    _MMCV_SFS_IMPORT_ERROR = exc
else:
    _MMCV_SFS_IMPORT_ERROR = None


def generate_structured_grid(
    n_heads: int,
    n_points: int,
    n_levels: int = 1,
    base_radius: float = 1.0,
    radius_step: float = 1.0,
) -> Tensor:
    """Create the fixed spiral sampling pattern."""
    offsets = []
    for head in range(n_heads):
        head_offsets = []
        delta_theta = 2 * math.pi * head / n_heads
        for point in range(n_points):
            theta = 2 * math.pi * point / n_points + delta_theta
            radius = base_radius + point * radius_step
            head_offsets.append(
                [radius * math.cos(theta), radius * math.sin(theta)]
            )
        offsets.append(head_offsets)
    grid = torch.tensor(offsets, dtype=torch.float32)
    return grid.unsqueeze(1).repeat(1, n_levels, 1, 1)


class SpiralAwareCrossDeformAttn2D(nn.Module):
    """Cross-scale spiral feature sampling.

    Args:
        dim: Feature channel count.
        n_heads: Number of attention heads.
        n_points: Sampling points per head.
    """

    def __init__(self, dim: int, n_heads: int = 8, n_points: int = 4) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(
                f'dim must be divisible by n_heads, got {dim} and {n_heads}'
            )
        if MultiScaleDeformableAttnFunction is None:
            raise ImportError(
                'SFS requires MMCV MultiScaleDeformableAttnFunction'
            ) from _MMCV_SFS_IMPORT_ERROR

        self.dim = dim
        self.n_heads = n_heads
        self.n_points = n_points
        self.query_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.key_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.shared_offsets_residual = nn.Parameter(
            torch.zeros(n_heads, n_points, 2)
        )
        fixed_bias = generate_structured_grid(
            n_heads, n_points, n_levels=1
        )
        self.register_buffer(
            'offset_base', fixed_bias.view(1, 1, n_heads, 1, n_points, 2)
        )
        self.query_norm = nn.LayerNorm(dim)
        self.key_norm = nn.LayerNorm(dim)
        self.out_norm = nn.LayerNorm(dim)
        self.attention_weights = nn.Linear(
            dim, n_heads * n_points
        )
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    def forward(self, query_feat: Tensor, key_feat: Tensor) -> Tensor:
        if query_feat.ndim != 4 or key_feat.ndim != 4:
            raise ValueError('SFS expects query/key tensors shaped [B,C,H,W]')
        batch, channels, h_query, w_query = query_feat.shape
        batch_key, channels_key, h_key, w_key = key_feat.shape
        if (batch, channels) != (batch_key, channels_key):
            raise ValueError('SFS query and key batch/channel dimensions differ')

        query_feat = self.query_conv(query_feat)
        key_feat = self.key_conv(key_feat)
        query = self.query_norm(
            query_feat.flatten(2).transpose(1, 2)
        )
        key = self.key_norm(
            key_feat.flatten(2).transpose(1, 2)
        )

        spatial_shapes = query.new_tensor(
            [[h_key, w_key]], dtype=torch.long
        )
        level_start_index = query.new_tensor([0], dtype=torch.long)

        # The fixed spiral and learnable residual are shared by every query
        # location, matching the official SFS implementation.
        shared_offsets = (
            self.offset_base.view(
                self.n_heads, 1, self.n_points, 2
            )
            + self.shared_offsets_residual.view(
                self.n_heads, 1, self.n_points, 2
            )
        )
        sampling_offsets = shared_offsets.view(
            1, 1, self.n_heads, 1, self.n_points, 2
        ).expand(batch, h_query * w_query, -1, -1, -1, -1)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(
                0.5 / h_query,
                1 - 0.5 / h_query,
                h_query,
                device=query_feat.device,
                dtype=query_feat.dtype,
            ),
            torch.linspace(
                0.5 / w_query,
                1 - 0.5 / w_query,
                w_query,
                device=query_feat.device,
                dtype=query_feat.dtype,
            ),
            indexing='ij',
        )
        reference_points = torch.stack((grid_x, grid_y), dim=-1)
        reference_points = reference_points.reshape(
            1, h_query * w_query, 1, 2
        ).expand(batch, -1, -1, -1)

        value = self.value_proj(key).view(
            batch,
            h_key * w_key,
            self.n_heads,
            self.dim // self.n_heads,
        )
        attention_weights = self.attention_weights(query).view(
            batch, h_query * w_query, self.n_heads, self.n_points
        )
        attention_weights = attention_weights.softmax(dim=-1).unsqueeze(3)

        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1
        )
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets
            / offset_normalizer[None, None, None, :, None, :]
        ).contiguous()

        sampled = MultiScaleDeformableAttnFunction.apply(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            64,
        )
        sampled = self.output_proj(sampled)
        out = query + query * sampled
        return self.out_norm(out).transpose(1, 2).reshape(
            batch, channels, h_query, w_query
        )
