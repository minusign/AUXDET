"""Metadata-driven dense mixture-of-experts layers.

This module is intentionally independent from AuxFPN.  It provides the
router, expert, and dense soft-MoE building blocks that can be integrated into
a neck at a later stage without changing the existing detector interface.
"""

import torch
from torch import Tensor, nn

from mmdet.registry import MODELS


@MODELS.register_module()
class MetadataRouter(nn.Module):
    """Generate per-sample soft routing weights from metadata features.

    Args:
        input_dim: Dimension of the metadata feature. Defaults to 128.
        hidden_dim: Hidden dimension of the router MLP. Defaults to 64.
        num_experts: Number of experts. Defaults to 4.
        temperature: Positive softmax temperature. Defaults to 1.0.
    """

    def __init__(self,
                 input_dim: int = 128,
                 hidden_dim: int = 64,
                 num_experts: int = 4,
                 temperature: float = 1.0) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or num_experts <= 0:
            raise ValueError('Router dimensions and num_experts must be > 0.')
        if temperature <= 0:
            raise ValueError('temperature must be positive.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.temperature = float(temperature)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, aux_feature: Tensor) -> Tensor:
        """Return a probability vector for every sample.

        Args:
            aux_feature: Tensor with shape ``[B, input_dim]``.

        Returns:
            Tensor with shape ``[B, num_experts]``. Rows sum to one.
        """
        if aux_feature.ndim != 2 or aux_feature.size(-1) != self.input_dim:
            raise ValueError(
                f'Expected aux_feature with shape [B, {self.input_dim}], '
                f'got {tuple(aux_feature.shape)}.')
        logits = self.net(aux_feature)
        return torch.softmax(logits / self.temperature, dim=-1)


@MODELS.register_module()
class MetadataExpert(nn.Module):
    """A convolutional expert that preserves feature shape.

    Args:
        feature_channels: Number of input and output channels.
        hidden_channels: Number of expert hidden channels.
    """

    def __init__(self,
                 feature_channels: int = 256,
                 hidden_channels: int = 64) -> None:
        super().__init__()
        if feature_channels <= 0 or hidden_channels <= 0:
            raise ValueError(
                'feature_channels and hidden_channels must be > 0.')

        self.feature_channels = feature_channels
        self.hidden_channels = hidden_channels
        self.net = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, feature_channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Process ``x`` without changing ``[B, C, H, W]``."""
        if x.ndim != 4 or x.size(1) != self.feature_channels:
            raise ValueError(
                f'Expected x with shape [B, {self.feature_channels}, H, W], '
                f'got {tuple(x.shape)}.')
        return self.net(x)


@MODELS.register_module()
class MetadataMoE(nn.Module):
    """Dense soft mixture-of-experts conditioned on metadata.

    All experts process the feature map and their outputs are combined using
    per-sample router weights.  The expert loop deliberately accumulates into
    a single tensor instead of stacking all high-resolution expert outputs.

    Args:
        feature_channels: Input/output feature channels. Defaults to 256.
        router_input_dim: Metadata feature dimension. Defaults to 128.
        num_experts: Number of dense experts. Defaults to 4.
        expert_hidden_ratio: Hidden channels as a fraction of feature channels.
        temperature: Router softmax temperature. Defaults to 1.0.
        residual_scale_init: Initial value of the learnable residual scale.
    """

    def __init__(self,
                 feature_channels: int = 256,
                 router_input_dim: int = 128,
                 num_experts: int = 4,
                 expert_hidden_ratio: float = 0.25,
                 temperature: float = 1.0,
                 residual_scale_init: float = 0.01) -> None:
        super().__init__()
        if feature_channels <= 0 or router_input_dim <= 0 or num_experts <= 0:
            raise ValueError(
                'feature_channels, router_input_dim, and num_experts must be '
                'positive.')
        if expert_hidden_ratio <= 0:
            raise ValueError('expert_hidden_ratio must be positive.')

        hidden_channels = max(1, int(feature_channels * expert_hidden_ratio))
        self.feature_channels = feature_channels
        self.router_input_dim = router_input_dim
        self.num_experts = num_experts
        self.expert_hidden_ratio = expert_hidden_ratio
        self.router = MetadataRouter(
            input_dim=router_input_dim,
            hidden_dim=64,
            num_experts=num_experts,
            temperature=temperature)
        self.experts = nn.ModuleList([
            MetadataExpert(
                feature_channels=feature_channels,
                hidden_channels=hidden_channels)
            for _ in range(num_experts)
        ])
        self.gamma = nn.Parameter(
            torch.tensor(float(residual_scale_init), dtype=torch.float32))

    def forward(self, x: Tensor, aux_feature: Tensor) -> Tensor:
        """Apply dense soft routing and a learnable residual connection."""
        if x.ndim != 4 or x.size(1) != self.feature_channels:
            raise ValueError(
                f'Expected x with shape [B, {self.feature_channels}, H, W], '
                f'got {tuple(x.shape)}.')
        if aux_feature.ndim != 2 or aux_feature.size(0) != x.size(0):
            raise ValueError(
                'aux_feature must have shape [B, router_input_dim] with the '
                'same batch size as x; got '
                f'{tuple(aux_feature.shape)} for x batch size {x.size(0)}.')
        if aux_feature.size(1) != self.router_input_dim:
            raise ValueError(
                f'Expected aux_feature dimension {self.router_input_dim}, '
                f'got {aux_feature.size(1)}.')
        if x.device != aux_feature.device:
            raise ValueError('x and aux_feature must be on the same device.')
        if x.dtype != aux_feature.dtype:
            raise ValueError('x and aux_feature must have the same dtype.')

        router_weights = self.router(aux_feature)
        mixed = torch.zeros_like(x)
        for expert_index, expert in enumerate(self.experts):
            expert_output = expert(x)
            sample_weights = router_weights[:, expert_index].reshape(
                x.size(0), 1, 1, 1)
            mixed = mixed + sample_weights * expert_output

        # Casting the scalar preserves the feature dtype under explicit
        # mixed-precision calls while retaining a gradient to self.gamma.
        gamma = self.gamma.to(device=x.device, dtype=x.dtype)
        return x + gamma * mixed


__all__ = ['MetadataRouter', 'MetadataExpert', 'MetadataMoE']
