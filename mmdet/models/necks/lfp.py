"""Low-frequency guided feature purification for AuxDet.

Algorithmic adaptation based on the LFP idea described in:
https://github.com/mengduann/NS-FPN

The NS-FPN repository does not provide an explicit LICENSE file in the
workspace, so this module is implemented independently rather than copied
verbatim. It is adapted for MMDetection/AuxDet feature tensors.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _WaveletTransform(nn.Module):
    """Thin wrappers around pytorch_wavelets with lazy dependency loading."""

    def __init__(self, wave: str, mode: str) -> None:
        super().__init__()
        try:
            from pytorch_wavelets import DWTForward, DWTInverse
        except ImportError as exc:
            raise ImportError(
                'LFP requires pytorch_wavelets. Install it in the runtime '
                'environment before enabling lfp_cfg.') from exc
        self.dwt = DWTForward(J=1, wave=wave, mode=mode)
        self.idwt = DWTInverse(wave=wave, mode=mode)

    def forward_dwt(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        with torch.cuda.amp.autocast(enabled=False):
            x_fp32 = x.float() if x.dtype != torch.float32 else x
            low, highs = self.dwt(x_fp32)
        # pytorch_wavelets returns [B, C, 3, H, W]. Move the
        # orientation axis before flattening so the inverse can restore it.
        high = highs[0].transpose(1, 2).reshape(
            highs[0].shape[0],
            3 * highs[0].shape[1],
            highs[0].shape[3],
            highs[0].shape[4],
        )
        return low, high

    def forward_idwt(self, low: Tensor, high: Tensor) -> Tensor:
        b, c, h, w = low.shape
        # Inverse of [B,C,3,H,W] -> [B,3C,H,W].
        high = high.reshape(b, 3, c, h, w).transpose(1, 2).contiguous()
        with torch.cuda.amp.autocast(enabled=False):
            rec = self.idwt((low, [high.float()]))
        return rec


class _SpatialAttention(nn.Module):
    """LL-guided spatial attention used to gate high-frequency components."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError('kernel_size must be 3 or 7')
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        max_value, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, max_value], dim=1)))


class _LearnableGaussianFilterBank(nn.Module):
    """Depthwise Gaussian filtering for the 3C high-frequency channels."""

    def __init__(self, kernel_size: int, num_channels: int) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError('kernel_size must be odd')
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.padding = kernel_size // 2
        self.sigma = nn.Parameter(torch.ones(1))

    def _kernel(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        coords = torch.arange(
            self.kernel_size, device=device, dtype=dtype
        ) - self.padding
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        sigma_sq = self.sigma.to(device=device, dtype=dtype).square().clamp_min(1e-6)
        kernel = torch.exp(-(xx.square() + yy.square()) / (2 * sigma_sq))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, self.kernel_size, self.kernel_size).repeat(
            self.num_channels, 1, 1, 1
        )

    def forward(self, x: Tensor) -> Tensor:
        weight = self._kernel(x.device, x.dtype)
        padded = F.pad(
            x,
            (self.padding, self.padding, self.padding, self.padding),
            mode='replicate',
        )
        return F.conv2d(padded, weight=weight, groups=self.num_channels)


class LFP(nn.Module):
    """Low-frequency guided feature purification.

    Input and output tensors both have shape ``[B, C, H, W]``. The module
    uses one-level Haar DWT/IDWT, LL-guided spatial attention, and a gated
    learnable depthwise Gaussian filter on the 3C high-frequency channels.
    """

    def __init__(
        self,
        wave: str = 'haar',
        mode: str = 'zero',
        with_gauss: bool = True,
        gauss_gate: float = 0.5,
        in_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.wave = wave
        self.mode = mode
        self.with_gauss = with_gauss
        self.gauss_gate = gauss_gate
        self.in_channels = in_channels
        self._wavelet = _WaveletTransform(wave=wave, mode=mode)
        self.attention = _SpatialAttention(kernel_size=7)
        self.gaussian_filter = None
        if with_gauss:
            if in_channels is None:
                raise ValueError('in_channels is required when with_gauss=True')
            self.gaussian_filter = _LearnableGaussianFilterBank(
                kernel_size=3, num_channels=3 * in_channels
            )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f'LFP expects [B,C,H,W], got {tuple(x.shape)}')
        input_size = x.shape[-2:]
        channels = x.shape[1]
        if self.in_channels is not None and channels != self.in_channels:
            raise ValueError(
                f'LFP expected {self.in_channels} channels, got {channels}'
            )

        low, high = self._wavelet.forward_dwt(x)
        high = high * self.attention(low)

        if self.with_gauss:
            assert self.gaussian_filter is not None
            blurred = self.gaussian_filter(high)
            mask = (high.abs() < self.gauss_gate).to(dtype=high.dtype)
            high = high * (1 - mask) + blurred * mask

        output = self._wavelet.forward_idwt(low, high)
        if output.shape[-2] < input_size[0] or output.shape[-1] < input_size[1]:
            raise RuntimeError(
                f'IDWT output {tuple(output.shape[-2:])} is smaller than '
                f'input {tuple(input_size)}')
        # DWT zero padding can make odd-size reconstruction one pixel larger.
        # Crop the inverse transform back to the original spatial size.
        output = output[..., :input_size[0], :input_size[1]]
        return output.to(dtype=x.dtype)


@torch.no_grad()
def check_dwt_idwt_reconstruction(
    wave: str = 'haar',
    mode: str = 'zero',
    shape: Tuple[int, int, int, int] = (2, 5, 31, 29),
    tolerance: float = 1e-5,
) -> float:
    """Check pure multi-channel DWT to IDWT reconstruction.

    This deliberately bypasses attention and Gaussian filtering. The returned
    value is the maximum absolute reconstruction error after cropping any
    padding introduced for odd spatial dimensions.
    """
    transform = _WaveletTransform(wave=wave, mode=mode)
    x = torch.randn(shape, dtype=torch.float32)
    low, high = transform.forward_dwt(x)
    reconstruction = transform.forward_idwt(low, high)
    if reconstruction.shape[-2] < shape[-2] or reconstruction.shape[-1] < shape[-1]:
        raise AssertionError(
            f'IDWT output {tuple(reconstruction.shape[-2:])} is smaller than '
            f'input {shape[-2:]}')
    reconstruction = reconstruction[..., :shape[-2], :shape[-1]]
    error = (reconstruction - x).abs().max().item()
    if error > tolerance:
        raise AssertionError(
            f'DWT→IDWT reconstruction error {error:.6g} exceeds '
            f'tolerance {tolerance:.6g}')
    return error
