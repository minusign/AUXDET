"""AuxDet necks that replace only the post-AuxDet fusion stage.

The first part of every fusion neck is inherited from :class:`AuxFPN`, so
the metadata, semantic compensation, dynamic modulation and edge path keep
their original module names and state-dict keys.  Fusion bodies consume the
four already projected 256-channel levels and return exactly the two levels
required by the existing Cascade RPN/ROI configuration.
"""

import importlib
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from .aux_fpn import AuxFPN


class _DctSpatialInteraction(BaseModule):
    def __init__(self, in_channels, ratio, isdct=True):
        super().__init__()
        self.ratio = ratio
        self.isdct = isdct
        if not isdct:
            self.spatial1x1 = ConvModule(in_channels, 1, 1, bias=False)
        try:
            self.dct = importlib.import_module('torch_dct')
        except ImportError as exc:
            raise RuntimeError(
                'HS-FPN requires torch-dct; install the compatible package '
                'on the server before selecting an HS configuration.') from exc

    def forward(self, x):
        if not self.isdct:
            return x * torch.sigmoid(self.spatial1x1(x))
        _, _, h, w = x.shape
        h0, w0 = int(h * self.ratio[0]), int(w * self.ratio[1])
        weight = x.new_ones((h, w))
        weight[:h0, :w0] = 0
        dct = self.dct.dct_2d(x, norm='ortho')
        dct = dct * weight.view(1, h, w).expand_as(dct)
        return x * self.dct.idct_2d(dct, norm='ortho')


class _DctChannelInteraction(BaseModule):
    def __init__(self, in_channels, patch, ratio, isdct=True):
        super().__init__()
        self.h, self.w = patch
        self.ratio, self.isdct = ratio, isdct
        self.channel1x1 = ConvModule(in_channels, in_channels, 1, groups=32, bias=False)
        self.channel2x1 = ConvModule(in_channels, in_channels, 1, groups=32, bias=False)
        self.relu = nn.ReLU()
        try:
            self.dct = importlib.import_module('torch_dct')
        except ImportError as exc:
            raise RuntimeError(
                'HS-FPN requires torch-dct; install the compatible package '
                'on the server before selecting an HS configuration.') from exc

    def forward(self, x):
        n, c, h, w = x.shape
        if not self.isdct:
            amax = F.adaptive_max_pool2d(x, 1)
            aavg = F.adaptive_avg_pool2d(x, 1)
            channel = self.channel1x1(self.relu(amax)) + self.channel1x1(self.relu(aavg))
            return x * torch.sigmoid(self.channel2x1(channel))
        h0, w0 = int(h * self.ratio[0]), int(w * self.ratio[1])
        weight = x.new_ones((h, w)); weight[:h0, :w0] = 0
        dct = self.dct.dct_2d(x, norm='ortho')
        dct = self.dct.idct_2d(dct * weight.view(1, h, w).expand_as(dct), norm='ortho')
        amax = torch.sum(self.relu(F.adaptive_max_pool2d(dct, (self.h, self.w))), dim=(2, 3)).view(n, c, 1, 1)
        aavg = torch.sum(self.relu(F.adaptive_avg_pool2d(dct, (self.h, self.w))), dim=(2, 3)).view(n, c, 1, 1)
        channel = self.channel1x1(amax) + self.channel1x1(aavg)
        return x * torch.sigmoid(self.channel2x1(channel))


class _HFP(BaseModule):
    def __init__(self, channels, ratio, patch, isdct):
        super().__init__()
        self.spatial = _DctSpatialInteraction(channels, ratio, isdct)
        self.channel = _DctChannelInteraction(channels, patch, ratio, isdct)
        self.out = nn.Sequential(
            ConvModule(channels, channels, 3, padding=1),
            nn.GroupNorm(32, channels))

    def forward(self, x):
        return self.out(self.spatial(x) + self.channel(x))


class _SDP(BaseModule):
    def __init__(self, dim=256, inter_dim=None, improved=False):
        super().__init__()
        try:
            from einops import rearrange
        except ImportError as exc:
            raise RuntimeError('HS-FPN requires einops; install a compatible version on the server.') from exc
        self._rearrange = rearrange
        inter_dim = inter_dim or dim
        kernel = 3 if improved else 1
        self.conv_q = nn.Sequential(ConvModule(dim, inter_dim, kernel, padding=kernel // 2, bias=False), nn.GroupNorm(32, inter_dim))
        self.conv_k = nn.Sequential(ConvModule(dim, inter_dim, kernel, padding=kernel // 2, bias=False), nn.GroupNorm(32, inter_dim))
        self.conv = (nn.Sequential(ConvModule(inter_dim, dim, 3, padding=1, bias=False), nn.GroupNorm(32, dim)) if improved else nn.Identity())
        self.inter_dim = inter_dim
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_low, x_high, patch_size):
        b, _, h, w = x_low.shape
        ph, pw = patch_size
        if h % ph or w % pw:
            raise ValueError(f'HS-FPN SDP requires {h}x{w} divisible by patch {patch_size}')
        rearrange = self._rearrange
        q = rearrange(self.conv_q(x_low), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=ph, p2=pw).transpose(1, 2)
        k = rearrange(self.conv_k(x_high), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=ph, p2=pw)
        attn = self.softmax(torch.matmul(q, k) / (self.inter_dim ** 0.5))
        out = torch.matmul(attn, k.transpose(1, 2))
        out = rearrange(out.transpose(1, 2).contiguous(), '(b h w) c (p1 p2) -> b c (h p1) (w p2)', p1=ph, p2=pw, h=h // ph, w=w // pw)
        return self.conv(out + x_low)


class _HSFusionBody(BaseModule):
    def __init__(self, channels=256, ratio=(0.25, 0.25)):
        super().__init__()
        self.SelfAttn_p4 = _HFP(channels, None, (1, 1), False)
        self.SelfAttn_p3 = _HFP(channels, None, (1, 1), False)
        self.SelfAttn_p2 = _HFP(channels, ratio, (8, 8), True)
        self.SelfAttn_p1 = _HFP(channels, ratio, (16, 16), True)
        self.CrossAtten_p4_p3 = _SDP(channels)
        self.CrossAtten_p3_p2 = _SDP(channels)
        self.CrossAtten_p2_p1 = _SDP(channels)

    @staticmethod
    def _up(x):
        return F.interpolate(x, scale_factor=2, mode='nearest')

    def forward(self, laterals):
        p1, p2, p3, p4 = laterals
        h, w = p4.shape[-2:]
        p4 = self.SelfAttn_p4(p4)
        p3 = self.CrossAtten_p4_p3(self.SelfAttn_p3(p3), self._up(p4), (h, w))
        p2 = self.CrossAtten_p3_p2(self.SelfAttn_p2(p2), self._up(p3), (h, w))
        p1 = self.CrossAtten_p2_p1(self.SelfAttn_p1(p1), self._up(p2), (h, w))
        laterals = [p1, p2, p3, p4]
        for i in range(3, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest')
        return laterals


class _NSConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.module = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=False))

    def forward(self, x):
        return self.module(x)


class _NSFusionBody(nn.Module):
    def __init__(self, channels=256, use_wav_enhance=True, use_crossattn_topdown=True):
        super().__init__()
        try:
            from pytorch_wavelets import DWTForward, DWTInverse
        except ImportError as exc:
            raise RuntimeError('NS-FPN requires pytorch-wavelets==1.3.0') from exc
        try:
            from .sfs_ms_deform_attn.modules import MSDeformAttn_for_sfs
        except ImportError as exc:
            raise RuntimeError('NS-FPN requires the compiled SFS extension. Build mmdet/models/necks/sfs_ms_deform_attn with setup.py before using NS configs.') from exc
        self.DWTForward, self.DWTInverse = DWTForward, DWTInverse
        self.channels = channels
        self.use_wav_enhance = use_wav_enhance
        self.use_crossattn_topdown = use_crossattn_topdown
        # AuxFPN has already performed the outer lateral 1x1 projection and
        # its original output 3x3 convolution is reused by the parent neck.
        # Keep no duplicate copies of either layer in the NS body.
        self.lateral = nn.ModuleList([nn.Identity() for _ in range(4)])
        self.dwt = nn.ModuleList([self._make_wav() if use_wav_enhance else nn.Identity() for _ in range(4)])
        self.cross = nn.ModuleList([_NSSpiral(channels, MSDeformAttn_for_sfs) for _ in range(3)]) if use_crossattn_topdown else nn.ModuleList()

    def _make_wav(self):
        return _NSWaveEnhance(self.channels, self.DWTForward, self.DWTInverse)

    def forward(self, inputs):
        laterals = [self.dwt[i](self.lateral[i](inputs[i])) for i in range(4)]
        if self.use_crossattn_topdown:
            for i in range(3, 0, -1):
                laterals[i - 1] = self.cross[i - 1](laterals[i - 1], laterals[i])
        else:
            for i in range(3, 0, -1):
                laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest')
        return laterals


class _NSWaveEnhance(nn.Module):
    def __init__(self, channels, dwt_cls, idwt_cls):
        super().__init__()
        self.dwt, self.idwt = dwt_cls(J=1, wave='haar', mode='zero'), idwt_cls(wave='haar', mode='zero')
        self.attention = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())
        self.channels = channels
        self.gaussian = _NSGaussianFilterBank(3, channels * 3)

    def forward(self, x):
        h, w = x.shape[-2:]
        with torch.cuda.amp.autocast(enabled=False):
            ll, yh = self.dwt(x.float())
            high = yh[0].transpose(1, 2).reshape(x.shape[0], -1, yh[0].shape[-2], yh[0].shape[-1])
            att = self.attention(torch.cat([ll.mean(1, keepdim=True), ll.max(1, keepdim=True).values], 1))
            high = high * att
            blurred = self.gaussian(high)
            gate = (high.abs() < 0.5).to(high.dtype)
            high = high * (1 - gate) + blurred * gate
            rec = self.idwt((ll, [high.reshape(x.shape[0], self.channels, 3, high.shape[-2], high.shape[-1])]))
            rec = F.interpolate(rec, size=(h, w), mode='bilinear', align_corners=False)
        return rec.to(dtype=x.dtype)


class _NSGaussianFilterBank(nn.Module):
    """Learnable depthwise Gaussian bank used by the reference LFP."""

    def __init__(self, kernel_size, channels):
        super().__init__()
        self.kernel_size, self.channels = kernel_size, channels
        self.sigmas = nn.ParameterList([nn.Parameter(torch.ones(1))])

    def forward(self, x):
        pad = self.kernel_size // 2
        sigma = self.sigmas[0].abs() + 1e-6
        coords = torch.arange(self.kernel_size, device=x.device, dtype=x.dtype) - pad
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        weight = kernel.view(1, 1, self.kernel_size, self.kernel_size).repeat(self.channels, 1, 1, 1)
        return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode='replicate'), weight, groups=self.channels)


class _NSSpiral(nn.Module):
    def __init__(self, dim, attn_cls):
        super().__init__()
        self.dim, self.n_heads, self.n_points = dim, 8, 4
        self.query_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
        self.key_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
        self.offset_residual = nn.Parameter(torch.zeros(8, 4, 2))
        grid = []
        import math
        for head in range(8):
            row = []
            for point in range(4):
                theta = 2 * math.pi * point / 4 + 2 * math.pi * head / 8
                radius = 1.0 + point
                row.append([radius * math.cos(theta), radius * math.sin(theta)])
            grid.append(row)
        self.register_buffer('offset_base', torch.tensor(grid, dtype=torch.float32).view(1, 1, 8, 1, 4, 2))
        self.query_norm, self.key_norm, self.out_norm = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = attn_cls(d_model=dim, n_levels=1, n_heads=8, n_points=4)

    def forward(self, query_feat, key_feat):
        b, c, h1, w1 = query_feat.shape
        _, _, h2, w2 = key_feat.shape
        qf, kf = self.query_conv(query_feat), self.key_conv(key_feat)
        offsets = (self.offset_base.view(8, 1, 4, 2) + self.offset_residual.view(8, 1, 4, 2)).view(1, 1, 8, 1, 4, 2).expand(b, h1 * w1, -1, -1, -1, -1)
        query, kv = self.query_norm(qf.flatten(2).transpose(1, 2)), self.key_norm(kf.flatten(2).transpose(1, 2))
        shapes = torch.tensor([[h2, w2]], device=key_feat.device, dtype=torch.long)
        starts = torch.tensor([0], device=key_feat.device, dtype=torch.long)
        gy, gx = torch.meshgrid(torch.linspace(.5 / h1, 1 - .5 / h1, h1, device=qf.device), torch.linspace(.5 / w1, 1 - .5 / w1, w1, device=qf.device), indexing='ij')
        refs = torch.stack((gx, gy), -1).view(1, h1 * w1, 1, 2).repeat(b, 1, 1, 1)
        out = self.attn(query, refs, kv, shapes, starts, offsets)
        return self.out_norm(query + query * out).transpose(1, 2).reshape(b, c, h1, w1)


@MODELS.register_module()
class AuxFPNFusion(AuxFPN):
    """AuxFPN with an HS-FPN or NS-FPN post-processing body."""

    def __init__(self, fusion_type='hs', hs_ratio=(0.25, 0.25), **kwargs):
        super().__init__(**kwargs)
        fusion_type = fusion_type.lower()
        if fusion_type not in ('hs', 'ns'):
            raise ValueError("fusion_type must be 'hs' or 'ns'")
        self.fusion_type = fusion_type
        if fusion_type == 'hs':
            self.fusion_body = _HSFusionBody(self.out_channels, hs_ratio)
        else:
            self.fusion_body = _NSFusionBody(self.out_channels)

    def forward(self, inputs, meta_inf):
        laterals = self._prepare_aux_laterals(inputs, meta_inf)
        laterals = self.fusion_body(laterals)
        # HS/NS bodies include their own fusion top-down path. Do not repeat
        # the original AuxFPN top-down pass after the selected body.
        return self._build_aux_outputs(inputs, laterals, use_top_down=False)
