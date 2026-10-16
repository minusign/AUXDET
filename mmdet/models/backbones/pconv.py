"""Pinwheel convolution and the AuxDet ResNet integration.

The PConv implementation is adapted from the MIT licensed
PConv-SDloss-Data repository (``model/APConv.py``).  It intentionally keeps
the reference module's internal BatchNorm and SiLU, while the surrounding
ResNet bottleneck keeps its original norm, residual and activation path.
"""

import torch
import torch.nn as nn

from mmdet.registry import MODELS

from .resnet import ResNet


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    return k // 2 if p is None and isinstance(k, int) else ([x // 2 for x in k] if p is None else p)


class PConvConv(nn.Module):
    """Reference Conv: convolution, BatchNorm2d and SiLU."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g,
                              dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = (self.default_act if act is True else act
                    if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


# Keep the reference source's public helper name without colliding with
# MMCV's builder imports in the surrounding ResNet code.
Conv = PConvConv


class PConv(nn.Module):
    """Pinwheel-shaped convolution using four asymmetric padding branches."""

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        if c2 % 4 != 0:
            raise ValueError(f'PConv requires c2 divisible by 4, got {c2}')
        pads = [(k, 0, 1, 0), (0, k, 0, 1),
                (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = nn.ModuleList([nn.ZeroPad2d(padding=p) for p in pads])
        self.cw = PConvConv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = PConvConv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = PConvConv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))


@MODELS.register_module()
class ResNetPConv(ResNet):
    """ResNet with the explicitly selected first blocks using PConv.

    Only ``layer1.0.conv2`` and ``layer2.0.conv2`` are replaced.  The
    original Bottleneck norm2, residual path and final activation remain in
    place, as required for a controlled experiment.
    """

    def __init__(self, pconv_kernel=3, pconv_stages=(1, 2), **kwargs):
        super().__init__(**kwargs)
        self.pconv_stages = tuple(pconv_stages)
        # Keep a plain list: registering the same module a second time would
        # add duplicate ``_pconv_modules.*`` state-dict prefixes.
        self._pconv_modules = []
        for stage_idx in self.pconv_stages:
            if stage_idx not in (1, 2):
                raise ValueError('ResNetPConv only supports stages 1 and 2')
            layer = getattr(self, f'layer{stage_idx}')
            block = layer[0]
            old_conv = block.conv2
            if getattr(old_conv, 'dilation', (1, 1)) not in ((1, 1), 1):
                raise ValueError('PConv integration currently requires dilation=1')
            replacement = PConv(block.planes, block.planes, k=pconv_kernel,
                                 s=old_conv.stride[0])
            block.conv2 = replacement
            self._pconv_modules.append(replacement)
        self._enable_pconv_parameters()

    def _enable_pconv_parameters(self):
        for module in self._pconv_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        # ResNet._freeze_stages is called by train(); explicitly re-enable the
        # new modules so the experimental layers are trainable even when the
        # baseline freezes layer1. Their BatchNorm follows the requested mode.
        self._enable_pconv_parameters()
        for module in self._pconv_modules:
            module.train(mode)
        return self
