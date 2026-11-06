"""Training targets and auxiliary supervision for the internal P2 feature."""
import math

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmdet.models.losses.gaussian_focal_loss import GaussianFocalLoss
from mmdet.models.utils.gaussian_target import (gaussian_radius,
                                               gen_gaussian_target)


class TargetHeatmapHead(BaseModule):
    """Preserve resolution and start with a spatially neutral sigmoid of .5.

    The child init_cfg re-applies the zero prediction initialization after
    AuxFPN's recursive Xavier initialization. No BatchNorm is used.
    """

    def __init__(self, in_channels=64, hidden_channels=64):
        super().__init__(init_cfg=dict(
            type='Constant', val=0, bias=0,
            override=dict(name='prediction')))
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GELU())
        self.prediction = nn.Conv2d(hidden_channels, 1, 1)
        nn.init.zeros_(self.prediction.weight)
        nn.init.zeros_(self.prediction.bias)

    def forward(self, x):
        return self.prediction(self.features(x))


@torch.no_grad()
def build_target_heatmaps(logits, batch_data_samples, batch_input_shape,
                          min_radius=1, max_radius=16, min_overlap=0.7):
    """Map augmented XYXY GT into the feature lattice, without re-scaling GT.

    ``img_shape`` describes the real resized image. ``batch_input_shape``
    must come from the actual preprocessed batch Tensor, including padding;
    per-image ``pad_shape`` cannot substitute for it. Padding is bottom/right
    in the current pipeline. A lattice point is valid iff its image-space
    coordinate is within img_shape (ceil division includes the border cell).
    Multiple objects use the repository's max-composed Gaussian drawing.
    """
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError('Expected heatmap logits with shape [B, 1, H, W].')
    if len(batch_data_samples) != logits.shape[0]:
        raise ValueError('Heatmap batch and data sample counts must match.')
    if not (isinstance(min_radius, int) and isinstance(max_radius, int)
            and 0 <= min_radius <= max_radius and 0 < min_overlap < 1):
        raise ValueError('Invalid Gaussian radius bounds or min_overlap.')
    input_h, input_w = batch_input_shape[-2:]
    height, width = logits.shape[-2:]
    if min(input_h, input_w, height, width) <= 0:
        raise ValueError('Image and feature sizes must be positive.')
    scale_x, scale_y = width / input_w, height / input_h
    targets = torch.zeros(logits.shape, device=logits.device,
                          dtype=torch.float32)
    valid = torch.zeros_like(targets, dtype=torch.bool)
    for index, sample in enumerate(batch_data_samples):
        img_h, img_w = sample.img_shape[:2]
        if not (0 < img_h <= input_h and 0 < img_w <= input_w):
            raise ValueError('img_shape must fit in the actual batch Tensor.')
        valid_h = min(height, math.ceil(img_h * scale_y))
        valid_w = min(width, math.ceil(img_w * scale_x))
        valid[index, 0, :valid_h, :valid_w] = True

        # The preprocessor normally converts BaseBoxes to Tensor. Support
        # both forms, and always clone so the detector GT cannot be changed.
        boxes = sample.gt_instances.bboxes
        boxes = getattr(boxes, 'tensor', boxes)
        boxes = boxes.to(device=logits.device, dtype=torch.float32).clone()
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError('GT boxes must have shape [N, 4] in XYXY format.')
        boxes = boxes[torch.isfinite(boxes).all(dim=1)]
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, img_w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, img_h)
        boxes = boxes[(boxes[:, 2] > boxes[:, 0])
                      & (boxes[:, 3] > boxes[:, 1])]
        # One small CPU transfer per image avoids repeated GPU scalar reads.
        for x1, y1, x2, y2 in boxes.cpu().tolist():
            center_x = min(valid_w - 1, int((x1 + x2) * 0.5 * scale_x))
            center_y = min(valid_h - 1, int((y1 + y2) * 0.5 * scale_y))
            radius = gaussian_radius(
                ((y2 - y1) * scale_y, (x2 - x1) * scale_x), min_overlap)
            radius = max(min_radius, min(max_radius, int(radius)))
            gen_gaussian_target(targets[index, 0],
                                [center_x, center_y], radius)
    # Gaussian tails in padding do not participate in supervision.
    targets.masked_fill_(~valid, 0)
    return targets, valid


class TargetHeatmapLoss(nn.Module):
    """Reuse GaussianFocalLoss, with separate per-image pos/neg averages.

    Repository GaussianFocalLoss expects probabilities and supports Gaussian
    targets, element weights and unreduced output. FP32 sigmoid/clamping keeps
    its logarithms finite under AMP. The only reduction change is to divide
    the positive term by the number of distinct center pixels and the negative
    term by the number of valid non-center pixels, then average the batch.
    Empty images have zero positive term and a mean negative term.
    """

    def __init__(self, loss_weight=0.05, alpha=2.0, gamma=4.0):
        super().__init__()
        if loss_weight < 0 or alpha < 0 or gamma < 0:
            raise ValueError('Heatmap loss weight and exponents must be >= 0.')
        self.loss_weight = loss_weight
        self.focal = GaussianFocalLoss(
            alpha=alpha, gamma=gamma, reduction='none')

    def forward(self, logits, targets, valid):
        # Casting preserves autograd; no detach on predictions or the loss.
        pred = logits.float().sigmoid()
        eps = torch.finfo(pred.dtype).eps
        pred = pred.clamp(eps, 1 - eps)
        targets = targets.float()
        pixel_loss = self.focal(pred, targets)
        positive = valid & targets.eq(1)
        negative = valid & ~positive
        dims = (1, 2, 3)
        pos_loss = (pixel_loss * positive).sum(dims)
        pos_loss = pos_loss / positive.sum(dims).clamp_min(1)
        neg_loss = (pixel_loss * negative).sum(dims)
        neg_loss = neg_loss / negative.sum(dims).clamp_min(1)
        return self.loss_weight * (pos_loss + neg_loss).mean()
