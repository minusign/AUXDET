"""Quality-aware wrapper for AuxDet's single-stage StandardRoIHead.

The original StandardRoIHead and BBoxHead are intentionally left untouched.
This subclass adds a small quality branch on the RoIAlign feature, keeps the
classification/regression losses unchanged, and feeds a fused foreground
score to the original BBoxHead.predict_by_feat (and therefore its original
NMS implementation).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList
from mmdet.models.task_modules.samplers import SamplingResult
from .standard_roi_head import StandardRoIHead


@MODELS.register_module()
class QualityAwareStandardRoIHead(StandardRoIHead):
    """Standard RoI head with an optional class-agnostic quality score.

    The current AuxDet task has one foreground class.  The quality head is
    consequently scalar per RoI and is multiplied only into the foreground
    probability before the inherited BBoxHead performs multiclass NMS.
    """

    def __init__(self,
                 quality_enabled: bool = True,
                 quality_score_fusion_enabled: bool = True,
                 quality_loss_weight: float = 0.05,
                 quality_alpha: float = 0.5,
                 quality_hidden_dim: int = 64,
                 quality_in_channels: int = 256,
                 quality_iou_legacy_coordinate: bool = True,
                 **kwargs) -> None:
        self.quality_enabled = bool(quality_enabled)
        self.quality_score_fusion_enabled = bool(
            quality_score_fusion_enabled)
        self.quality_loss_weight = float(quality_loss_weight)
        self.quality_alpha = float(quality_alpha)
        self.quality_hidden_dim = int(quality_hidden_dim)
        self.quality_in_channels = int(quality_in_channels)
        self.quality_iou_legacy_coordinate = bool(
            quality_iou_legacy_coordinate)
        super().__init__(**kwargs)

        # Do not add state_dict keys when the feature is explicitly disabled;
        # this keeps the disabled path structurally identical to the original
        # StandardRoIHead and allows old checkpoints to load strictly.
        self.quality_head = None
        if self.quality_enabled:
            in_channels = int(getattr(self.bbox_roi_extractor,
                                     'out_channels',
                                     self.quality_in_channels))
            self.quality_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(in_channels, self.quality_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.quality_hidden_dim, 1))

    def init_weights(self) -> None:
        """Initialize parent modules and the quality prior.

        A zero final weight makes the initial quality prediction spatially
        independent.  The bias is logit(0.95), so the first forward pass does
        not attenuate normal classification scores by q=0.5.
        """
        super().init_weights()
        if self.quality_head is not None:
            final = self.quality_head[-1]
            nn.init.zeros_(final.weight)
            nn.init.constant_(final.bias, math.log(0.95 / 0.05))

    def _bbox_forward(self, x: Tuple[Tensor], rois: Tensor) -> dict:
        bbox_results = super()._bbox_forward(x, rois)
        if self.quality_head is not None:
            bbox_results['quality_logits'] = self.quality_head(
                bbox_results['bbox_feats'])
        return bbox_results

    @staticmethod
    def _aligned_iou(boxes1: Tensor, boxes2: Tensor,
                     legacy_coordinate: bool = True) -> Tensor:
        """IoU for aligned boxes using the explicitly selected convention.

        AuxDet's VOCMetric uses ``use_legacy_coordinate=True`` (the ``+1``
        width/height convention), so that is the default for quality targets.
        The flag remains explicit because BBoxCoder itself stores continuous
        xyxy coordinates.  Degenerate/non-finite boxes produce a finite zero
        target.
        """
        boxes1 = torch.nan_to_num(boxes1, nan=0.0, posinf=0.0, neginf=0.0)
        boxes2 = torch.nan_to_num(boxes2, nan=0.0, posinf=0.0, neginf=0.0)
        lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
        rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
        extra = 1.0 if legacy_coordinate else 0.0
        wh = (rb - lt + extra).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        area1 = ((boxes1[:, 2] - boxes1[:, 0] + extra).clamp(min=0) *
                 (boxes1[:, 3] - boxes1[:, 1] + extra).clamp(min=0))
        area2 = ((boxes2[:, 2] - boxes2[:, 0] + extra).clamp(min=0) *
                 (boxes2[:, 3] - boxes2[:, 1] + extra).clamp(min=0))
        union = (area1 + area2 - inter).clamp(min=torch.finfo(
            boxes1.dtype).eps)
        return torch.nan_to_num(inter / union, nan=0.0, posinf=0.0,
                                neginf=0.0).clamp(0.0, 1.0)

    def _quality_targets(self, bbox_pred: Optional[Tensor],
                         sampling_results: List[SamplingResult]) -> Tuple[
                             Tensor, Tensor, Tensor]:
        """Build quality targets in the exact positive-then-negative order."""
        if not sampling_results:
            device = (bbox_pred.device if bbox_pred is not None else
                      torch.device('cpu'))
            empty = torch.empty(0, device=device, dtype=torch.float32)
            return empty, empty.bool(), empty.bool()
        device = (bbox_pred.device if bbox_pred is not None else
                  sampling_results[0].priors.device)
        total = sum(len(result.priors) for result in sampling_results)
        targets = torch.zeros(total, device=device, dtype=torch.float32)
        positive = torch.zeros(total, device=device, dtype=torch.bool)
        negative = torch.zeros(total, device=device, dtype=torch.bool)
        if bbox_pred is None:
            offset = 0
            for result in sampling_results:
                num_pos = len(result.pos_priors)
                num_neg = len(result.neg_priors)
                positive[offset:offset + num_pos] = True
                negative[offset + num_pos:offset + num_pos + num_neg] = True
                offset += num_pos + num_neg
            return targets, positive, negative

        offset = 0
        reg_dim = self.bbox_head.bbox_coder.encode_size
        for result in sampling_results:
            num_pos = len(result.pos_priors)
            num_neg = len(result.neg_priors)
            if num_pos:
                positive[offset:offset + num_pos] = True
                pos_pred = bbox_pred[offset:offset + num_pos]
                labels = result.pos_gt_labels.long()
                if not self.bbox_head.reg_class_agnostic:
                    pos_pred = pos_pred.view(
                        num_pos, self.bbox_head.num_classes, reg_dim)
                    pos_pred = pos_pred[
                        torch.arange(num_pos, device=pos_pred.device), labels]
                pred_boxes = self.bbox_head.bbox_coder.decode(
                    result.pos_priors, pos_pred).detach()
                gt_boxes = result.pos_gt_bboxes.detach().to(pred_boxes)
                targets[offset:offset + num_pos] = self._aligned_iou(
                    pred_boxes, gt_boxes,
                    self.quality_iou_legacy_coordinate).to(targets)
            if num_neg:
                negative[offset + num_pos:offset + num_pos + num_neg] = True
            offset += num_pos + num_neg
        return targets.clamp(0.0, 1.0), positive, negative

    def _quality_loss(self, quality_logits: Tensor,
                      bbox_pred: Optional[Tensor], rois: Tensor,
                      sampling_results: List[SamplingResult]) -> Tensor:
        del rois  # The SamplingResult priors define the same order directly.
        targets, positive, negative = self._quality_targets(
            bbox_pred, sampling_results)
        logits = torch.nan_to_num(quality_logits.reshape(-1).float(),
                                  nan=0.0, posinf=20.0, neginf=-20.0)
        if logits.numel() == 0:
            return quality_logits.sum() * 0.0
        losses = []
        if positive.any():
            losses.append(F.binary_cross_entropy_with_logits(
                logits[positive], targets[positive].float(),
                reduction='mean'))
        if negative.any():
            losses.append(F.binary_cross_entropy_with_logits(
                logits[negative], targets[negative].float(),
                reduction='mean'))
        if not losses:
            return logits.sum() * 0.0
        # Equal group weighting prevents the usually larger negative group
        # from dominating the quality objective.
        return self.quality_loss_weight * torch.stack(losses).mean()

    def bbox_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult]) -> dict:
        bbox_results = super().bbox_loss(x, sampling_results)
        if self.quality_head is not None:
            rois = bbox2roi([result.priors for result in sampling_results])
            bbox_results['loss_bbox']['loss_roi_quality'] = self._quality_loss(
                bbox_results['quality_logits'], bbox_results['bbox_pred'],
                rois, sampling_results)
        return bbox_results

    @staticmethod
    def _fuse_cls_scores(cls_score: Tensor, quality_logits: Tensor,
                         alpha: float) -> Tensor:
        """Return logits whose softmax foreground probability is p*q^alpha."""
        if cls_score.size(-1) != 2:
            raise ValueError(
                'QualityAwareStandardRoIHead currently supports one '
                'foreground class (two softmax channels).')
        safe_cls = torch.nan_to_num(cls_score.float(), nan=0.0,
                                    posinf=20.0, neginf=-20.0)
        probs = F.softmax(safe_cls, dim=-1)
        p = probs[:, :1]
        safe_quality = torch.nan_to_num(quality_logits.float(), nan=0.0,
                                        posinf=20.0, neginf=-20.0)
        q = safe_quality.sigmoid().clamp(1e-6, 1.0)
        p_new = (p * q.pow(alpha)).clamp(1e-6, 1.0 - 1e-6)
        fused = torch.cat([p_new, 1.0 - p_new], dim=-1)
        return fused.log()

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False) -> InstanceList:
        # This exact parent path preserves the old behavior when either
        # quality switch is disabled.
        if self.quality_head is None or not self.quality_score_fusion_enabled:
            return super().predict_bbox(
                x, batch_img_metas, rpn_results_list, rcnn_test_cfg,
                rescale=rescale)

        proposals = [res.bboxes for res in rpn_results_list]
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = bbox2roi(proposals)
        if rois.shape[0] == 0:
            # Delegate empty handling to the original implementation.
            return super().predict_bbox(
                x, batch_img_metas, rpn_results_list, rcnn_test_cfg,
                rescale=rescale)

        bbox_results = self._bbox_forward(x, rois)
        cls_scores = bbox_results['cls_score'].split(num_proposals_per_img, 0)
        quality_logits = bbox_results['quality_logits'].split(
            num_proposals_per_img, 0)
        bbox_preds = bbox_results['bbox_pred']
        if bbox_preds is not None:
            if isinstance(bbox_preds, torch.Tensor):
                bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
            else:
                bbox_preds = self.bbox_head.bbox_pred_split(
                    bbox_preds, num_proposals_per_img)
        else:
            bbox_preds = (None, ) * len(proposals)
        fused_cls_scores = [
            self._fuse_cls_scores(cls, quality, self.quality_alpha)
            for cls, quality in zip(cls_scores, quality_logits)
        ]
        return self.bbox_head.predict_by_feat(
            rois=rois.split(num_proposals_per_img, 0),
            cls_scores=fused_cls_scores,
            bbox_preds=bbox_preds,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=rcnn_test_cfg,
            rescale=rescale)
