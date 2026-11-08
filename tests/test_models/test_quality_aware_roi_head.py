"""Server-side tests for the opt-in QualityAwareStandardRoIHead."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
from mmengine.config import Config  # noqa: E402
from mmdet.registry import MODELS  # noqa: E402
from mmdet.utils import register_all_modules  # noqa: E402

import mmdet.models.roi_heads.quality_aware_standard_roi_head  # noqa: E402,F401
from mmdet.models.roi_heads.quality_aware_standard_roi_head import (  # noqa: E402
    QualityAwareStandardRoIHead)


class FakeSamplingResult:

    def __init__(self, pos_priors, neg_priors, gt_bboxes, labels):
        self.pos_priors = pos_priors
        self.neg_priors = neg_priors
        self.pos_gt_bboxes = gt_bboxes
        self.pos_gt_labels = labels

    @property
    def priors(self):
        return torch.cat([self.pos_priors, self.neg_priors], dim=0)


def build_quality_head():
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(
        'configs/auxdet/auxdet_r50_fpn_1x_voc_quality.py')
    head_cfg = cfg.model.roi_head.copy()
    # Config inheritance already supplies the extractor and bbox head fields.
    full_cfg = Config.fromfile(
        'configs/auxdet/auxdet_r50_fpn_1x_voc.py').model.roi_head.copy()
    full_cfg.update(head_cfg)
    head = MODELS.build(full_cfg)
    head.init_weights()
    return head


def test_original_standard_roi_head_still_builds():
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile('configs/auxdet/auxdet_r50_fpn_1x_voc.py')
    head = MODELS.build(cfg.model.roi_head)
    assert head.__class__.__name__ == 'StandardRoIHead'


def test_quality_head_shape_and_initial_prior():
    head = build_quality_head()
    logits = head.quality_head(torch.randn(5, 256, 7, 7))
    assert logits.shape == (5, 1)
    assert torch.allclose(logits, torch.full_like(logits, logits[0, 0]))
    assert torch.sigmoid(logits[0, 0]).item() == pytest.approx(.95, abs=1e-5)


def test_quality_targets_order_and_ranges():
    head = build_quality_head()
    result = FakeSamplingResult(
        torch.tensor([[0., 0., 10., 10.]]),
        torch.tensor([[20., 20., 30., 30.], [40., 40., 50., 50.]]),
        torch.tensor([[0., 0., 10., 10.]]), torch.tensor([0]))
    targets, positive, negative = head._quality_targets(
        torch.zeros(3, 4), [result])
    assert positive.tolist() == [True, False, False]
    assert negative.tolist() == [False, True, True]
    assert targets[0].item() == pytest.approx(1.0)
    assert torch.all(targets[1:] == 0)
    assert torch.all((targets >= 0) & (targets <= 1))
    second = FakeSamplingResult(
        torch.tensor([[2., 2., 4., 4.], [5., 5., 8., 8.]]),
        torch.empty(0, 4),
        torch.tensor([[2., 2., 4., 4.], [5., 5., 8., 8.]]),
        torch.tensor([0, 0]))
    targets2, pos2, neg2 = head._quality_targets(
        torch.zeros(5, 4), [result, second])
    assert len(targets2) == 5
    assert pos2.tolist() == [True, False, False, True, True]
    assert neg2.tolist() == [False, True, True, False, False]


def test_quality_loss_handles_no_positive_and_has_gradient():
    head = build_quality_head()
    result = FakeSamplingResult(
        torch.empty(0, 4), torch.tensor([[0., 0., 1., 1.]]),
        torch.empty(0, 4), torch.empty(0, dtype=torch.long))
    logits = torch.zeros(1, 1, requires_grad=True)
    loss = head._quality_loss(logits, torch.zeros(1, 4),
                              torch.zeros(1, 5), [result])
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_quality_iou_and_logits_are_finite_for_degenerate_inputs():
    head = build_quality_head()
    boxes = torch.tensor([[float('nan'), 0., 0., 0.]])
    target = torch.tensor([[0., 0., 1., 1.]])
    iou = head._aligned_iou(boxes, target)
    assert torch.isfinite(iou).all() and torch.all((iou >= 0) & (iou <= 1))
    fused = head._fuse_cls_scores(
        torch.tensor([[float('nan'), 0.]]), torch.tensor([[float('inf')]]), .5)
    assert torch.isfinite(fused).all()


def test_score_fusion_is_before_bbox_head_nms_contract():
    cls_logits = torch.tensor([[2., 0.], [0., 2.]])
    quality_logits = torch.tensor([[0.], [2.]])
    fused = QualityAwareStandardRoIHead._fuse_cls_scores(
        cls_logits, quality_logits, alpha=.5)
    probs = fused.softmax(dim=-1)[:, 0]
    expected = cls_logits.softmax(dim=-1)[:, 0] * quality_logits.sigmoid().squeeze(1).pow(.5)
    assert torch.allclose(probs, expected, atol=1e-6)


def test_predict_bbox_passes_fused_scores_to_original_bbox_head_nms_entry():
    captured = {}

    class FakeBBoxHead:

        def predict_by_feat(self, **kwargs):
            captured['scores'] = kwargs['cls_scores'][0]
            captured['bbox_preds'] = kwargs['bbox_preds'][0]
            return ['original_bbox_head_result']

    head = QualityAwareStandardRoIHead.__new__(
        QualityAwareStandardRoIHead)
    head.quality_head = object()
    head.quality_score_fusion_enabled = True
    head.quality_alpha = .5
    head.bbox_head = FakeBBoxHead()
    head._bbox_forward = lambda x, rois: dict(
        cls_score=torch.tensor([[2., 0.], [0., 2.]]),
        bbox_pred=torch.zeros(2, 4),
        quality_logits=torch.tensor([[0.], [2.]]),
        bbox_feats=None)
    proposals = [SimpleNamespace(
        bboxes=torch.tensor([[0., 0., 10., 10.], [20., 20., 30., 30.]]))]
    output = head.predict_bbox(None, [{}], proposals, {}, rescale=False)
    assert output == ['original_bbox_head_result']
    fused_probs = captured['scores'].softmax(dim=-1)[:, 0]
    expected = torch.tensor([[2., 0.], [0., 2.]]).softmax(dim=-1)[:, 0]
    expected *= torch.tensor([.5, torch.sigmoid(torch.tensor(2.)).item()]).pow(.5)
    assert torch.allclose(fused_probs, expected, atol=1e-6)


def test_disabled_quality_has_no_quality_state_keys_and_config_builds():
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile('configs/auxdet/auxdet_r50_fpn_1x_voc.py')
    cfg.model.roi_head.type = 'QualityAwareStandardRoIHead'
    cfg.model.roi_head.quality_enabled = False
    cfg.model.roi_head.quality_score_fusion_enabled = False
    head = MODELS.build(cfg.model.roi_head)
    assert head.quality_head is None
    assert not any('quality_head' in key for key in head.state_dict())
