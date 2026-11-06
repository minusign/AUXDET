"""Small synthetic checks; no dataset download, training run or checkpoint."""
from pathlib import Path
from unittest.mock import patch

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('mmcv')
pytest.importorskip('mmengine')

from mmengine.config import Config
from mmengine.structures import InstanceData

from mmdet.models.necks.aux_fpn import AuxFPN
from mmdet.models.necks.target_heatmap import (TargetHeatmapHead,
                                              TargetHeatmapLoss,
                                              build_target_heatmaps)
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import HorizontalBoxes


def _sample(boxes=(), img_shape=(64, 96), box_type=False):
    sample = DetDataSample(metainfo=dict(
        img_shape=img_shape, pad_shape=(64, 96),
        batch_input_shape=(64, 96), scale_factor=(2., 2.),
        view='Air', band_type='LWIR', width=96, height=64))
    boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    sample.gt_instances = InstanceData(
        bboxes=HorizontalBoxes(boxes) if box_type else boxes,
        labels=torch.zeros(len(boxes), dtype=torch.long))
    return sample


def _inputs():
    return tuple(torch.randn(2, 8, h, w)
                 for h, w in ((16, 24), (8, 12), (4, 6), (2, 3)))


def _neck(moe, **overrides):
    cfg = dict(
        in_channels=[8] * 4, out_channels=8, num_outs=4,
        heatmap_enabled=True, heatmap_hidden_channels=4,
        heatmap_gate_enabled=moe, enable_rahmoe=moe)
    if moe:
        cfg.update(rahmoe_position='p2_bottleneck', rahmoe_level=0,
                   rahmoe_bottleneck_dim=4,
                   rahmoe_cfg=dict(compute_aux_loss=True))
    cfg.update(overrides)
    return AuxFPN(**cfg)


def _samples():
    return [_sample([[20, 12, 28, 20]]), _sample()]


def test_head_shape_zero_initialization_and_first_step_gradient():
    head = TargetHeatmapHead(4, 4)
    head.init_weights()
    features = torch.randn(2, 4, 8, 12, requires_grad=True)
    logits = head(features)
    assert logits.shape == (2, 1, 8, 12)
    torch.testing.assert_close(logits.sigmoid(), torch.full_like(logits, .5))
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                   for m in head.modules())
    targets = torch.zeros_like(logits)
    targets[:, :, 2, 3] = 1
    loss = TargetHeatmapLoss()(logits, targets,
                               torch.ones_like(targets, dtype=torch.bool))
    loss.backward()
    assert head.prediction.weight.grad.abs().sum() > 0
    # Exact zero prediction weights necessarily block the initial upstream
    # gradient; a separate test verifies propagation after one head update.
    assert torch.count_nonzero(features.grad) == 0


@pytest.mark.parametrize('box_type', [False, True])
def test_resized_multiple_centers_and_unchanged_gt(box_type):
    # GT is already resized by factor 2. Never apply scale_factor again.
    sample = _sample([[20, 12, 28, 20], [60, 36, 68, 44]], box_type=box_type)
    boxes = sample.gt_instances.bboxes
    original = getattr(boxes, 'tensor', boxes).clone()
    logits = torch.zeros(1, 1, 16, 24)
    target, valid = build_target_heatmaps(logits, [sample], (64, 96))
    assert target[0, 0, 4, 6] == 1
    assert target[0, 0, 10, 16] == 1
    assert target.eq(1).sum() == 2
    assert 0 < target[0, 0, 4, 5] < 1
    assert valid.all()
    torch.testing.assert_close(getattr(boxes, 'tensor', boxes), original)


def test_empty_invalid_tiny_and_border_boxes_with_padding():
    boxes = [[-.5, -.5, .1, .1], [62, 30, 70, 40],
             [100, 100, 102, 102], [8, 8, 4, 4], [3, 3, 3, 4],
             [float('nan'), 1, 4, 4], [1, 1, float('inf'), 3]]
    sample = _sample(boxes, img_shape=(32, 64))
    logits = torch.zeros(2, 1, 16, 24)
    target, valid = build_target_heatmaps(
        logits, [sample, _sample(img_shape=(32, 64))], (64, 96))
    assert target[0, 0, 0, 0] == 1
    assert target[0, 0, 7, 15] == 1
    assert target[0].eq(1).sum() == 2
    assert target[1].count_nonzero() == 0
    assert valid[:, :, :8, :16].all()
    assert not valid[:, :, 8:, :].any()
    assert not valid[:, :, :, 16:].any()
    assert target[~valid].count_nonzero() == 0


def test_batch_tensor_size_not_individual_pad_shape_controls_mapping():
    sample = _sample([[40, 20, 48, 28]], img_shape=(31, 61))
    sample.set_metainfo(dict(pad_shape=(32, 64)))
    target, valid = build_target_heatmaps(
        torch.zeros(1, 1, 16, 24), [sample], (64, 96))
    assert target[0, 0, 6, 11] == 1
    assert valid.sum() == 8 * 16


def test_radius_bounds_and_colliding_centers_use_maximum():
    sample = _sample([[0, 0, 96, 64], [46, 30, 50, 34]])
    target, _ = build_target_heatmaps(
        torch.zeros(1, 1, 16, 24), [sample], (64, 96),
        min_radius=0, max_radius=1)
    assert target.eq(1).sum() == 1
    assert target[0, 0, 8, 12] == 1
    assert target.count_nonzero() <= 9


def test_loss_normalization_empty_images_padding_and_extreme_fp16():
    loss_fn = TargetHeatmapLoss()
    logits = torch.zeros(2, 1, 4, 6, requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, 0, 1, 1] = 1
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[:, :, 3, :] = False
    loss = loss_fn(logits, target, valid)
    # At p=.5 a positive/negative mean is .25*log(2), independently
    # of background pixel count. The empty image has only a negative term.
    expected = .05 * .25 * torch.log(torch.tensor(2.)) * 1.5
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[~valid].count_nonzero() == 0
    assert logits.grad[valid].abs().sum() > 0

    half_logits = torch.tensor([-100., 100., 0., -20., 20., 0.],
                               dtype=torch.float16).reshape(1, 1, 2, 3)
    half_logits.requires_grad_()
    value = loss_fn(half_logits, torch.zeros_like(half_logits),
                    torch.ones_like(half_logits, dtype=torch.bool))
    assert value.dtype == torch.float32 and torch.isfinite(value)
    value.backward()
    assert torch.isfinite(half_logits.grad).all()


def test_neck_init_keeps_prediction_zero_and_old_p2_detection_output():
    old = _neck(True, heatmap_enabled=False).eval()
    assert not old.heatmap_gate_enabled
    new = _neck(True).eval()
    new.init_weights()
    assert new.target_heatmap_head.prediction.weight.count_nonzero() == 0
    assert new.target_heatmap_head.prediction.bias.count_nonzero() == 0
    incompatible = new.load_state_dict(old.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(k.startswith('target_heatmap_head.')
               for k in incompatible.missing_keys)
    inputs, samples = _inputs(), _samples()
    with torch.no_grad():
        expected = old(inputs, samples)
        actual = new(inputs, samples)
    for a, b in zip(expected, actual):
        torch.testing.assert_close(a, b)


def test_head_reads_moe_output_once_and_gate_only_changes_p2():
    model = _neck(True).eval()
    inputs, samples = _inputs(), _samples()
    captured = {}
    moe_hook = model.rahmoe.register_forward_hook(
        lambda module, args, out: captured.update(moe_out=out))
    head_hook = model.target_heatmap_head.register_forward_pre_hook(
        lambda module, args: captured.update(head_input=args[0]))
    with patch.object(model.rahmoe, 'forward',
                      wraps=model.rahmoe.forward) as moe_forward:
        with torch.no_grad():
            initial = model(inputs, samples)
        assert moe_forward.call_count == 1
    assert captured['moe_out'] is captured['head_input']
    moe_hook.remove()
    head_hook.remove()
    with torch.no_grad():
        model.target_heatmap_head.prediction.bias.fill_(2.)
        gated = model(inputs, samples)
    assert not torch.allclose(initial[0], gated[0])
    torch.testing.assert_close(initial[1], gated[1])


@pytest.mark.parametrize('moe', [False, True])
def test_heatmap_only_loss_reaches_moe_or_fpn_after_zero_head_warmup(moe):
    model = _neck(moe).train()
    inputs, samples = _inputs(), _samples()
    # One head update unlocks upstream gradients after exact zero init.
    optimizer = torch.optim.SGD(model.target_heatmap_head.parameters(), lr=.1)
    model(inputs, samples)
    first = model.compute_target_heatmap_loss(samples, (64, 96))
    first.backward()
    optimizer.step()
    model.zero_grad(set_to_none=True)

    model(inputs, samples)
    loss = model.compute_target_heatmap_loss(samples, (64, 96))
    assert loss.grad_fn is not None and torch.isfinite(loss)
    assert model.heatmap_logits is None
    assert model.compute_target_heatmap_loss(samples, (64, 96)) is None
    loss.backward()
    modules = [model.p2_down, model.target_heatmap_head,
               model.lateral_convs[0]]
    if moe:
        assert model.moe_aux_loss is not None
        modules += [model.rahmoe.router, model.rahmoe.shared,
                    *model.rahmoe.experts]
    else:
        assert not hasattr(model, 'rahmoe')
        assert model.moe_aux_loss is None
    for module in modules:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0


@pytest.mark.parametrize('moe', [False, True])
def test_eval_needs_no_gt_and_baseline_skips_head(moe):
    model = _neck(moe).eval()
    samples = _samples()
    for sample in samples:
        del sample.gt_instances
    with patch.object(model.target_heatmap_head, 'forward',
                      wraps=model.target_heatmap_head.forward) as head:
        with torch.no_grad():
            outputs = model(_inputs(), samples)
    assert len(outputs) == 2
    assert outputs[0].shape == (2, 8, 16, 24)
    assert outputs[1].shape == (2, 8, 8, 12)
    assert head.call_count == int(moe)
    assert model.heatmap_logits is None
    assert model.compute_target_heatmap_loss(samples, (64, 96)) is None


def test_p2_bypass_skips_moe_projections_and_heatmap_gate():
    model = _neck(True, rahmoe_eval_bypass=True).eval()
    with patch.object(model.rahmoe, 'forward', side_effect=AssertionError), \
            patch.object(model.p2_down, 'forward',
                         side_effect=AssertionError), \
            patch.object(model.p2_up, 'forward', side_effect=AssertionError), \
            patch.object(model.target_heatmap_head, 'forward',
                         side_effect=AssertionError):
        with torch.no_grad():
            assert len(model(_inputs(), _samples())) == 2


def test_baseline_supervision_leaves_detection_features_unchanged():
    baseline = _neck(False, heatmap_enabled=False).train()
    control = _neck(False).train()
    control.load_state_dict(baseline.state_dict(), strict=False)
    inputs, samples = _inputs(), _samples()
    for expected, actual in zip(baseline(inputs, samples),
                                control(inputs, samples)):
        torch.testing.assert_close(expected, actual)
    assert control.heatmap_logits is not None
    # A fresh eval forward clears the previous training graph reference.
    control.eval()
    control(inputs, samples)
    assert control.heatmap_logits is None


def test_detector_preserves_detection_and_moe_losses_and_consumes_heatmap():
    from mmdet.models.detectors.two_stage import TwoStageDetector

    class RoiHead(torch.nn.Module):
        def loss(self, features, proposals, samples):
            return dict(loss_roi=sum(f.square().mean() for f in features))

    detector = TwoStageDetector.__new__(TwoStageDetector)
    torch.nn.Module.__init__(detector)
    detector.neck = _neck(True)
    detector.roi_head = RoiHead()
    inputs = _inputs()
    def extract_feat(images, samples):
        return detector.neck(inputs, samples)

    detector.extract_feat = extract_feat
    samples = _samples()
    for sample in samples:
        sample.proposals = InstanceData()
    original = samples[0].gt_instances.bboxes.clone()
    losses = detector.loss(torch.zeros(2, 3, 64, 96), samples)
    assert set(losses) == {
        'loss_roi', 'loss_rahmoe_aux', 'loss_target_heatmap'}
    assert all(torch.isfinite(loss) for loss in losses.values())
    assert losses['loss_target_heatmap'].grad_fn is not None
    assert detector.neck.heatmap_logits is None
    sum(losses.values()).backward()
    for module in (detector.neck.p2_down, detector.neck.p2_up,
                   detector.neck.rahmoe.router):
        assert any(p.grad is not None for p in module.parameters())
    assert detector.neck.p2_res_scale.grad is not None
    torch.testing.assert_close(samples[0].gt_instances.bboxes, original)


def test_disabled_heatmap_preserves_old_modes_and_state_keys():
    for position, level in [('post_topdown', 1), ('mid_topdown', 1),
                            ('p2_bottleneck', 0)]:
        kwargs = dict(in_channels=[8] * 4, out_channels=8, num_outs=4,
                      enable_rahmoe=True, rahmoe_position=position,
                      rahmoe_level=level, rahmoe_bottleneck_dim=4)
        original = AuxFPN(**kwargs).eval()
        disabled = AuxFPN(**kwargs, heatmap_enabled=False).eval()
        disabled.load_state_dict(original.state_dict(), strict=True)
        assert not hasattr(disabled, 'target_heatmap_head')
        inputs, samples = _inputs(), _samples()
        with torch.no_grad():
            expected = original(inputs, samples)
            actual = disabled(inputs, samples)
        for a, b in zip(expected, actual):
            torch.testing.assert_close(a, b)


def test_both_configs_share_training_settings_and_build():
    from mmdet.registry import MODELS

    root = Path(__file__).resolve().parents[4] / 'configs' / 'auxdet'
    a = Config.fromfile(str(root / 'auxdet_r50_fpn_1x_voc_rahmoe_p2_heat.py'))
    b = Config.fromfile(str(root / 'auxdet_r50_fpn_1x_voc_baseline_heat.py'))
    for key in ('train_cfg', 'param_scheduler', 'optim_wrapper',
                'train_dataloader', 'val_dataloader', 'test_dataloader',
                'val_evaluator', 'test_evaluator', 'randomness'):
        assert a[key] == b[key]
    assert a.work_dir != b.work_dir
    for cfg in (a, b):
        assert cfg.load_from is None and cfg.resume is False
        assert cfg.randomness.seed == 42
        assert cfg.train_cfg.max_epochs == 12
        assert cfg.train_cfg.val_interval == 1
        assert cfg.param_scheduler[1].milestones == [8, 11]
        assert cfg.param_scheduler[1].end == 12
        assert cfg.default_hooks.checkpoint.max_keep_ckpts == 2
        assert cfg.default_hooks.checkpoint.save_best == 'pascal_voc/AP50'
        assert (cfg.model.backbone.init_cfg.checkpoint
                == 'torchvision://resnet50')
        assert cfg.model.neck.heatmap_loss_weight == .05
        neck = MODELS.build(cfg.model.neck)
        assert neck.heatmap_enabled
    assert a.model.neck.rahmoe_cfg.compute_aux_loss
    assert a.model.neck.heatmap_gate_enabled
    assert not b.model.neck.enable_rahmoe
    assert not b.model.neck.heatmap_gate_enabled


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA AMP check')
def test_cuda_amp_heatmap_loss_is_finite():
    model = _neck(True).cuda().train()
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(tuple(x.cuda() for x in _inputs()), _samples())
        heat_loss = model.compute_target_heatmap_loss(_samples(), (64, 96))
        total = (sum(x.float().square().mean() for x in outputs)
                 + heat_loss + model.moe_aux_loss)
    assert heat_loss.dtype == torch.float32 and torch.isfinite(total)
    total.backward()
    assert all(torch.isfinite(p.grad).all()
               for p in model.parameters() if p.grad is not None)
