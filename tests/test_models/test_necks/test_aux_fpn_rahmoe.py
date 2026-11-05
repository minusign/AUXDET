from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('mmcv')
pytest.importorskip('mmengine')

from mmengine.config import Config

from mmdet.models.necks.rahmoe import RAHMoE


def _samples(batch_size=2):
    return [
        SimpleNamespace(
            view='Air', band_type='LWIR', width=1024, height=1024)
        for _ in range(batch_size)
    ]


def _aux_inputs(batch_size=2, channels=8):
    return (
        torch.randn(batch_size, channels, 16, 16),
        torch.randn(batch_size, channels, 8, 8),
        torch.randn(batch_size, channels, 4, 4),
        torch.randn(batch_size, channels, 2, 2),
    )


def test_rahmoe_dim_256_shape_and_dense_router():
    module = RAHMoE(dim=256, compute_aux_loss=False)
    x = torch.randn(2, 256, 8, 8)
    out = module(x)

    assert out.shape == x.shape
    gates, _ = module.router(x)
    assert gates.shape == (2, 4, 8, 8)
    torch.testing.assert_close(gates.sum(dim=1), torch.ones(2, 8, 8))


def test_rahmoe_gradients_reach_shared_experts_router_and_gammas():
    module = RAHMoE(dim=256, compute_aux_loss=False)
    x = torch.randn(2, 256, 8, 8, requires_grad=True)
    module(x).mean().backward()

    assert module.shared.net[0].weight.grad is not None
    for expert in module.experts:
        assert any(p.grad is not None for p in expert.parameters())
    assert any(p.grad is not None for p in module.router.parameters())
    assert module.gamma_shared.grad is not None
    assert module.gamma_routed.grad is not None


def test_rahmoe_aux_loss_switch():
    x = torch.randn(2, 256, 8, 8)

    disabled = RAHMoE(dim=256, compute_aux_loss=False)
    disabled._compute_aux = lambda *args, **kwargs: pytest.fail(
        '_compute_aux must not run when compute_aux_loss=False')
    disabled(x)
    assert disabled.aux_loss is None

    enabled = RAHMoE(dim=256, compute_aux_loss=True)
    enabled(x)
    assert enabled.aux_loss is not None
    assert torch.isfinite(enabled.aux_loss)

    enabled.eval()
    enabled(x)
    assert enabled.aux_loss is None


def test_two_stage_loss_exposes_rahmoe_aux_loss():
    from mmdet.models.detectors.two_stage import TwoStageDetector

    class Sample:
        def get(self, key, default=None):
            return 'proposals' if key == 'proposals' else default

    class RoiHead:
        def loss(self, x, proposals, data_samples):
            return dict(loss_roi=torch.tensor(2.0))

    detector = TwoStageDetector.__new__(TwoStageDetector)
    torch.nn.Module.__init__(detector)
    detector.extract_feat = lambda inputs, samples: ('feature',)
    detector.neck = type('Neck', (), {
        'moe_aux_loss': torch.tensor(0.25, requires_grad=True)
    })()
    detector.roi_head = RoiHead()
    losses = detector.loss(torch.tensor(0.0), [Sample()])

    assert losses['loss_roi'].item() == 2.0
    assert losses['loss_rahmoe_aux'] is detector.neck.moe_aux_loss

    detector.neck.moe_aux_loss = None
    losses_without_aux = detector.loss(torch.tensor(0.0), [Sample()])
    assert 'loss_rahmoe_aux' not in losses_without_aux


@pytest.mark.parametrize('position', ['post_topdown', 'mid_topdown'])
def test_aux_fpn_rahmoe_switch_and_output_count(position):
    from mmdet.models.necks.aux_fpn import AuxFPN

    kwargs = dict(
        in_channels=[8, 8, 8, 8],
        out_channels=256,
        num_outs=4,
        start_level=0,
        end_level=3,
    )
    inputs = _aux_inputs()
    samples = _samples()

    baseline = AuxFPN(**kwargs, enable_rahmoe=False).eval()
    disabled = AuxFPN(**kwargs, enable_rahmoe=False,
                      rahmoe_position=position).eval()
    disabled.load_state_dict(baseline.state_dict())
    assert not hasattr(disabled, 'rahmoe')
    baseline_outs = baseline(inputs, samples)
    disabled_outs = disabled(inputs, samples)
    assert len(disabled_outs) == 2
    for baseline_out, disabled_out in zip(baseline_outs, disabled_outs):
        torch.testing.assert_close(baseline_out, disabled_out)
    assert disabled.moe_aux_loss is None

    enabled = AuxFPN(
        **kwargs,
        enable_rahmoe=True,
        rahmoe_level=1,
        rahmoe_position=position,
        rahmoe_cfg=dict(compute_aux_loss=False),
    ).eval()
    enabled_outs = enabled(inputs, samples)
    assert len(enabled_outs) == 2
    assert enabled_outs[0].shape == (2, 256, 16, 16)
    assert enabled_outs[1].shape == (2, 256, 8, 8)


def test_aux_fpn_rahmoe_config_parses_and_builds_when_dependencies_available():
    from mmdet.registry import MODELS
    import mmdet.models.necks  # noqa: F401

    config_path = 'configs/auxdet/auxdet_r50_fpn_1x_voc_rahmoe.py'
    cfg = Config.fromfile(config_path)
    assert cfg.model.neck.enable_rahmoe is True
    assert cfg.model.neck.rahmoe_level == 1
    assert cfg.model.neck.rahmoe_cfg.compute_aux_loss is False
    MODELS.build(cfg.model.neck)


@pytest.mark.parametrize('upsample_cfg', [
    dict(mode='nearest'), dict(mode='nearest', scale_factor=2)
])
@pytest.mark.parametrize('position', ['post_topdown', 'mid_topdown'])
def test_rahmoe_position_propagation_and_single_call(position, upsample_cfg):
    from mmdet.models.necks.aux_fpn import AuxFPN

    class AddOne(torch.nn.Module):
        """Known perturbation to verify the actual top-down data flow."""
        def __init__(self):
            super().__init__()
            self.shift = torch.nn.Parameter(torch.tensor(1.0))
            self.calls = 0
            self.aux_loss = None

        def forward(self, x):
            self.calls += 1
            assert x.shape == (2, 8, 8, 8)
            return x + self.shift

    kwargs = dict(in_channels=[8] * 4, out_channels=8, num_outs=4,
                  upsample_cfg=upsample_cfg)
    baseline = AuxFPN(**kwargs).eval()
    enabled = AuxFPN(**kwargs, enable_rahmoe=True,
                     rahmoe_position=position).eval()
    enabled.rahmoe = AddOne()
    # Keep every original weight identical; replace only output convolutions
    # with identity to observe the exact lateral perturbation before them.
    result = enabled.load_state_dict(baseline.state_dict(), strict=False)
    assert result.missing_keys == ['rahmoe.shift']
    assert not result.unexpected_keys
    baseline.fpn_convs = torch.nn.ModuleList([
        torch.nn.Identity() for _ in range(4)])
    enabled.fpn_convs = torch.nn.ModuleList([
        torch.nn.Identity() for _ in range(4)])
    inputs, samples = _aux_inputs(), _samples()
    with torch.no_grad():
        reference = baseline(inputs, samples)
    outs = enabled(inputs, samples)
    assert enabled.rahmoe.calls == 1
    assert len(outs) == 2
    torch.testing.assert_close(outs[1], reference[1] + 1,
                               atol=1e-5, rtol=1e-5)
    expected_p2 = reference[0] + (1 if position == 'mid_topdown' else 0)
    torch.testing.assert_close(outs[0], expected_p2, atol=1e-5, rtol=1e-5)
    # P2 depends on the MoE perturbation only with the new position.
    grad, = torch.autograd.grad(outs[0].sum(), enabled.rahmoe.shift,
                                allow_unused=True)
    if position == 'mid_topdown':
        assert grad is not None and torch.isfinite(grad) and grad != 0
    else:
        assert grad is None


def test_auxiliary_loss_alone_reaches_router_and_experts():
    module = RAHMoE(dim=8, compute_aux_loss=True).train()
    module(torch.randn(2, 8, 8, 8))
    aux = module.aux_loss
    assert aux.grad_fn is not None
    assert torch.isfinite(aux) and aux > 0
    aux.backward()
    for group in (module.router, *module.experts):
        grads = [p.grad for p in group.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads) > 0
    assert set(module.aux_loss_logs) == {
        'balance', 'diversity', 'entropy', 'zloss'}
    assert all(not v.requires_grad for v in module.aux_loss_logs.values())


def test_midfpn_config_preserves_training_and_state_dict():
    from pathlib import Path

    from mmdet.registry import MODELS

    root = Path(__file__).resolve().parents[4]
    config_dir = root / 'configs' / 'auxdet'
    old = Config.fromfile(str(
        config_dir / 'auxdet_r50_fpn_1x_voc_rahmoe_aux.py'))
    new = Config.fromfile(str(
        config_dir / 'auxdet_r50_fpn_1x_voc_rahmoe_aux_midfpn.py'))
    assert new.model.neck.rahmoe_position == 'mid_topdown'
    assert new.log_processor.num_digits == 8
    assert new.load_from is None and new.resume is False
    assert new.model.backbone.init_cfg.checkpoint == 'torchvision://resnet50'
    assert new.model.neck.rahmoe_cfg == old.model.neck.rahmoe_cfg
    for key in ('train_cfg', 'optim_wrapper', 'param_scheduler',
                'train_dataloader', 'val_evaluator'):
        assert new[key] == old[key]
    previous, current = MODELS.build(old.model.neck), MODELS.build(new.model.neck)
    current.load_state_dict(previous.state_dict(), strict=True)
    assert previous.rahmoe_position == 'post_topdown'
    assert current.rahmoe_position == 'mid_topdown'


def test_invalid_rahmoe_position_raises():
    from mmdet.models.necks.aux_fpn import AuxFPN

    with pytest.raises(ValueError, match='rahmoe_position'):
        AuxFPN(in_channels=[8] * 4, out_channels=8, num_outs=4,
               rahmoe_position='unknown')


@pytest.mark.parametrize('position', ['post_topdown', 'mid_topdown'])
def test_eval_rahmoe_bypass_only_skips_eval_call(position):
    from mmdet.models.necks.aux_fpn import AuxFPN

    model = AuxFPN(
        in_channels=[8] * 4,
        out_channels=8,
        num_outs=4,
        enable_rahmoe=True,
        rahmoe_position=position,
        rahmoe_eval_bypass=False,
    )
    inputs, samples = _aux_inputs(channels=8), _samples()
    state_keys = tuple(model.state_dict().keys())

    model.eval()
    with patch.object(model.rahmoe, 'forward',
                      wraps=model.rahmoe.forward) as forward:
        outputs = model(inputs, samples)
    assert forward.call_count == 1
    assert len(outputs) == 2

    model.rahmoe_eval_bypass = True
    with patch.object(model.rahmoe, 'forward',
                      wraps=model.rahmoe.forward) as forward:
        bypass_outputs = model(inputs, samples)
    assert forward.call_count == 0
    assert [tuple(x.shape) for x in bypass_outputs] == [
        tuple(x.shape) for x in outputs]
    assert model.moe_aux_loss is None

    model.train()
    with patch.object(model.rahmoe, 'forward',
                      wraps=model.rahmoe.forward) as forward:
        train_outputs = model(inputs, samples)
    assert forward.call_count == 1
    assert [tuple(x.shape) for x in train_outputs] == [
        tuple(x.shape) for x in outputs]
    assert tuple(model.state_dict().keys()) == state_keys


def test_eval_bypass_default_is_disabled_and_does_not_change_state_dict():
    from mmdet.models.necks.aux_fpn import AuxFPN

    model = AuxFPN(in_channels=[8] * 4, out_channels=8, num_outs=4,
                   enable_rahmoe=True)
    assert model.rahmoe_eval_bypass is False
    before = tuple(model.state_dict().keys())
    model.rahmoe_eval_bypass = True
    after = tuple(model.state_dict().keys())
    assert before == after
