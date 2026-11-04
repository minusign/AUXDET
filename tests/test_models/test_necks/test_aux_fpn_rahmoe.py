from types import SimpleNamespace

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


def test_aux_fpn_rahmoe_switch_and_output_count():
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
    disabled = AuxFPN(**kwargs, enable_rahmoe=False).eval()
    disabled.load_state_dict(baseline.state_dict())
    assert not hasattr(disabled, 'rahmoe')
    baseline_outs = baseline(inputs, samples)
    disabled_outs = disabled(inputs, samples)
    assert len(disabled_outs) == 2
    for baseline_out, disabled_out in zip(baseline_outs, disabled_outs):
        torch.testing.assert_close(baseline_out, disabled_out)

    enabled = AuxFPN(
        **kwargs,
        enable_rahmoe=True,
        rahmoe_level=1,
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
