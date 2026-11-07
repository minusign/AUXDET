"""MC-RAHMoE compatibility, joint routing and gradient checks."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('mmcv')
pytest.importorskip('mmengine')

from mmengine.config import Config

from mmdet.models.necks.aux_fpn import AuxFPN
from mmdet.models.necks.rahmoe import DenseResponseStateRouter, RAHMoE


def _router(**kwargs):
    return DenseResponseStateRouter(
        dim=8, metadata_conditioned_router=True, **kwargs)


def _neck(enabled=True, **kwargs):
    cfg = dict(
        in_channels=[8] * 4, out_channels=8, num_outs=4,
        enable_rahmoe=True, rahmoe_position='p2_bottleneck',
        rahmoe_level=0, rahmoe_bottleneck_dim=4,
        rahmoe_cfg=dict(metadata_conditioned_router=enabled,
                        meta_feature_dim=4, compute_aux_loss=True))
    cfg.update(kwargs)
    return AuxFPN(**cfg)


def _inputs():
    return tuple(torch.randn(2, 8, s, s) for s in (16, 8, 4, 2))


def _samples():
    return [SimpleNamespace(view='Air', band_type='LWIR',
                            width=1024, height=1024) for _ in range(2)]


def test_visual_router_matches_original_formula_and_state_keys():
    router = DenseResponseStateRouter(dim=8).eval()
    x = torch.randn(2, 8, 6, 8)
    visual = router.router(router.build_cues(x))
    expected = torch.softmax(visual / router.temperature.clamp_min(.3), dim=1)
    gates, logits = router(x)
    torch.testing.assert_close(logits, visual, rtol=0, atol=0)
    torch.testing.assert_close(gates, expected, rtol=0, atol=0)
    assert not hasattr(router, 'metadata_mlp')
    assert not hasattr(router, 'meta_scale')
    assert router.route_stats == {}
    assert all(k.startswith(('feat_reduce.', 'router.')) or k == 'temperature'
               for k in router.state_dict())


@pytest.mark.parametrize('batch_size', [1, 2, 4])
def test_joint_shapes_neutral_initialization_and_single_softmax(batch_size):
    visual = DenseResponseStateRouter(dim=8).eval()
    joint = _router().eval()
    joint.load_state_dict(visual.state_dict(), strict=False)
    x = torch.randn(batch_size, 8, 6, 8)
    meta = torch.randn(batch_size, 128)
    assert joint.metadata_mlp(meta).shape == (batch_size, 4)
    assert torch.count_nonzero(joint.metadata_mlp(meta)) == 0
    reference_gates, reference_logits = visual(x)
    with patch.object(torch, 'softmax', wraps=torch.softmax) as softmax:
        gates, logits = joint(x, meta_feature=meta)
    assert softmax.call_count == 1
    assert gates.shape == logits.shape == (batch_size, 4, 6, 8)
    torch.testing.assert_close(gates.sum(1), torch.ones(batch_size, 6, 8))
    torch.testing.assert_close(logits, reference_logits, rtol=0, atol=0)
    torch.testing.assert_close(gates, reference_gates, rtol=0, atol=0)


def test_different_metadata_changes_logits_by_spatially_shared_bias():
    router = _router().eval()
    # Simulate a learned final layer; zero initialization must be neutral.
    with torch.no_grad():
        router.metadata_mlp[-1].weight.normal_(std=.1)
    x = torch.randn(2, 8, 6, 8)
    meta_a, meta_b = torch.randn(2, 128), torch.randn(2, 128)
    gates_a, logits_a = router(x, meta_a)
    gates_b, logits_b = router(x, meta_b)
    expected_bias = router.meta_scale * (
        router.metadata_mlp(meta_b) - router.metadata_mlp(meta_a))
    torch.testing.assert_close(
        logits_b - logits_a,
        expected_bias[:, :, None, None].expand_as(logits_a),
        rtol=1e-4, atol=1e-6)
    assert not torch.allclose(gates_a, gates_b)
    router.set_temperature(.1)
    torch.testing.assert_close(router.temperature, torch.tensor(.3))


def test_metadata_gradient_after_two_optimizer_steps():
    router = _router()
    x = torch.randn(2, 8, 6, 8)
    meta = torch.randn(2, 128, requires_grad=True)
    optimizer = torch.optim.SGD(router.parameters(), lr=.1)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        meta.grad = None
        gates, _ = router(x, meta)
        loss = -gates[:, 0].clamp_min(1e-8).log().mean()
        loss.backward()
        assert router.metadata_mlp[-1].weight.grad.abs().sum() > 0
        if step == 0:
            assert meta.grad.count_nonzero() == 0
            assert router.metadata_mlp[1].weight.grad.count_nonzero() == 0
        else:
            for p in router.metadata_mlp.parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all()
                assert p.grad.abs().sum() > 0
            assert meta.grad.abs().sum() > 0
            assert router.meta_scale.grad.abs() > 0
        optimizer.step()


def test_missing_or_invalid_metadata_rejected_only_when_enabled():
    router = _router()
    x = torch.randn(2, 8, 6, 8)
    for meta in (None, torch.randn(1, 128), torch.randn(2, 127)):
        with pytest.raises(ValueError, match='meta_feature'):
            router(x, meta)
    visual = DenseResponseStateRouter(dim=8)
    visual(x)


def test_moe_experts_once_and_aux_receives_joint_logits():
    moe = RAHMoE(dim=8, metadata_conditioned_router=True)
    with torch.no_grad():
        moe.router.metadata_mlp[-1].weight.normal_(std=.1)
    x = torch.randn(2, 8, 6, 8, requires_grad=True)
    meta = torch.randn(2, 128, requires_grad=True)
    counts = [0] * 5

    def counter(index):
        def hook(module, args, out):
            counts[index] += 1
        return hook

    handles = [module.register_forward_hook(counter(i))
               for i, module in enumerate([moe.shared, *moe.experts])]
    with patch.object(moe, '_compute_aux', wraps=moe._compute_aux) as aux:
        out, vis = moe(x, return_vis=True, meta_feature=meta)
    for handle in handles:
        handle.remove()
    assert counts == [1] * 5
    assert aux.call_count == 1
    torch.testing.assert_close(aux.call_args.args[0], vis['route_weights'])
    torch.testing.assert_close(aux.call_args.args[1], vis['router_logits'])
    assert set(moe.aux_loss_logs) == {
        'balance', 'diversity', 'entropy', 'zloss'}
    assert torch.isfinite(moe.aux_loss) and moe.aux_loss.grad_fn is not None
    (out.square().mean() + moe.aux_loss).backward()
    for module in [moe.shared, *moe.experts, moe.router.metadata_mlp]:
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in module.parameters())
    assert meta.grad.abs().sum() > 0
    assert moe.route_logs == {}


def test_rahmoe_zero_metadata_matches_existing_output_and_aux():
    old = RAHMoE(dim=8)
    new = RAHMoE(dim=8, metadata_conditioned_router=True)
    missing = new.load_state_dict(old.state_dict(), strict=False)
    assert not missing.unexpected_keys
    assert all(k.startswith('router.metadata_mlp.') or k == 'router.meta_scale'
               for k in missing.missing_keys)
    x = torch.randn(2, 8, 6, 8)
    expected = old(x)
    actual = new(x, meta_feature=torch.randn(2, 128))
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    torch.testing.assert_close(old.aux_loss, new.aux_loss, rtol=0, atol=0)
    disabled = RAHMoE(dim=8, metadata_conditioned_router=False)
    disabled.load_state_dict(old.state_dict(), strict=True)
    assert set(disabled.state_dict()) == set(old.state_dict())


def test_neck_passes_idx_zero_feature_without_detach_or_cache():
    neck = _neck().train()
    recorded = []
    captured = {}

    def record(module, args, out):
        out.retain_grad()
        recorded.append(out)

    handle = neck.meta_process_with_sem.register_forward_hook(record)
    forward = neck.rahmoe.forward

    def capture(x, return_vis=False, meta_feature=None):
        captured['meta'] = meta_feature
        return forward(x, return_vis=return_vis, meta_feature=meta_feature)

    with patch.object(neck.rahmoe, 'forward', side_effect=capture) as moe:
        outputs = neck(_inputs(), _samples())
    handle.remove()
    assert moe.call_count == 1
    assert len(recorded) == 2
    assert captured['meta'] is recorded[0]
    assert captured['meta'] is not recorded[1]
    assert captured['meta'].shape == (2, 4)
    assert captured['meta'].grad_fn is not None
    assert not hasattr(neck, 'p2_aux_feature')
    assert [tuple(o.shape) for o in outputs] == [(2, 8, 16, 16), (2, 8, 8, 8)]
    assert neck.moe_aux_loss is not None
    (sum(o.square().mean() for o in outputs) + neck.moe_aux_loss).backward()
    assert recorded[0].grad is not None
    assert torch.isfinite(recorded[0].grad).all()


def test_diagnostics_are_opt_in_detached_small_statistics():
    moe = RAHMoE(dim=8, metadata_conditioned_router=True,
                 route_diagnostics=True)
    moe(torch.randn(2, 8, 6, 8), meta_feature=torch.randn(2, 128))
    logs = moe.route_logs
    assert set(logs) == {'visual_logits_mean', 'visual_logits_std',
                         'meta_logits_mean', 'meta_logits_std', 'meta_scale',
                         'expert_usage', 'entropy'}
    assert logs['expert_usage'].shape == (4,)
    assert all(not v.requires_grad and v.numel() <= 4 for v in logs.values())
    assert logs['meta_logits_std'] == 0
    moe.router.route_diagnostics = False
    moe(torch.randn(2, 8, 6, 8), meta_feature=torch.randn(2, 128))
    assert moe.route_logs == {} and moe.router.route_stats == {}


def test_metadata_swap_keeps_visual_and_m2dm_fixed():
    neck = _neck().eval()
    with torch.no_grad():
        neck.rahmoe.router.metadata_mlp[-1].weight.normal_(std=.1)
    captured = {}
    forward = neck.rahmoe.forward

    def capture(x, return_vis=False, meta_feature=None):
        captured.update(x=x, meta=meta_feature)
        return forward(x, return_vis=return_vis, meta_feature=meta_feature)

    with torch.no_grad(), patch.object(neck.rahmoe, 'forward',
                                      side_effect=capture):
        neck(_inputs(), _samples())
        router = neck.rahmoe.router
        before, _ = router(captured['x'], captured['meta'])
        after, _ = router(captured['x'], captured['meta'].roll(1, 0))
    assert not torch.allclose(before, after)


def test_configs_and_old_heatmap_modes_build():
    from mmdet.registry import MODELS

    config_dir = Path(__file__).resolve().parents[4] / 'configs' / 'auxdet'
    old = Config.fromfile(str(
        config_dir / 'auxdet_r50_fpn_1x_voc_rahmoe_p2.py'))
    new = Config.fromfile(str(
        config_dir / 'auxdet_r50_fpn_1x_voc_rahmoe_p2_meta.py'))
    assert not new.model.neck.heatmap_enabled
    assert new.model.neck.rahmoe_cfg.metadata_conditioned_router
    assert new.train_cfg.max_epochs == 12 and new.train_cfg.val_interval == 1
    assert new.param_scheduler[1].end == 12
    assert new.param_scheduler[1].milestones == [8, 11]
    assert new.load_from is None and not new.resume
    assert new.model.backbone.init_cfg.checkpoint == 'torchvision://resnet50'
    for key in ('optim_wrapper', 'train_dataloader', 'val_dataloader',
                'test_dataloader', 'val_evaluator', 'test_evaluator'):
        assert old[key] == new[key]
    neck = MODELS.build(new.model.neck)
    neck.init_weights()
    assert neck.rahmoe.router.metadata_mlp[-1].weight.count_nonzero() == 0
    for filename in (
        'auxdet_r50_fpn_1x_voc.py', 'auxdet_r50_fpn_1x_voc_rahmoe_p2.py',
        'auxdet_r50_fpn_1x_voc_rahmoe_aux_midfpn.py',
        'auxdet_r50_fpn_1x_voc_rahmoe_p2_heat.py',
        'auxdet_r50_fpn_1x_voc_baseline_heat.py',
    ):
        cfg = Config.fromfile(str(config_dir / filename))
        built = MODELS.build(cfg.model.neck)
        if built.enable_rahmoe:
            assert not built.rahmoe.metadata_conditioned_router
            assert not hasattr(built.rahmoe.router, 'metadata_mlp')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_amp_combined_router_and_metadata_gradients():
    moe = RAHMoE(dim=8, metadata_conditioned_router=True).cuda()
    with torch.no_grad():
        moe.router.metadata_mlp[-1].weight.normal_(std=.05)
    meta = torch.randn(2, 128, device='cuda', requires_grad=True)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        out = moe(torch.randn(2, 8, 6, 8, device='cuda'), meta_feature=meta)
        total = out.float().square().mean() + moe.aux_loss
    assert torch.isfinite(total)
    total.backward()
    assert torch.isfinite(meta.grad).all() and meta.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in moe.parameters()
               if p.grad is not None)
