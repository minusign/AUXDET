"""Small CPU tests for the optional Meta-LFP residual gate."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[2]
VERIFY = ROOT / 'tools/analysis_tools/verify_nsfpn.py'
spec = importlib.util.spec_from_file_location('meta_lfp_verify', VERIFY)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)
_, _, aux, _ = verify.load_modules('reference')


def test_blend_endpoints_and_shapes():
    feature = torch.randn(3, 8, 5, 7)
    purified = torch.randn_like(feature)
    zero = torch.zeros(3, 1, 1, 1)
    one = torch.ones(3, 1, 1, 1)
    assert torch.equal(aux.ConditionalLFPGate.blend(feature, purified, zero), feature)
    assert torch.equal(aux.ConditionalLFPGate.blend(feature, purified, one), purified)


@pytest.mark.parametrize('mode', ['visual', 'visual_aux'])
def test_gate_batch_shape_gradients_and_initialization(mode):
    gate = aux.ConditionalLFPGate(8, 6, hidden_dim=4, mode=mode, gate_init=.1)
    feature = torch.randn(3, 8, 5, 7, requires_grad=True)
    purified = torch.randn_like(feature, requires_grad=True)
    aux_feature = torch.randn(3, 6, requires_grad=True)
    output = gate(feature, purified, aux_feature)
    assert output.shape == feature.shape
    assert gate.get_gate_stats()['mean'] == pytest.approx(.1, abs=1e-6)
    output.square().mean().backward()
    assert feature.grad is not None and purified.grad is not None
    if mode == 'visual_aux':
        # The zero initialized final layer intentionally blocks this path on
        # the first backward pass; later training updates make it active.
        assert aux_feature.grad is not None and torch.equal(aux_feature.grad, torch.zeros_like(aux_feature.grad))
    else:
        assert aux_feature.grad is None
    gradients = (feature.grad, purified.grad) + (() if aux_feature.grad is None else (aux_feature.grad,))
    assert all(torch.isfinite(value).all() for value in (output, *gradients))


def test_visual_aux_uses_aux_but_visual_does_not():
    visual = aux.ConditionalLFPGate(4, 3, hidden_dim=5, mode='visual', gate_init=.2)
    visual_aux = aux.ConditionalLFPGate(4, 3, hidden_dim=5, mode='visual_aux', gate_init=.2)
    with torch.no_grad():
        for module in (visual, visual_aux):
            module.mlp[0].weight.zero_()
            module.mlp[0].bias.zero_()
            module.mlp[0].weight[0, 4] = 1.
            module.mlp[2].weight.fill_(1.)
    feature = torch.ones(2, 4, 3, 5)
    purified = torch.zeros_like(feature)
    aux_a = torch.zeros(2, 3)
    aux_b = torch.ones(2, 3)
    visual(feature, purified, aux_a)
    visual_stats_a = visual.get_gate_stats()
    visual(feature, purified, aux_b)
    visual_stats_b = visual.get_gate_stats()
    visual_aux(feature, purified, aux_a)
    aux_stats_a = visual_aux.get_gate_stats()
    visual_aux(feature, purified, aux_b)
    aux_stats_b = visual_aux.get_gate_stats()
    assert visual_stats_a == visual_stats_b
    assert aux_stats_a != aux_stats_b


def test_new_configs_have_equal_gate_parameters_and_no_sfs():
    paths = [
        ROOT / 'configs/auxdet/auxdet_r50_lfp_p2_after_edge_visual_gate_1x_voc.py',
        ROOT / 'configs/auxdet/auxdet_r50_lfp_p2_after_edge_visual_aux_gate_1x_voc.py',
    ]
    configs = [Config.fromfile(str(path)) for path in paths]
    necks = [dict(config.model.neck) for config in configs]
    assert [neck['lfp_gate_cfg']['mode'] for neck in necks] == ['visual', 'visual_aux']
    assert all(neck['lfp_position'] == 'after_edge' and neck['lfp_levels'] == (0,)
               for neck in necks)
    assert all(neck.get('sfs_cfg') is None and neck.get('sfs_fusions', ()) == ()
               for neck in necks)
    gates = [aux.ConditionalLFPGate(256, 128, hidden_dim=64, mode=neck['lfp_gate_cfg']['mode'])
             for neck in necks]
    assert sum(parameter.numel() for parameter in gates[0].parameters()) == \
        sum(parameter.numel() for parameter in gates[1].parameters()) == 24705
