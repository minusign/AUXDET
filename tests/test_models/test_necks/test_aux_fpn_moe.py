# Copyright (c) OpenMMLab. All rights reserved.

import torch

from mmdet.structures import DetDataSample
from mmdet.models.necks.aux_fpn import AuxFPN


def _make_meta(batch_size):
    return [
        DetDataSample(
            metainfo=dict(
                view='Air',
                band_type='LWIR',
                width=256,
                height=256)) for _ in range(batch_size)
    ]


def _make_inputs(batch_size=2, spatial_sizes=(16, 8, 4, 2)):
    channels = (256, 512, 1024, 2048)
    return tuple(
        torch.randn(batch_size, channel, size, size)
        for channel, size in zip(channels, spatial_sizes))


def _make_neck(enable_moe=False):
    return AuxFPN(
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4,
        start_level=0,
        end_level=3,
        enable_moe=enable_moe,
        moe_cfg=dict(
            num_experts=4,
            expert_hidden_ratio=0.25,
            temperature=1.0,
            residual_scale_init=0.01))


def test_aux_fpn_moe_disabled_preserves_interface_and_state_keys():
    reference = _make_neck(enable_moe=False).eval()
    neck = _make_neck(enable_moe=False).eval()
    neck.load_state_dict(reference.state_dict())
    inputs = _make_inputs()
    metadata = _make_meta(2)
    with torch.no_grad():
        reference_outputs = reference(inputs, metadata)
        outputs = neck(inputs, metadata)

    assert not hasattr(neck, 'meta_moes')
    assert len(outputs) == 2
    assert all(isinstance(output, torch.Tensor) for output in outputs)
    for output, reference_output in zip(outputs, reference_outputs):
        torch.testing.assert_close(output, reference_output)
    assert all(not key.startswith('meta_moes.')
               for key in neck.state_dict().keys())


def test_aux_fpn_moe_enabled_builds_two_independent_moes():
    neck = _make_neck(enable_moe=True).eval()

    assert len(neck.meta_moes) == 2
    assert neck.meta_moes[0] is not neck.meta_moes[1]
    assert neck.meta_moes[0].router_input_dim == 128
    assert neck.meta_moes[1].router_input_dim == 128
    assert neck.meta_moes[0].feature_channels == 256
    assert neck.meta_moes[1].feature_channels == 256

    first_parameter = next(neck.meta_moes[0].experts[0].parameters())
    second_parameter = next(neck.meta_moes[1].experts[0].parameters())
    assert first_parameter is not second_parameter
    assert first_parameter.data_ptr() != second_parameter.data_ptr()


def test_aux_fpn_moe_uses_both_stage_moes_and_preserves_outputs():
    neck = _make_neck(enable_moe=True).eval()
    calls = [0, 0]
    aux_shapes = [None, None]

    def make_hook(index):
        def hook(_module, inputs, _output):
            calls[index] += 1
            aux_shapes[index] = inputs[1].shape
        return hook

    handles = [
        moe.register_forward_hook(make_hook(index))
        for index, moe in enumerate(neck.meta_moes)
    ]
    try:
        outputs = neck(_make_inputs(spatial_sizes=(16, 8, 4, 2)),
                       _make_meta(2))
    finally:
        for handle in handles:
            handle.remove()

    assert calls == [1, 1]
    assert aux_shapes == [torch.Size((2, 128)), torch.Size((2, 128))]
    assert len(outputs) == 2
    assert outputs[0].shape[1] == 256
    assert outputs[1].shape[1] == 256


def test_aux_fpn_moe_gradients_reach_router_experts_and_metadata_processor():
    neck = _make_neck(enable_moe=True).train()
    outputs = neck(_make_inputs(), _make_meta(2))
    loss = sum(output.square().mean() for output in outputs)
    loss.backward()

    for moe in neck.meta_moes:
        assert moe.gamma.grad is not None
        assert torch.isfinite(moe.gamma.grad).all()
        for parameter in moe.router.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
        for expert in moe.experts:
            for parameter in expert.parameters():
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all()

    metadata_grads = [
        parameter.grad for parameter in neck.meta_process_with_sem.parameters()
    ]
    assert any(grad is not None for grad in metadata_grads)


def test_aux_fpn_moe_supports_different_spatial_sizes():
    neck = _make_neck(enable_moe=True).eval()
    outputs = neck(
        _make_inputs(spatial_sizes=(20, 10, 5, 3)), _make_meta(2))

    assert len(outputs) == 2
    assert all(torch.isfinite(output).all() for output in outputs)
    assert outputs[0].dtype == torch.float32
    assert outputs[1].dtype == torch.float32


def test_moe_config_builds_detector_from_registry():
    from mmengine import Config
    from mmengine.registry import init_default_scope

    import mmdet.models  # noqa: F401
    from mmdet.registry import MODELS

    init_default_scope('mmdet')
    config = Config.fromfile(
        'configs/auxdet/auxdet_r50_fpn_1x_voc_moe.py')
    model = MODELS.build(config.model)

    assert model.neck.enable_moe is True
    assert len(model.neck.meta_moes) == 2
