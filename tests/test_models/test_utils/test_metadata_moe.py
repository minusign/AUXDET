# Copyright (c) OpenMMLab. All rights reserved.

import pytest
import torch

from mmdet.models.utils.metadata_moe import (MetadataExpert, MetadataMoE,
                                              MetadataRouter)


def _make_inputs(batch_size=2, height=32, width=32):
    torch.manual_seed(123)
    x = torch.randn(batch_size, 256, height, width)
    aux_feature = torch.randn(batch_size, 128)
    return x, aux_feature


def test_metadata_router_shape_and_normalization():
    router = MetadataRouter()
    aux_feature = torch.randn(2, 128)
    weights = router(aux_feature)

    assert weights.shape == (2, 4)
    torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones(2), rtol=1e-5, atol=1e-6)


def test_metadata_expert_preserves_shape():
    expert = MetadataExpert()
    x = torch.randn(2, 256, 32, 32)
    output = expert(x)

    assert output.shape == x.shape


@pytest.mark.parametrize('batch_size', [1, 2])
def test_metadata_moe_preserves_shape(batch_size):
    moe = MetadataMoE()
    x, aux_feature = _make_inputs(batch_size=batch_size)
    output = moe(x, aux_feature)

    assert moe.gamma.item() == pytest.approx(0.01)
    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert output.device == x.device


def test_router_parameters_receive_gradient():
    router = MetadataRouter()
    aux_feature = torch.randn(2, 128, requires_grad=True)
    weights = router(aux_feature)
    objective = (weights * torch.arange(1., 5.)).sum()
    objective.backward()

    assert aux_feature.grad is not None
    assert torch.isfinite(aux_feature.grad).all()
    assert any(param.grad is not None for param in router.parameters())
    assert all(torch.isfinite(param.grad).all()
               for param in router.parameters() if param.grad is not None)


def test_all_dense_experts_receive_gradient():
    moe = MetadataMoE()
    x, aux_feature = _make_inputs()
    x.requires_grad_()
    output = moe(x, aux_feature)
    output.mean().backward()

    for expert in moe.experts:
        expert_grads = [
            param.grad for param in expert.parameters() if param.requires_grad
        ]
        assert expert_grads
        assert all(grad is not None for grad in expert_grads)
        assert all(torch.isfinite(grad).all() for grad in expert_grads)
        assert all(grad.abs().sum() > 0 for grad in expert_grads)


def test_aux_feature_receives_gradient_through_moe():
    moe = MetadataMoE()
    x, aux_feature = _make_inputs()
    aux_feature.requires_grad_()
    output = moe(x, aux_feature)
    output.square().mean().backward()

    assert aux_feature.grad is not None
    assert torch.isfinite(aux_feature.grad).all()
    assert aux_feature.grad.abs().sum() > 0


def test_moe_output_has_no_nan_or_inf():
    moe = MetadataMoE()
    x, aux_feature = _make_inputs()
    output = moe(x, aux_feature)

    assert torch.isfinite(output).all()


def test_seed_reproducibility_and_seed_sensitivity():
    torch.manual_seed(7)
    first = MetadataMoE()
    x, aux_feature = _make_inputs()
    first_output = first(x, aux_feature)

    torch.manual_seed(7)
    second = MetadataMoE()
    second_output = second(x, aux_feature)
    torch.testing.assert_close(first_output, second_output)

    torch.manual_seed(8)
    third = MetadataMoE()
    third_output = third(x, aux_feature)
    assert not torch.allclose(first_output, third_output)


def test_metadata_moe_rejects_incompatible_batch_or_dtype():
    moe = MetadataMoE()
    x, aux_feature = _make_inputs()

    with pytest.raises(ValueError, match='same batch size'):
        moe(x, aux_feature[:1])

    with pytest.raises(ValueError, match='same dtype'):
        moe(x, aux_feature.double())
