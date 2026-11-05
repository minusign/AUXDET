"""Diagnose one real training batch; never update weights or write checkpoints.

Run from the repository root with PYTHONPATH=. and a trusted local checkpoint.
The default config diagnoses the existing post_topdown experiment. Pass the
midfpn config explicitly to diagnose the new position with the same weights.
"""

import argparse
import copy

import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner, load_checkpoint, set_random_seed

from mmdet.registry import MODELS
from mmdet.utils import register_all_modules


def gradient_report(label, gradients):
    present = [g for g in gradients if g is not None]
    finite = all(torch.isfinite(g).all().item() for g in present)
    norm = sum(g.detach().float().square().sum().item()
               for g in present) ** 0.5
    print(f'{label}: tensors={len(present)}/{len(gradients)}, '
          f'finite={finite}, L2={norm:.6e}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', help='Existing local epoch_12.pth')
    parser.add_argument(
        '--config', default=(
            'configs/auxdet/auxdet_r50_fpn_1x_voc_rahmoe_aux.py'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()
    if args.batch_size < 2:
        parser.error('--batch-size must be >=2 for metadata BatchNorm1d')

    register_all_modules(init_default_scope=True)
    set_random_seed(args.seed)
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    model = MODELS.build(cfg.model).to(args.device)
    # Full trained model only: fail on any missing/unexpected/shape keys.
    # Do not call init_weights (which could download ImageNet weights) or
    # resume an optimizer. This script only loads existing trained weights.
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=True)
    model.train()
    neck = model.neck
    if not getattr(neck, 'enable_rahmoe', False):
        raise ValueError('The diagnostic config must enable RAHMoE')
    moe = neck.rahmoe
    if not moe.compute_aux_loss:
        raise ValueError('The diagnostic config must set compute_aux_loss=True')

    loader_cfg = copy.deepcopy(cfg.train_dataloader)
    loader_cfg.update(batch_size=args.batch_size, num_workers=0,
                      persistent_workers=False)
    loader = Runner.build_dataloader(loader_cfg, seed=args.seed)
    batch = next(iter(loader))
    data = model.data_preprocessor(batch, training=True)
    print(f'config={args.config}, position={neck.rahmoe_position}')
    print(f'images={tuple(data["inputs"].shape)}, diagnostic dtype=FP32')

    # One forward with real annotations; no optimizer, scheduler or train loop.
    losses = model(**data, mode='loss')
    aux = losses.get('loss_rahmoe_aux')
    if aux is None or not aux.requires_grad:
        raise RuntimeError('loss_rahmoe_aux missing or disconnected from autograd')
    if not torch.isfinite(aux).all():
        raise RuntimeError('loss_rahmoe_aux is NaN/Inf')
    assert aux is neck.moe_aux_loss and aux is moe.aux_loss
    print(f'loss_rahmoe_aux={aux.item():.6e}, nonzero={aux.item() != 0}, '
          f'requires_grad={aux.requires_grad}, grad_fn={aux.grad_fn}')
    print(f'four-decimal display={aux.item():.4f} '
          '(a rounded zero does not prove a zero Tensor)')
    weighted_sum = 0.0
    for name, raw in moe.aux_loss_logs.items():
        weight = getattr(moe, f'{name}_weight')
        weighted = raw.item() * weight
        weighted_sum += weighted
        print(f'{name}: raw={raw.item():.6e}, weight={weight:.6e}, '
              f'weighted={weighted:.6e}')
    print('entropy raw is (mean gate entropy - entropy_target)^2')
    print(f'weighted_sum={weighted_sum:.6e}')
    torch.testing.assert_close(
        aux.detach(), aux.new_tensor(weighted_sum), rtol=1e-5, atol=1e-10)

    detection, _ = model.parse_losses({
        name: value for name, value in losses.items()
        if name != 'loss_rahmoe_aux'
    })
    total, _ = model.parse_losses(losses)
    torch.testing.assert_close(total, detection + aux)
    print(f'detection_loss={detection.item():.6e}, '
          f'total_loss={total.item():.6e}')

    # Independent autograd probes: no detach, no accumulated .grad buffers.
    groups = [('router', list(moe.router.parameters()))]
    groups += [(f'expert_{i}_{type(expert).__name__}',
                list(expert.parameters()))
               for i, expert in enumerate(moe.experts)]
    parameters = [p for _, group in groups for p in group]
    gradients = torch.autograd.grad(
        aux, parameters, retain_graph=True, allow_unused=True)
    offset = 0
    for label, group in groups:
        gradient_report('aux -> ' + label,
                        gradients[offset:offset + len(group)])
        offset += len(group)

    gamma_grads = torch.autograd.grad(
        detection, (moe.gamma_shared, moe.gamma_routed), allow_unused=True)
    for name, param, grad in zip(
            ('gamma_shared', 'gamma_routed'),
            (moe.gamma_shared, moe.gamma_routed), gamma_grads):
        print(f'{name}={param.item():.6e}')
        gradient_report('detection -> ' + name, (grad,))
    print('Done: one batch only; no parameter updates or files saved.')


if __name__ == '__main__':
    main()
