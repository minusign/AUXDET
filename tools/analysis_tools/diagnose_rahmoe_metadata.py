"""Inspect one real batch, optionally backward; never step or save weights.

Use a trusted MC-RAHMoE checkpoint to inspect a trained branch. Without
--checkpoint this checks the configured initialization (ImageNet weights may
be loaded from the normal local torchvision cache). Counterfactual routing
reuses exactly the captured visual Tensor and swaps only the Router metadata;
it neither changes image/M2DM inputs nor constitutes a retraining experiment.
"""
import argparse
import copy
from unittest.mock import patch

import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner, load_checkpoint, set_random_seed

from mmdet.registry import MODELS
from mmdet.utils import register_all_modules


def print_stats(label, values):
    print(label)
    for name, value in values.items():
        values_cpu = value.detach().float().reshape(-1).cpu().tolist()
        print(f'  {name}: ' + ', '.join(f'{v:.8e}' for v in values_cpu))


def grad_stats(name, params):
    params = list(params)
    grads = [p.grad for p in params if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):
        raise RuntimeError(f'{name}: missing or nonfinite gradients')
    norm = sum(g.float().square().sum().item() for g in grads) ** .5
    print(f'{name}: grads={len(grads)}/{len(params)}, L2={norm:.8e}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=(
        'configs/auxdet/auxdet_r50_fpn_1x_voc_rahmoe_p2_meta.py'))
    parser.add_argument('--checkpoint', help='Optional trusted MC checkpoint')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--check-backward', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.check_backward and args.batch_size == 1:
        parser.error('Existing AuxDet metadata BatchNorm requires '
                     'training B>=2')
    if args.amp and not args.device.startswith('cuda'):
        parser.error('--amp checks CUDA float16')

    register_all_modules(init_default_scope=True)
    set_random_seed(args.seed)
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    model = MODELS.build(cfg.model)
    moe = model.neck.rahmoe
    if not moe.metadata_conditioned_router or model.neck.heatmap_enabled:
        raise ValueError('Use MC-RAHMoE with heatmap disabled')
    initial_scale = cfg.model.neck.rahmoe_cfg.get('meta_router_scale_init', 1.)
    print(f'configured meta_scale init={initial_scale:.8e}')
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint,
                        map_location='cpu', strict=True)
    else:
        model.init_weights()
    model.to(args.device).train(args.check_backward)
    moe.router.route_diagnostics = True
    print(f'current meta_scale={moe.router.meta_scale.item():.8e}')

    loader_cfg = copy.deepcopy(cfg.train_dataloader if args.check_backward
                               else cfg.val_dataloader)
    loader_cfg.update(batch_size=args.batch_size, num_workers=0,
                      persistent_workers=False, drop_last=False)
    loader = Runner.build_dataloader(loader_cfg, seed=args.seed)
    data = model.data_preprocessor(next(iter(loader)),
                                   training=args.check_backward)
    captured = {}
    original_forward = moe.forward

    def capture(x, return_vis=False, meta_feature=None):
        captured.update(x=x, meta=meta_feature)
        if args.check_backward:
            meta_feature.retain_grad()
        return original_forward(x, return_vis=return_vis,
                                meta_feature=meta_feature)

    with patch.object(moe, 'forward', side_effect=capture) as forward:
        with torch.set_grad_enabled(args.check_backward):
            with torch.autocast(device_type=torch.device(args.device).type,
                                dtype=torch.float16, enabled=args.amp):
                if args.check_backward:
                    losses = model(**data, mode='loss')
                    if 'loss_target_heatmap' in losses:
                        raise RuntimeError('Heatmap must be disabled')
                    aux = losses.get('loss_rahmoe_aux')
                    if aux is None or aux.grad_fn is None:
                        raise RuntimeError('MoE auxiliary loss is '
                                           'disconnected')
                    total, _ = model.parse_losses(losses)
                else:
                    model.extract_feat(data['inputs'], data['data_samples'])
        assert forward.call_count == 1
    meta = captured['meta']
    expected = (data['inputs'].shape[0], moe.router.meta_feature_dim)
    assert tuple(meta.shape) == expected
    print(f'p2_aux_feature={tuple(meta.shape)}, grad_fn={meta.grad_fn}')
    print_stats('current routing', moe.route_logs)

    if args.check_backward:
        if not torch.isfinite(total):
            raise RuntimeError('Nonfinite total loss')
        total.backward()
        print(f'total={total.item():.8e}, loss_rahmoe_aux={aux.item():.8e}')
        grad_stats('metadata_mlp', moe.router.metadata_mlp.parameters())
        grad_stats('meta_scale', [moe.router.meta_scale])
        grad_stats('p2_down', model.neck.p2_down.parameters())
        grad_stats('p2_up', model.neck.p2_up.parameters())
        for index, expert in enumerate(moe.experts):
            grad_stats(f'expert_{index}', expert.parameters())
        if meta.grad is None or not torch.isfinite(meta.grad).all():
            raise RuntimeError('P2 auxiliary feature gradient is invalid')
        print(f'p2_aux_feature grad L2={meta.grad.float().norm().item():.8e}')
        print('No optimizer update. At zero init the MLP front layers and '
              'meta_scale can have zero gradients; two-step unit tests cover '
              'their later gradients.')

    # Diagnostic-only counterfactuals; all keep the same visual input and
    # never execute M2DM, experts or heads again. No parameters change.
    router = moe.router
    x, meta = captured['x'].detach(), meta.detach()
    with torch.no_grad(), torch.autocast(
            device_type=torch.device(args.device).type,
            dtype=torch.float16, enabled=args.amp):
        normal, _ = router(x, meta)
        router.metadata_conditioned_router = False
        try:
            visual_only, _ = router(x)
        finally:
            router.metadata_conditioned_router = True
        effect = (normal.float() - visual_only.float()).abs()
        print('metadata vs visual-only gates: '
              f'mean_abs={effect.mean().item():.8e}, '
              f'max_abs={effect.max().item():.8e}')
        del visual_only, effect
        if meta.shape[0] > 1:
            swapped, _ = router(x, meta.roll(1, dims=0))
            effect = (normal.float() - swapped.float()).abs()
            print(f'swap-only gates: mean_abs={effect.mean().item():.8e}, '
                  f'max_abs={effect.max().item():.8e}')
            print('A zero difference at initialization or with identical '
                  'features is expected, not proof of a broken branch.')
        else:
            print('Swap requires B>=2; skipped for this batch.')
    captured.clear()


if __name__ == '__main__':
    main()
