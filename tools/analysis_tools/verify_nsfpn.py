"""Check the local LFP/SFS implementation and B0--B4 neck configurations.

CPU mathematical checks (requires torch, mmcv-lite, mmengine, einops,
pytorch-wavelets, PyWavelets, scipy, and setuptools<81 for wavelets 1.3.0)::

    python tools/analysis_tools/verify_nsfpn.py --backend reference

Actual MMCV CUDA operator and normal MMDetection imports::

    python tools/analysis_tools/verify_nsfpn.py --backend cuda --full-model

The reference backend explicitly substitutes MMCV's own PyTorch attention
reference for its compiled operator. It loads the real neck, metadata, and
dynamic-MLP source files without importing unrelated detector extensions.
It does NOT verify CUDA kernels, full-package imports, or detection accuracy.
No model/configuration files are modified; no checkpoints are downloaded.
"""

import argparse
import ast
import copy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import sys
import types
import warnings

import mmcv
from mmengine.config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = [
    'auxdet_r50_fpn_1x_voc.py',
    'auxdet_r50_lfp_p2_1x_voc.py',
    'auxdet_r50_lfp_p2_after_edge_1x_voc.py',
    'auxdet_r50_sfs_p3_p2_1x_voc.py',
    'auxdet_r50_lfp_after_edge_sfs_p3_p2_1x_voc.py',
]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def definitions(path, names, namespace):
    """Load selected existing definitions, bypassing optional extension imports."""
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise RuntimeError(f'Missing reference definitions in {path}')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'),
         namespace)
    return namespace


def load_modules(backend):
    # Multiple focused test files can share this source-loading helper in one
    # pytest process. Reuse the same backend rather than registering classes
    # again. Different backends must run in separate processes.
    cached = sys.modules.get('mmdet.models.necks.aux_fpn')
    if cached is not None and hasattr(cached, '_verification_backend'):
        if cached._verification_backend != backend:
            raise RuntimeError('Run different verification backends in separate processes')
        return (sys.modules['mmdet.models.necks.lfp'],
                sys.modules['mmdet.models.necks.sfs'], cached,
                sys.modules['mmdet.structures'].DetDataSample)
    sys.path.insert(0, str(ROOT))
    if backend == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA backend requested, but CUDA is unavailable')
        from mmdet.utils import register_all_modules
        register_all_modules()
    else:
        # Only package aggregation is bypassed. All computational classes below
        # are loaded directly from the checked-out source, without replacement.
        import mmdet
        for name, path in [('mmdet.models', ROOT / 'mmdet/models'),
                           ('mmdet.models.necks', ROOT / 'mmdet/models/necks')]:
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
        load_file('mmdet.utils', ROOT / 'mmdet/utils/typing_utils.py')
        load_file('mmdet.structures', ROOT / 'mmdet/structures/det_data_sample.py')
        load_file('mmdet.models.utils', ROOT / 'mmdet/models/utils/dynamic_mlp.py')

    sfs = importlib.import_module('mmdet.models.necks.sfs')
    if backend == 'reference':
        path = Path(mmcv.__file__).parent / 'ops/multi_scale_deform_attn.py'
        namespace = definitions(path, ['multi_scale_deformable_attn_pytorch'],
                                {'torch': torch, 'F': F})
        core = namespace['multi_scale_deformable_attn_pytorch']

        class ReferenceOperator:
            @staticmethod
            def apply(value, shapes, starts, locations, weights, step):
                assert starts.tolist() == [0]
                return core(value, shapes, locations, weights)

        sfs.MultiScaleDeformableAttnFunction = ReferenceOperator

    lfp = importlib.import_module('mmdet.models.necks.lfp')
    aux = importlib.import_module('mmdet.models.necks.aux_fpn')
    aux._verification_backend = backend
    from mmdet.structures import DetDataSample
    return lfp, sfs, aux, DetDataSample


def check_tensor(tensor, shape=None):
    if shape is not None:
        assert tuple(tensor.shape) == tuple(shape), (tensor.shape, shape)
    assert torch.isfinite(tensor).all(), 'Non-finite values'


def check_gradient(tensor):
    assert tensor.grad is not None, 'Missing gradient'
    check_tensor(tensor.grad)
    assert tensor.grad.abs().sum() > 0, 'Zero gradient'
    return tensor.grad.abs().max().item()


def backward(output):
    # Random upstream gradients avoid the nearly constant squared norm after LN.
    (output * torch.randn_like(output)).mean().backward()


def check_lfp(lfp, device, large):
    shapes = [(2, 5, 31, 29), (1, 256, 32, 32), (2, 7, 1, 3)]
    if large:
        shapes.append((1, 256, 256, 256))
    reconstruction = []
    for shape in shapes:
        transform = lfp._WaveletTransform('haar', 'zero').to(device)
        x = torch.randn(shape, device=device)
        low, high = transform.forward_dwt(x)
        rec = transform.forward_idwt(low, high)[..., :shape[-2], :shape[-1]]
        error = (rec - x).abs().max().item()
        assert error < 1e-5, error
        reconstruction.append(dict(shape=shape, max_error=error))

    forward = []
    for use_gauss in (False, True):
        module = lfp.LFP(in_channels=256, with_gauss=use_gauss).to(device)
        x = torch.randn(2, 256, 31, 29, device=device, requires_grad=True)
        y = module(x)
        check_tensor(y, x.shape)
        backward(y)
        grads = {name: check_gradient(param)
                 for name, param in module.named_parameters()}
        check_gradient(x)
        # Even sizes avoid boundary padding effects in this LL-preservation test.
        with torch.no_grad():
            even = torch.randn(1, 256, 32, 32, device=device)
            low, high = module._wavelet.forward_dwt(even)
            low_out, high_out = module._wavelet.forward_dwt(module(even))
            ll_error = (low_out - low).abs().max().item()
            ratio = (high_out.square().mean() / high.square().mean()).item()
            assert ll_error < 1e-5, ll_error
            assert ratio < 1, ratio
        forward.append(dict(with_gauss=use_gauss, gradient_max=grads,
                            ll_error=ll_error, random_input_hf_energy_ratio=ratio))
    result = dict(reconstruction=reconstruction, forward_backward=forward)
    if large:
        module = lfp.LFP(in_channels=256).to(device)
        x = torch.randn(1, 256, 256, 256, device=device, requires_grad=True)
        y = module(x)
        check_tensor(y, x.shape)
        backward(y)
        result['large_forward_backward'] = dict(
            shape=list(y.shape), input_gradient=check_gradient(x),
            parameter_gradients={name: check_gradient(param)
                                 for name, param in module.named_parameters()})
    return result


def check_sfs(sfs, device, large):
    results = []
    shapes = [((2, 256, 32, 32), (2, 256, 16, 16)),
              ((2, 256, 31, 29), (2, 256, 16, 15))]
    if large:
        shapes.append(((1, 256, 256, 256), (1, 256, 128, 128)))
    for qshape, kshape in shapes:
        module = sfs.SpiralAwareCrossDeformAttn2D(256).to(device).eval()
        q = torch.randn(qshape, device=device, requires_grad=True)
        k = torch.randn(kshape, device=device, requires_grad=True)
        out = module(q, k)
        check_tensor(out, qshape)
        assert out.is_contiguous(), out.stride()
        backward(out)
        grads = {name: check_gradient(param)
                 for name, param in module.named_parameters()}
        results.append(dict(query=qshape, key=kshape, contiguous=True,
                            query_gradient=check_gradient(q),
                            key_gradient=check_gradient(k), gradient_max=grads))
    return results


def check_original_sfs(sfs, device):
    original = ROOT.parent / 'NS-FPN'
    if not (original / 'model/diff_cross_attns.py').is_file():
        return dict(status='SKIP', reason='Sibling NS-FPN reference is absent')
    namespace = dict(torch=torch, nn=nn, F=F, math=math, warnings=warnings,
                     Tensor=torch.Tensor, constant_=nn.init.constant_,
                     xavier_uniform_=nn.init.xavier_uniform_,
                     MSDeformAttnFunction=sfs.MultiScaleDeformableAttnFunction)
    definitions(original / 'SFS_MSDeformAttn/ops/modules/ms_deform_attn.py',
                ['_is_power_of_2', 'MSDeformAttn_for_sfs'], namespace)
    definitions(original / 'model/diff_cross_attns.py',
                ['generate_structured_grid', 'SpiralAware_CrossDeformAttn2D'],
                namespace)
    adapted = sfs.SpiralAwareCrossDeformAttn2D(32).to(device).eval()
    reference = namespace['SpiralAware_CrossDeformAttn2D'](32).to(device).eval()
    with torch.no_grad():
        adapted.attention_weights.weight.normal_(0, .1)
        adapted.shared_offsets_residual.normal_(0, .2)
    state = {}
    for name, tensor in adapted.state_dict().items():
        name = name.replace('query_conv.', 'query_Conv.').replace('key_conv.', 'key_Conv.')
        if name.startswith(('attention_weights.', 'value_proj.', 'output_proj.')):
            name = 'attn.' + name
        state[name] = tensor
    reference.load_state_dict(state, strict=True)
    q = torch.randn(2, 32, 13, 11, device=device, requires_grad=True)
    k = torch.randn(2, 32, 7, 6, device=device, requires_grad=True)
    qr = q.detach().clone().requires_grad_()
    kr = k.detach().clone().requires_grad_()
    out, expected = adapted(q, k), reference(qr, kr)
    upstream = torch.randn_like(out)
    (out * upstream).mean().backward()
    (expected * upstream).mean().backward()
    errors = dict(output=(out - expected).abs().max().item(),
                  query_grad=(q.grad - qr.grad).abs().max().item(),
                  key_grad=(k.grad - kr.grad).abs().max().item(),
                  offset_grad=(adapted.shared_offsets_residual.grad -
                               reference.shared_offsets_residual.grad).abs().max().item())
    assert max(errors.values()) < 1e-5, errors
    return errors


def samples(data_sample, device='cpu', include_gt=False):
    result = []
    for view, band in [('Air', 'LWIR'), ('Space', 'NIR')]:
        sample = data_sample(metainfo=dict(
            view=view, band_type=band, width=128, height=128,
            img_shape=(128, 128), ori_shape=(128, 128),
            scale_factor=(1., 1.), batch_input_shape=(128, 128)))
        if include_gt:
            from mmengine.structures import InstanceData
            sample.gt_instances = InstanceData(
                bboxes=torch.tensor([[36., 36., 44., 44.]], device=device),
                labels=torch.zeros(1, dtype=torch.long, device=device))
        result.append(sample)
    return result


def check_necks(aux, data_sample, device):
    results = []
    for index, filename in enumerate(CONFIGS):
        cfg = Config.fromfile(str(ROOT / 'configs/auxdet' / filename))
        neck_cfg = copy.deepcopy(dict(cfg.model.neck))
        neck_cfg.pop('type')
        module = aux.AuxFPN(**neck_cfg).to(device).train()
        feats = [torch.randn(2, channels, size, size, device=device, requires_grad=True)
                 for channels, size in zip(neck_cfg['in_channels'], (32, 16, 8, 4))]
        events, handles = [], []
        for name, child in module.named_modules():
            if name in ('lfp_modules.0', 'sfs_modules.0', 'dynamic_atts.0', 'edge_convs.0'):
                handles.append(child.register_forward_hook(
                    lambda mod, inputs, output, name=name: events.append(name)))
        out = module(feats, samples(data_sample))
        for handle in handles:
            handle.remove()
        assert len(out) == 2  # AuxDet's heads deliberately consume P2 and P3.
        for tensor, size in zip(out, (32, 16)):
            check_tensor(tensor, (2, 256, size, size))
            assert tensor.is_contiguous(), tensor.stride()
        sum((tensor * torch.randn_like(tensor)).mean() for tensor in out).backward()
        for tensor in feats:
            check_gradient(tensor)
        new_grads = {name: check_gradient(param)
                     for name, param in module.named_parameters()
                     if name.startswith(('lfp_modules.', 'sfs_modules.'))}
        expected = ['dynamic_atts.0', 'edge_convs.0']
        if index == 1:
            expected.insert(0, 'lfp_modules.0')
        elif index in (2, 4):
            expected.append('lfp_modules.0')
        if index in (3, 4):
            expected.append('sfs_modules.0')
        assert events == expected, (events, expected)
        module.eval()
        with torch.no_grad():
            eval_out = module([feat.detach()[:1] for feat in feats], samples(data_sample)[:1])
            for tensor, size in zip(eval_out, (32, 16)):
                check_tensor(tensor, (1, 256, size, size))
        results.append(dict(config=filename, train_call_order=events,
                            output_shapes=[list(t.shape) for t in out],
                            new_parameter_gradient_max=new_grads, eval_batch_one='PASS'))
    return results


def check_full_model(data_sample, device):
    from mmdet.registry import MODELS
    cfg = Config.fromfile(str(ROOT / 'configs/auxdet' / CONFIGS[4]))
    cfg.model.backbone.init_cfg = None
    model = MODELS.build(cfg.model).to(device)
    model.init_weights()
    model.train()
    inputs = torch.randn(2, 3, 128, 128, device=device)
    data = samples(data_sample, device, include_gt=True)
    losses = model.loss(inputs, data)
    total = sum(v.sum() if torch.is_tensor(v) else sum(t.sum() for t in v)
                for key, v in losses.items() if 'loss' in key)
    check_tensor(total)
    total.backward()
    grads = {name: check_gradient(param)
             for name, param in model.neck.named_parameters()
             if name.startswith(('lfp_modules.', 'sfs_modules.'))}
    model.eval()
    with torch.no_grad():
        predicted = model.predict(inputs, data)
    assert len(predicted) == 2
    for sample in predicted:
        check_tensor(sample.pred_instances.bboxes)
        check_tensor(sample.pred_instances.scores)
    return dict(loss=total.item(), gradient_max=grads,
                predictions=[len(sample.pred_instances) for sample in predicted])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('reference', 'cuda'), required=True)
    parser.add_argument('--large', action='store_true', help='Also check 256x256 P2')
    parser.add_argument('--full-model', action='store_true', help='CUDA B4 loss/predict smoke check')
    parser.add_argument('--json', type=Path, help='Save machine-readable evidence')
    args = parser.parse_args()
    if args.full_model and args.backend != 'cuda':
        parser.error('--full-model requires --backend cuda')
    torch.manual_seed(20260927)
    torch.set_num_threads(4)
    device = 'cuda' if args.backend == 'cuda' else 'cpu'
    source_paths = [ROOT / 'mmdet/models/necks' / name
                    for name in ('aux_fpn.py', 'lfp.py', 'sfs.py')]
    source_paths += [ROOT / 'configs/auxdet' / name for name in CONFIGS]
    report = dict(
        backend=args.backend, device=device, torch=torch.__version__,
        mmcv=mmcv.__version__, seed=20260927, large=args.large,
        full_model=args.full_model, python=sys.version,
        dependencies={name: importlib.metadata.version(name) for name in
                      ('mmengine', 'pytorch-wavelets', 'PyWavelets', 'numpy', 'scipy', 'setuptools')},
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in source_paths}, results={})
    lfp, sfs, aux, data_sample = load_modules(args.backend)
    checks = dict(lfp=lambda: check_lfp(lfp, device, args.large),
                  sfs=lambda: check_sfs(sfs, device, args.large),
                  original_sfs_equivalence=lambda: check_original_sfs(sfs, device),
                  b0_b4_necks=lambda: check_necks(aux, data_sample, device))
    if args.full_model:
        checks['full_model'] = lambda: check_full_model(data_sample, device)
    failed = False
    for name, check in checks.items():
        try:
            details = check()
            report['results'][name] = dict(status='PASS', details=details)
            print(f'PASS {name}', flush=True)
        except Exception as exc:
            failed = True
            report['results'][name] = dict(status='FAIL', error=f'{type(exc).__name__}: {exc}')
            print(f'FAIL {name}: {type(exc).__name__}: {exc}', flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + '\n')
        print(f'Report: {args.json.resolve()}')
    else:
        print(json.dumps(report, indent=2))
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
