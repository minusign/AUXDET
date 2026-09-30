"""DLC math and real AuxFPN source checks, without compiled detector ops."""
import ast
import importlib.util
from pathlib import Path

import pytest
import torch
from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[2]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

mlc = load('meta_mlc', ROOT / 'mmdet/models/necks/mlc.py')
verify = load('meta_mlc_verify', ROOT / 'tools/analysis_tools/verify_nsfpn.py')
_, _, aux, DataSample = verify.load_modules('reference')
torch.set_num_threads(2)


def oracle(x, distance):
    """Independent coordinate indexing, with explicit coordinate clamping."""
    result = torch.empty_like(x)
    h, w = x.shape[-2:]
    for y in range(h):
        for xx in range(w):
            products = []
            for dy, dx in ((distance, distance), (distance, 0),
                           (distance, -distance), (0, distance)):
                a = x[..., max(0, min(h - 1, y + dy)), max(0, min(w - 1, xx + dx))]
                b = x[..., max(0, min(h - 1, y - dy)), max(0, min(w - 1, xx - dx))]
                products.append((x[..., y, xx] - a) * (x[..., y, xx] - b))
            result[..., y, xx] = torch.stack(products).min(dim=0).values
    return result


@pytest.mark.parametrize('distance', [1, 2, 3])
def test_dlc_independent_oracle_and_scale_max(distance):
    torch.manual_seed(3)
    x = torch.randn(2, 4, 5, 7)
    assert torch.equal(mlc.directional_dlc(x, distance), oracle(x, distance))
    module = mlc.LocalContrastMLC(4)
    fused, weights = module.contrast(x)
    assert weights is None
    assert torch.equal(fused, torch.stack([oracle(x, d) for d in (1, 2, 3)]).amax(0))


def test_official_source_algebra_interior():
    path = ROOT.parent / 'ALCNet-reference/model/contrast.py'
    if not path.exists():
        pytest.skip('Read-only sibling ALCNet reference unavailable')
    class ND:
        concat = staticmethod(lambda *xs, dim: torch.cat(xs, dim=dim))
        minimum = staticmethod(torch.minimum)
    namespace = {'nd': ND}
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ('circ_shift', 'cal_pcm')]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    x = torch.randn(2, 4, 11, 13)
    for distance in (1, 2, 3):
        # The reference's actual Python functions run with Torch tensor ops;
        # no MXNet installation. Interior matches; boundary difference is intentional.
        official = namespace['cal_pcm'](x, distance)
        port = mlc.directional_dlc(x, distance)
        assert torch.equal(official[..., distance:-distance, distance:-distance],
                           port[..., distance:-distance, distance:-distance])


def test_boundary_no_wrap_and_no_batch_mixing():
    x = torch.zeros(2, 1, 4, 6)
    x[0, :, :, -1] = 20
    x[1] = 100
    shifted = mlc._shift_replicate(x, 0, -1)
    assert torch.equal(shifted[0, :, :, 0], torch.zeros(1, 4))
    assert shifted[1].eq(100).all()
    assert torch.equal(mlc.directional_dlc(x, 1)[0], mlc.directional_dlc(x[:1], 1)[0])
    assert mlc.directional_dlc(x, 1)[1].eq(0).all()


@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('mode', ['original', 'visual', 'visual_metadata'])
def test_shapes_stats_and_finite_gradients(batch, mode):
    module = mlc.LocalContrastMLC(8, mode=mode, metadata_dim=6,
                                  collect_stats=True, store_weight_maps=True)
    x = torch.randn(batch, 8, 7, 5, requires_grad=True)
    z = torch.randn(batch, 6, requires_grad=True)
    y = module(x, z)
    assert y.shape == x.shape and y.dtype == x.dtype and y.device == x.device
    assert torch.isfinite(y).all()
    stats = module.get_scale_stats()
    assert len(stats) == batch
    maps = module.get_weight_maps()
    if mode != 'original':
        assert maps.shape == (batch, 3, 7, 5) and not maps.requires_grad
        assert torch.allclose(maps.sum(1), torch.ones(batch, 7, 5), atol=1e-6)
        assert all(sum(row['mean']) == pytest.approx(1, abs=1e-6) for row in stats)
    else:
        assert maps is None and all(row['mean'] is None for row in stats)
    (y * torch.randn_like(y)).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for name, param in module.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
    assert module.residual_projection.weight.grad.abs().sum() > 0
    if mode == 'visual_metadata':
        assert module.metadata_condition.weight.grad.abs().sum() > 0
        assert z.grad is not None and z.grad.eq(0).all()  # intentional zero init
    else:
        assert z.grad is None


def test_condition_control_and_active_gradient():
    module = mlc.LocalContrastMLC(4, mode='visual_metadata', metadata_dim=3,
                                  hidden_dim=4)
    visual = mlc.LocalContrastMLC(4, mode='visual', metadata_dim=3, hidden_dim=4)
    x = torch.randn(2, 4, 5, 7, requires_grad=True)
    z = torch.zeros(2, 3, requires_grad=True)
    changed = torch.zeros_like(z)
    changed[1, 0] = 1
    assert torch.equal(module.scale_weights(x, z), module.scale_weights(x, changed))
    with torch.no_grad():
        module.metadata_condition.weight[-3, 0] = 1
        module.metadata_condition.weight[-2, 0] = -1
        module.metadata_condition.weight[0, 0] = .25  # exercises FiLM path too
    a, b = module.scale_weights(x, z), module.scale_weights(x, changed)
    assert torch.equal(a[0], b[0]) and not torch.equal(a[1], b[1])
    assert torch.equal(visual.scale_weights(x, z), visual.scale_weights(x, changed))
    y = module(x, z)
    (y * torch.randn_like(y)).mean().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0 and torch.isfinite(z.grad).all()
    assert module.visual_descriptor[0].weight.grad.abs().sum() > 0


def config(mode):
    return Config.fromfile(str(ROOT / f'configs/auxdet/auxdet_r50_mlc_p2_{mode}_1x_voc.py'))


def test_configs_actual_initialization_and_neck_forward():
    base = Config.fromfile(str(ROOT / 'configs/auxdet/auxdet_r50_fpn_1x_voc.py'))
    counts = []
    for mode in ('original', 'visual', 'visual_metadata'):
        cfg = config(mode)
        for key in ('train_dataloader', 'val_dataloader', 'test_dataloader',
                    'optim_wrapper', 'param_scheduler', 'train_cfg'):
            assert cfg[key] == base[key]
        options = dict(cfg.model.neck)
        options.pop('type')
        module = aux.AuxFPN(**options)
        module.init_weights()  # actual MMEngine/AuxFPN default init_cfg
        module.eval()
        assert not module.lfp_modules and not module.lfp_gates and not module.sfs_modules
        assert list(module.mlc_modules) == ['0']
        branch = module.mlc_modules['0']
        counts.append(sum(p.numel() for p in branch.parameters()))
        assert torch.nn.functional.softplus(branch.lambda_raw).item() == pytest.approx(.1)
        if mode == 'visual_metadata':
            assert branch.metadata_condition.weight.eq(0).all()
            assert branch.metadata_condition.bias.eq(0).all()
        branch.collect_stats = True
        for batch in (1, 2):
            features = [torch.randn(batch, c, h, w) for c, h, w in
                        [(256, 32, 24), (512, 16, 12), (1024, 8, 6), (2048, 4, 3)]]
            samples = verify.samples(DataSample)[:batch]
            events = []
            handles = [child.register_forward_hook(lambda m, i, o, name=name: events.append(name))
                       for name, child in module.named_modules()
                       if name in ('dynamic_atts.0', 'edge_convs.0', 'mlc_modules.0', 'fpn_convs.0')]
            with torch.no_grad():
                outputs = module(features, samples)
            for handle in handles:
                handle.remove()
            assert [tuple(y.shape) for y in outputs] == [(batch, 256, 32, 24), (batch, 256, 16, 12)]
            assert all(torch.isfinite(y).all() for y in outputs)
            assert events == ['dynamic_atts.0', 'edge_convs.0', 'mlc_modules.0', 'fpn_convs.0']
            assert len(module.get_mlc_scale_stats()['0']) == batch
    assert counts == [66049, 82692, 95399]


def test_metadata_optional_interface_and_missing_policy():
    processor = aux.MetaFeatureProcessorWithSem(128).eval()
    samples = verify.samples(DataSample)
    sem = torch.randn(2, 256)
    default = processor(samples, sem.dtype, sem.device, sem)
    fused, view, band, size = processor(samples, sem.dtype, sem.device, sem, True)
    assert torch.equal(default, fused) and fused.shape == (2, 128)
    pure = torch.cat([view, band, size], 1)
    assert pure.shape == (2, 96)
    other = processor(samples, sem.dtype, sem.device, sem + 100, True)
    assert torch.equal(pure, torch.cat(other[1:], 1))
    missing = DataSample(metainfo={'ori_shape': (128, 192)})
    assert torch.isfinite(processor([missing], sem.dtype, sem.device, sem[:1], True)[0]).all()
    with pytest.raises(ValueError, match='width/height'):
        processor([DataSample()], sem.dtype, sem.device, sem[:1])


def test_default_configs_and_invalid_combinations():
    for filename in verify.CONFIGS:
        options = dict(Config.fromfile(str(ROOT / 'configs/auxdet' / filename)).model.neck)
        options.pop('type')
        module = aux.AuxFPN(**options)
        assert not module.mlc_modules
        assert not any(key.startswith('mlc_modules.') for key in module.state_dict())
    options = dict(config('visual_metadata').model.neck)
    options.pop('type')
    with pytest.raises(ValueError, match='disable LFP'):
        aux.AuxFPN(**dict(options, lfp_cfg={}, lfp_levels=(0,)))
    with pytest.raises(ValueError, match='P2'):
        aux.AuxFPN(**dict(options, start_level=1))
    with pytest.raises(ValueError, match='metadata_dim'):
        aux.AuxFPN(**dict(options, mlc_cfg=dict(mode='visual_metadata', metadata_dim=128)))
    with pytest.raises(ValueError, match='pure metadata'):
        mlc.LocalContrastMLC(8, mode='visual_metadata')(torch.randn(2, 8, 4, 6))


def test_half_precision_products_do_not_overflow_and_stats_opt_in():
    x = torch.randn(2, 4, 7, 5).half() * 500
    response = mlc.directional_dlc(x, 2)
    assert response.dtype == torch.float32 and torch.isfinite(response).all()
    module = mlc.LocalContrastMLC(4, mode='visual_metadata', metadata_dim=3)
    module(torch.randn(2, 4, 7, 5), torch.randn(2, 3))
    assert module.get_scale_stats() is None and module.get_weight_maps() is None


def test_contrast_path_gradient_and_cpu_autocast():
    module = mlc.LocalContrastMLC(8, mode='visual_metadata', metadata_dim=6)
    x = torch.randn(2, 8, 7, 5, requires_grad=True)
    fused, _ = module.contrast(x, torch.randn(2, 6))
    (fused * torch.randn_like(fused)).mean().backward()
    assert x.grad.abs().sum() > 0 and torch.isfinite(x.grad).all()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = module(x.detach(), torch.randn(2, 6))
    assert output.dtype == x.dtype and torch.isfinite(output).all()


def test_checkpoint_load_preserves_trained_condition_after_init():
    options = dict(config('visual_metadata').model.neck)
    options.pop('type')
    neck = aux.AuxFPN(**options)
    neck.init_weights()
    state = neck.state_dict()
    state['mlc_modules.0.metadata_condition.weight'] = torch.ones_like(
        state['mlc_modules.0.metadata_condition.weight']) * .125
    state['mlc_modules.0.lambda_raw'] = torch.tensor(-1.0)
    neck.load_state_dict(state, strict=True)
    assert neck.mlc_modules['0'].metadata_condition.weight.eq(.125).all()
    assert neck.mlc_modules['0'].lambda_raw.item() == -1.0
    base_options = dict(Config.fromfile(str(ROOT / 'configs/auxdet/auxdet_r50_fpn_1x_voc.py')).model.neck)
    base_options.pop('type')
    incompatible = neck.load_state_dict(aux.AuxFPN(**base_options).state_dict(), strict=False)
    assert incompatible.missing_keys and all(k.startswith('mlc_modules.0.') for k in incompatible.missing_keys)
    assert not incompatible.unexpected_keys


def test_export_hook_schema_and_map(tmp_path):
    from types import SimpleNamespace
    exporter = load('meta_mlc_export', ROOT / 'tools/analysis_tools/export_mlc_scales.py')
    module = mlc.LocalContrastMLC(4, mode='visual', collect_stats=True, store_weight_maps=True)
    module(torch.randn(2, 4, 7, 5))
    rows = module.get_scale_stats()
    for i, row in enumerate(rows):
        row.update(img_id=i, view='Air', band_type='LWIR')
    neck = SimpleNamespace(get_mlc_scale_stats=lambda: {'0': rows}, mlc_modules={'0': module})
    runner = SimpleNamespace(model=SimpleNamespace(neck=neck), cfg=Config(dict(check='unit_test')))
    hook = exporter.MLCScaleExportHook(tmp_path / 'export', map_ids=['1'])
    hook.before_test(runner)
    hook.after_test_iter(runner, 0)
    hook.after_test(runner)
    import json
    records = [json.loads(line) for line in (tmp_path / 'export/scale_stats.jsonl').read_text().splitlines()]
    assert len(records) == 2 and records[1]['img_id'] == 1 and records[0]['level'] == 0
    assert 'weight_map' not in records[0]
    saved = torch.load(tmp_path / 'export' / records[1]['weight_map'])
    assert saved['weights'].shape == (3, 7, 5) and not saved['weights'].requires_grad
    with pytest.raises(FileExistsError):
        exporter.MLCScaleExportHook(tmp_path / 'export').before_test(runner)


def test_real_neck_training_backward_reuses_metadata_encoding():
    options = dict(config('visual_metadata').model.neck)
    options.pop('type')
    neck = aux.AuxFPN(**options).train()
    features = [torch.randn(2, c, h, w, requires_grad=True) for c, h, w in
                [(256, 32, 24), (512, 16, 12), (1024, 8, 6), (2048, 4, 3)]]
    calls = []
    handle = neck.meta_process_with_sem.view_mlp.register_forward_hook(lambda m, i, o: calls.append(1))
    outputs = neck(features, verify.samples(DataSample))
    handle.remove()
    assert len(calls) == 2  # Existing M2DM calls for P2/P3; no extra encoding.
    sum((y * torch.randn_like(y)).mean() for y in outputs).backward()
    for tensor in features:
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    branch = neck.mlc_modules['0']
    for parameter in (branch.metadata_condition.weight, branch.residual_projection.weight,
                      branch.visual_descriptor[0].weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
        assert torch.isfinite(parameter.grad).all()
