"""Small Meta-MLC neck checks; never trains or downloads checkpoints/data.

source: load actual computational source without detector package aggregation;
SFS's reference operator is imported by the existing helper but never executed.
native: normal MMDetection imports (requires compiled MMCV). Optional full
model construction reports parameters; it does not run detector loss/predict.
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import torch
from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('source', 'native'), default='source')
    parser.add_argument('--full-model', action='store_true')
    parser.add_argument('--baseline-ref', help='Optional local Git ref for default B0 equivalence')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    if args.full_model and args.backend != 'native':
        parser.error('--full-model requires --backend native')
    if args.json and args.json.exists():
        parser.error('--json must name a new file')
    torch.manual_seed(20260930)
    torch.set_num_threads(2)
    if args.backend == 'source':
        spec = importlib.util.spec_from_file_location('nsfpn_verify', ROOT / 'tools/analysis_tools/verify_nsfpn.py')
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        _, _, aux, DataSample = helper.load_modules('reference')
    else:
        from mmdet.utils import register_all_modules
        register_all_modules()
        from mmdet.models.necks import aux_fpn as aux
        from mmdet.structures import DetDataSample as DataSample
    samples = [DataSample(metainfo=dict(img_id=i, view=view, band_type=band,
                                       width=128, height=96))
               for i, (view, band) in enumerate((('Air', 'LWIR'), ('Space', 'NIR')))]
    features = [torch.randn(2, c, h, w) for c, h, w in
                [(256, 32, 24), (512, 16, 12), (1024, 8, 6), (2048, 4, 3)]]
    report = dict(backend=args.backend, torch=torch.__version__, device='cpu', modes={})
    base_cfg = Config.fromfile(str(ROOT / 'configs/auxdet/auxdet_r50_fpn_1x_voc.py'))
    base_options = dict(base_cfg.model.neck)
    base_options.pop('type')
    base = aux.AuxFPN(**base_options).eval()
    report['baseline_neck_parameters'] = sum(p.numel() for p in base.parameters())
    if args.baseline_ref:
        source = subprocess.check_output(
            ['git', 'show', args.baseline_ref + ':mmdet/models/necks/aux_fpn.py'], cwd=ROOT, text=True)
        namespace = dict(aux.__dict__)
        nodes = []
        for node in ast.parse(source).body:
            if isinstance(node, ast.ClassDef) and node.name in ('AuxFPN', 'MetaFeatureProcessorWithSem'):
                node.decorator_list = []
                nodes.append(node)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<local baseline source>', 'exec'), namespace)
        old = namespace['AuxFPN'](**base_options).eval()
        base.load_state_dict(old.state_dict(), strict=True)
        with torch.no_grad():
            errors = [(a - b).abs().max().item() for a, b in
                      zip(old(features, samples), base(features, samples))]
        assert errors == [0.0, 0.0], errors
        report['default_b0_comparison'] = dict(ref=args.baseline_ref, max_abs_errors=errors)
    for mode in ('original', 'visual', 'visual_metadata'):
        path = ROOT / f'configs/auxdet/auxdet_r50_mlc_p2_{mode}_1x_voc.py'
        cfg = Config.fromfile(str(path))
        options = dict(cfg.model.neck)
        options.pop('type')
        module = aux.AuxFPN(**options)
        module.init_weights()
        module.eval()
        branch = module.mlc_modules['0']
        branch.collect_stats = True
        assert not module.lfp_modules and not module.lfp_gates and not module.sfs_modules
        assert torch.allclose(torch.nn.functional.softplus(branch.lambda_raw), torch.tensor(.1))
        if mode == 'visual_metadata':
            assert branch.metadata_condition.weight.eq(0).all()
        with torch.no_grad():
            outputs = module(features, samples)
            stress = features[0] * 1000
            stress_output = branch(stress, torch.randn(2, 96) if mode == 'visual_metadata' else None)
        assert [list(t.shape) for t in outputs] == [[2, 256, 32, 24], [2, 256, 16, 12]]
        assert all(torch.isfinite(t).all() for t in (*outputs, stress_output))
        row = dict(config=str(path.relative_to(ROOT)),
                   neck_parameters=sum(p.numel() for p in module.parameters()),
                   added_parameters=sum(p.numel() for p in branch.parameters()),
                   output_shapes=[list(t.shape) for t in outputs],
                   scale_stats=module.get_mlc_scale_stats(),
                   stress_residual_rms=(stress_output - stress).square().mean().sqrt().item(),
                   other_optional_modules_absent=True)
        if args.full_model:
            from mmdet.registry import MODELS
            # Construction only; pretrained backbone init is not invoked.
            detector = MODELS.build(cfg.model)
            row['full_detector_parameters'] = sum(p.numel() for p in detector.parameters())
        report['modes'][mode] = row
        print(f'PASS {mode}: +{row["added_parameters"]} parameters')
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
    print('PASS default compatibility' if args.baseline_ref else 'No historical baseline comparison requested')


if __name__ == '__main__':
    main()
