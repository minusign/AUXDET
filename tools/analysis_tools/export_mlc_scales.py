"""Run the normal test loop and export detached per-image MLC scale statistics.

Single-process export only. Outputs are created exclusively to protect previous
experiments. Optional .pt maps are feature-resolution [K,H,W] tensors.
"""
import argparse
import json
from pathlib import Path
import sys

from mmengine.config import Config, DictAction
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class MLCScaleExportHook(Hook):
    def __init__(self, output_dir, map_ids=()):
        self.output_dir = Path(output_dir)
        self.map_ids = {str(value) for value in map_ids}
        self._handle = None
        self._index = 0

    def before_test(self, runner):
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._handle = (self.output_dir / 'scale_stats.jsonl').open('x', encoding='utf-8')
        cfg_path = self.output_dir / 'resolved_config.py'
        runner.cfg.dump(str(cfg_path))

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        neck = model.neck
        stats = neck.get_mlc_scale_stats()
        if not stats:
            raise RuntimeError('No MLC statistics: check model.neck.mlc_cfg and collect_stats')
        for level, rows in stats.items():
            maps = neck.mlc_modules[level].get_weight_maps()
            for index, row in enumerate(rows):
                record = dict(row, level=int(level), batch_idx=batch_idx)
                image_id = str(row['img_id'])
                if image_id in self.map_ids:
                    if maps is None:
                        raise RuntimeError('Original max MLC has no scalar softmax weight maps')
                    import torch
                    # Numeric names are safe even when image IDs are paths.
                    name = f'scale_map_{self._index:06d}_p{int(level) + 2}.pt'
                    torch.save(dict(img_id=row['img_id'], dilations=row['dilations'],
                                    weights=maps[index]), self.output_dir / name)
                    record['weight_map'] = name
                    self._index += 1
                self._handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        self._handle.flush()

    def after_test(self, runner):
        if self._handle is not None:
            self._handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--map-ids', nargs='*', default=[])
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()
    if Path(args.output_dir).exists():
        parser.error('--output-dir must be a new directory')
    from mmengine.runner import Runner
    from mmdet.registry import HOOKS
    from mmdet.utils import register_all_modules
    register_all_modules()
    HOOKS.register_module(module=MLCScaleExportHook)
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if not cfg.model.neck.get('mlc_cfg'):
        parser.error('Configuration does not enable MLC')
    cfg.model.neck.mlc_cfg.collect_stats = True
    cfg.model.neck.mlc_cfg.store_weight_maps = bool(args.map_ids)
    cfg.load_from = args.checkpoint
    cfg.launcher = 'none'
    cfg.work_dir = str(Path(args.output_dir).parent / (Path(args.output_dir).name + '_runner'))
    # Keep normal evaluation hooks and data processing intact.
    cfg.custom_hooks = list(cfg.get('custom_hooks', [])) + [
        dict(type='MLCScaleExportHook', output_dir=args.output_dir, map_ids=args.map_ids)]
    Runner.from_cfg(cfg).test()


if __name__ == '__main__':
    main()
