"""Server-side integration check for the six experiment configurations.

The script never reports a skipped model build as success.  It separates
config parsing from registry/model construction; tensor and data checks are
left to the training environment where the required dependencies and dataset
are available.
"""

import argparse
import os


CONFIGS = (
    'auxdet_base.py', 'auxdet_pconv.py', 'auxdet_hsfpn.py',
    'auxdet_nsfpn.py', 'auxdet_pconv_hsfpn.py', 'auxdet_pconv_nsfpn.py')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-dir', default='configs/graduation')
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args()

    try:
        from mmengine.config import Config
    except ImportError as exc:
        raise SystemExit(f'CONFIG_PARSE_BLOCKED: missing MMEngine ({exc})')

    configs = []
    for name in CONFIGS:
        path = os.path.join(args.config_dir, name)
        if not os.path.isfile(path):
            raise SystemExit(f'CONFIG_PARSE_FAILED: missing {path}')
        try:
            cfg = Config.fromfile(path)
        except Exception as exc:
            raise SystemExit(f'CONFIG_PARSE_FAILED: {path}: {exc}')
        configs.append((path, cfg))
        print(f'CONFIG_PARSE_OK {path} model={cfg.model.type if "type" in cfg.model else "nested"}')

    if not args.build:
        print('MODEL_BUILD_NOT_REQUESTED: rerun with --build in the server environment')
        return

    try:
        from mmdet.registry import MODELS
        from mmdet.utils import register_all_modules
        register_all_modules()
    except Exception as exc:
        raise SystemExit(f'MODEL_BUILD_BLOCKED: registry imports failed ({exc})')
    for path, cfg in configs:
        try:
            model = MODELS.build(cfg.model)
        except Exception as exc:
            raise SystemExit(f'MODEL_BUILD_FAILED: {path}: {exc}')
        print(f'MODEL_BUILD_OK {path} detector={model.__class__.__name__}')
    print('MODEL_BUILD_COMPLETE: tensor, loss, backward, checkpoint and real-data checks remain separate')


if __name__ == '__main__':
    main()

