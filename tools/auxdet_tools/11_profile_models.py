#!/usr/bin/env python3
"""Profile complete detectors on one CUDA device with a recorded fixed protocol."""
from __future__ import annotations

import argparse
import copy
import gc
from pathlib import Path
import sys
import time
import numpy as np

from eval_common import PROJECT, digest, file_info, read_manifest, write_csv, write_json


def latency_statistics(seconds, batch_size):
    values = np.asarray(seconds, dtype=np.float64) * 1000
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Timing samples must be finite, positive and nonempty")
    return dict(latency_mean_ms=float(values.mean()), latency_median_ms=float(np.median(values)),
                latency_p95_ms=float(np.quantile(values, .95)),
                latency_std_ms=float(values.std(ddof=1)) if len(values) > 1 else None,
                throughput_images_per_second=float(batch_size * 1000 / values.mean()))


def profile_one(experiment, args, torch):
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmdet.registry import MODELS
    from mmdet.structures import DetDataSample
    from mmdet.utils import register_all_modules
    register_all_modules()
    cfg = Config.fromfile(experiment["config"])
    cfg.model.backbone.init_cfg = None
    model = MODELS.build(cfg.model)
    # Fail if a B0 checkpoint is accidentally paired with a B4 model.
    load_checkpoint(model, experiment["checkpoint"], map_location="cpu", strict=True)
    model = model.to(args.device).eval()
    torch.manual_seed(args.input_seed)
    height, width = args.shape
    batch = args.batch_size
    raw_inputs = [torch.rand(3, height, width) * 255 for _ in range(batch)]
    data_samples = [DetDataSample(metainfo=dict(
        img_id=f"profile_{i}", ori_shape=(height, width), img_shape=(height, width),
        scale_factor=(1., 1.), width=width, height=height,
        view=args.view, band_type=args.band_type)) for i in range(batch)]
    raw_batch = dict(inputs=raw_inputs, data_samples=data_samples)
    processed = model.data_preprocessor(copy.deepcopy(raw_batch), training=False)
    def execute():
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.precision == "fp16"):
            if args.scope == "test-step":
                return model.test_step(copy.deepcopy(raw_batch))
            return model.predict(processed["inputs"], copy.deepcopy(processed["data_samples"]), rescale=True)
    for _ in range(args.warmup):
        execute()
    torch.cuda.synchronize(args.device)
    resident = torch.cuda.memory_allocated(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    durations = []
    for _ in range(args.iterations):
        torch.cuda.synchronize(args.device)
        started = time.perf_counter()
        execute()
        torch.cuda.synchronize(args.device)
        durations.append(time.perf_counter()-started)
    result = dict(experiment=experiment["experiment"], model=experiment["model"], status="pass",
                  params=sum(p.numel() for p in model.parameters()),
                  trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  checkpoint_bytes=Path(experiment["checkpoint"]).stat().st_size,
                  **latency_statistics(durations, batch),
                  peak_allocated_mb=torch.cuda.max_memory_allocated(args.device)/1024**2,
                  peak_reserved_mb=torch.cuda.max_memory_reserved(args.device)/1024**2,
                  resident_allocated_mb=resident/1024**2,
                  incremental_peak_mb=(torch.cuda.max_memory_allocated(args.device)-resident)/1024**2,
                  flops=None, flops_status="not_requested")
    details = dict(config=file_info(experiment["config"]), checkpoint=file_info(experiment["checkpoint"]),
                   actual_input_shape=list(processed["inputs"].shape),
                   duration_seconds=durations, unsupported_ops={}, uncalled_modules=[])
    if not args.skip_flops:
        try:
            from mmengine.analysis import FlopAnalyzer
            class TensorForward(torch.nn.Module):
                def __init__(self, detector):
                    super().__init__()
                    self.detector = detector

                def forward(self, tensor):
                    return self.detector(tensor, copy.deepcopy(processed["data_samples"]), mode="tensor")
            with torch.no_grad():
                analyzer = FlopAnalyzer(TensorForward(model), processed["inputs"])
                result["flops"] = float(analyzer.total())
                unsupported = dict(analyzer.unsupported_ops())
                details["unsupported_ops"] = {str(k): int(v) for k, v in unsupported.items()}
                details["uncalled_modules"] = sorted(analyzer.uncalled_modules())
                result["flops_status"] = "partial" if unsupported else "counted"
                details["flops_scope"] = "FP32 mode=tensor trace; FMA=1; excludes uncounted ops; differs from latency scope"
        except Exception as exc:
            result["flops_status"] = "error"
            details["flops_error"] = f"{type(exc).__name__}: {exc}"
    return result, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shape", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--scope", choices=("predict", "test-step"), default="predict")
    parser.add_argument("--view", default="Space")
    parser.add_argument("--band-type", default="NIR")
    parser.add_argument("--input-seed", type=int, default=0)
    parser.add_argument("--skip-flops", action="store_true")
    args = parser.parse_args()
    if min(*args.shape, args.batch_size, args.iterations) <= 0 or args.warmup < 0:
        parser.error("Dimensions/batch/iterations must be positive and warmup nonnegative")
    experiments = read_manifest(args.manifest, args.experiments)
    sys.path.insert(0, str(PROJECT))
    import torch
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("This profiling protocol requires an available CUDA device")
    torch.cuda.set_device(args.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = dict(device=torch.cuda.get_device_name(args.device), device_index=args.device,
                    torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    shape=args.shape, batch_size=args.batch_size, precision=args.precision,
                    warmup=args.warmup, iterations=args.iterations, scope=args.scope,
                    input="fixed CPU synthetic uniform[0,255]; no file IO", input_seed=args.input_seed,
                    view=args.view, band_type=args.band_type, tf32=False,
                    includes_NMS=True, includes_preprocessing=args.scope == "test-step",
                    includes_H2D=args.scope == "test-step",
                    timing="synchronized wall clock per batch; includes Python call/sample-copy overhead")
    rows, details = [], {}
    for experiment in experiments:
        gc.collect()
        torch.cuda.empty_cache()
        try:
            row, detail = profile_one(experiment, args, torch)
        except Exception as exc:
            row = dict(experiment=experiment["experiment"], model=experiment["model"],
                       status="fail", error=f"{type(exc).__name__}: {exc}")
            detail = {"error": row["error"]}
        row["protocol_sha256"] = digest(protocol)
        rows.append(row)
        details[experiment["experiment"]] = detail
        write_csv(args.output_dir / "complexity_metrics.csv", rows)
        write_json(args.output_dir / "profiling_metadata.json",
                   dict(protocol=protocol, protocol_sha256=digest(protocol), experiments=details))
        print(f"{experiment['experiment']}: {row['status']}", flush=True)
    return int(any(row["status"] != "pass" or row.get("flops_status") == "error" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
