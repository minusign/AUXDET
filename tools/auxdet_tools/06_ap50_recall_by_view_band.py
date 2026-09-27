#!/usr/bin/env python3
"""Official VOC evaluation from tools/test.py --out, or the official dataloader."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_common import (PROJECT, attach_predictions, data_arguments, evaluate,
                         file_info, groups, load_dataset, load_prediction_file,
                         write_csv, write_json)


def parity_report(metrics, path=None, step=None, decimals=3):
    report = {"internal_official_matcher": "pass", "status": "not_requested",
              "checks": {}, "note": "Recall is final detection recall, not recall@N proposals."}
    if path is None:
        return report
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        payload = json.loads(text)
        candidates = payload if isinstance(payload, list) else [payload]
    except json.JSONDecodeError:
        candidates = [json.loads(line) for line in text.splitlines() if line.strip()]
    def flattened(row):
        for key in ("metrics", "metric"):
            if isinstance(row.get(key), dict):
                return {**row, **row[key]}
        return row
    candidates = [flattened(row) for row in candidates if isinstance(row, dict)]
    candidates = [row for row in candidates
                  if any(key.split("/")[-1] in ("AP50", "mAP") for key in row)
                  and (step is None or row.get("step", row.get("epoch")) == step)]
    if len(candidates) != 1:
        raise ValueError("Identify one official evaluation with --official-step or a single-result JSON")
    reference = {key.split("/")[-1]: value for key, value in candidates[0].items()}
    if "AP50" not in reference and "mAP" in reference:
        reference["AP50"] = reference["mAP"]
        report["ap_reference"] = "mAP (single IoU=0.5)"
    else:
        report["ap_reference"] = "AP50"
    for key in ("AP50", "Recall", "gt_count"):
        if key not in reference:
            report["checks"][key] = {"status": "unavailable"}
            continue
        expected = float(reference[key])
        if key != "gt_count" and not 0 <= expected <= 1:
            raise ValueError("Official AP/Recall must be fractions, not percentages")
        actual = metrics[key]
        tolerance = 0 if key == "gt_count" else 0.5 * 10 ** (-decimals) + 1e-8
        if key == "AP50" and report["ap_reference"].startswith("mAP"):
            tolerance = 1e-6
        passed = actual is not None and abs(actual-expected) <= tolerance
        report["checks"][key] = dict(status="pass" if passed else "fail",
                                      actual=actual, reference=expected,
                                      difference=None if actual is None else actual-expected,
                                      tolerance=tolerance)
    statuses = [row["status"] for row in report["checks"].values()]
    report["status"] = "fail" if "fail" in statuses else ("partial" if "unavailable" in statuses else "pass")
    report["reference"] = file_info(path)
    return report


def official_online(data, checkpoint, device):
    import sys
    sys.path.insert(0, str(PROJECT))
    import torch
    from mmengine.evaluator.metric import _to_cpu
    from mmengine.runner import Runner
    from mmdet.apis import init_detector
    from mmdet.utils import register_all_modules
    register_all_modules()
    cfg = data.config.copy()
    settings = cfg.test_dataloader.dataset
    settings.data_root = ""
    settings.data_prefix = dict(sub_data_root="")
    settings.ann_file = data.metadata["id_list"]["path"]
    settings.img_subdir = data.metadata["image_dir"]
    settings.ann_subdir = data.metadata["ann_dir"]
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False
    loader = Runner.build_dataloader(cfg.test_dataloader)
    model = init_detector(cfg, str(checkpoint), device=device)
    predictions = []
    with torch.no_grad():
        for batch in loader:
            for sample in model.test_step(batch):
                item = sample.to_dict() if hasattr(sample, "to_dict") else sample
                for key in ("gt_instances", "ignored_instances"):
                    item.pop(key, None)
                predictions.append(_to_cpu(item))
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Required online; provenance only in cache mode")
    parser.add_argument("--save-predictions", type=Path)
    parser.add_argument("--model", default="model")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--id-list", type=Path)
    parser.add_argument("--ann-dir", type=Path)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-thr", type=float)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output", type=Path, help="Also write the previous combined CSV")
    parser.add_argument("--include-overall", action="store_true")
    parser.add_argument("--official-metrics", type=Path)
    parser.add_argument("--official-step", type=int)
    parser.add_argument("--reference-decimals", type=int, default=3)
    parser.add_argument("--require-parity", action="store_true")
    args = parser.parse_args()
    if args.iou_thr != 0.5 or args.reference_decimals < 0:
        parser.error("Require IoU=0.5 and nonnegative reference decimals")
    if not args.predictions and not args.checkpoint:
        parser.error("Provide --predictions or --checkpoint for official online inference")
    kwargs = data_arguments(args)
    kwargs.pop("score_thr")
    kwargs.pop("iou_thr")
    data = load_dataset(args.config, **kwargs)
    evaluator = data.config.test_evaluator
    if isinstance(evaluator, (list, tuple)):
        evaluator = next(v for v in evaluator if v.get("type") == "VOCMetric")
    if evaluator.get("iou_thrs", [0.5]) not in (0.5, [0.5], (0.5,)):
        parser.error("Use a VOCMetric configured for only IoU=0.5")
    if args.predictions:
        payload = load_prediction_file(args.predictions)
    else:
        payload = official_online(data, args.checkpoint, args.device)
        if args.save_predictions:
            import pickle
            args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
            with args.save_predictions.open("wb") as stream:
                pickle.dump(payload, stream)
    attach_predictions(data, payload, args.predictions or args.save_predictions,
                       score_thr=args.score_thr, iou_thr=args.iou_thr, checkpoint=args.checkpoint)
    rows = []
    for scope, view, band, images in groups(data):
        metrics = evaluate(images, data.classes, args.iou_thr, data.metadata["protocol"]["eval_mode"])
        rows.append(dict(model=args.model, evaluation_id=data.metadata["evaluation_id"],
                         scope=scope, view=view, band_type=band, **metrics))
    overall, domains = rows[0], rows[1:]
    for key in ("image_count", "gt_count", "tp", "fp", "fn", "ignored_prediction_count"):
        if sum(row[key] for row in domains) != overall[key]:
            raise AssertionError(f"Domain sum != overall: {key}")
    parity = parity_report(overall, args.official_metrics, args.official_step, args.reference_decimals)
    output = args.output_dir or (args.output.parent / args.output.stem if args.output else
                                 Path("results/ablation") / args.model)
    write_csv(output / "overall_metrics.csv", [overall])
    write_csv(output / "domain_metrics.csv", domains)
    write_json(output / "parity_report.json", parity)
    data.metadata.update(model=args.model, parity_status=parity["status"])
    write_json(output / "evaluation_metadata.json", data.metadata)
    if args.output:
        write_csv(args.output, rows if args.include_overall else domains)
    print(f"{args.model}: AP50={overall['AP50']:.6f}, Recall={overall['Recall']}, "
          f"GT={overall['gt_count']}, parity={parity['status']}")
    print(f"Reports: {output.resolve()}")
    return int(parity["status"] == "fail" or (args.require_parity and parity["status"] != "pass"))


if __name__ == "__main__":
    raise SystemExit(main())
