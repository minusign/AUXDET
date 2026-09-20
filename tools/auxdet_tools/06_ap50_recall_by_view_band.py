#!/usr/bin/env python3
"""Evaluate AP50 and Recall by ``(view, band_type)`` with VOCMetric logic.

This script follows the same evaluation path as ``tools/test.py`` for this
project: ``VOCMetric -> eval_map``. It uses the configured ``eval_mode``, VOC
legacy box coordinates, model test-time outputs, and the XML ``difficult``
flag as ignored ground truth.

The Recall column is the final recall reported by the official VOC evaluator
at IoU 0.50. AP50 is rounded to three decimals, as in VOCMetric.

Example::

    python tools/auxdet_tools/06_ap50_recall_by_view_band.py \
        --config configs/auxdet/auxdet_r50_fpn_1x_voc.py \
        --checkpoint work_dirs/auxdet/latest.pth \
        --ann-dir data_root/VOC2007/Annotations \
        --image-dir data_root/VOC2007/PNGImages \
        --id-list data_root/VOC2007/ImageSets/Main/val.txt \
        --output data_root/VOC2007/ap50_recall_by_view_band.csv \
        --device cuda:0

Omit ``--score-thr`` when comparing with ``tools/test.py``. This leaves the
model's own ``test_cfg.rcnn.score_thr`` in control, exactly as test.py does.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--ann-dir", type=Path, default=Path("data_root/VOC2007/Annotations")
    )
    parser.add_argument(
        "--image-dir", type=Path, default=Path("data_root/VOC2007/PNGImages")
    )
    parser.add_argument(
        "--id-list", type=Path, default=None,
        help="Optional train/val ID list; pass val.txt to match test.py exactly.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data_root/VOC2007/ap50_recall_by_view_band.csv"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--score-thr", type=float, default=None,
        help="Optional extra filter. Omit it to use model test_cfg, like test.py.",
    )
    parser.add_argument(
        "--iou-thr", type=float, default=0.50,
        help="IoU threshold; keep 0.50 for AP50/test.py comparison.",
    )
    return parser.parse_args()


def load_helpers():
    helper_path = Path(__file__).with_name("04_extract_mismatches.py")
    spec = importlib.util.spec_from_file_location("auxdet_mismatch_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unwrap_dataset_cfg(dataset_cfg):
    while "dataset" in dataset_cfg:
        dataset_cfg = dataset_cfg["dataset"]
    if "datasets" in dataset_cfg and dataset_cfg["datasets"]:
        return unwrap_dataset_cfg(dataset_cfg["datasets"][0])
    return dataset_cfg


def resolve_evaluator(config):
    evaluator = config.get("test_evaluator", {})
    if isinstance(evaluator, (list, tuple)):
        for item in evaluator:
            if item.get("type") == "VOCMetric":
                return item
        return evaluator[0] if evaluator else {}
    return evaluator


def resolve_id_list(args: argparse.Namespace, dataset_cfg) -> Optional[Path]:
    if args.id_list is not None:
        path = args.id_list.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ID list does not exist: {path}")
        return path
    ann_file = dataset_cfg.get("ann_file")
    if not ann_file:
        return None
    ann_file = Path(str(ann_file))
    candidates = [Path.cwd() / ann_file]
    data_root = dataset_cfg.get("data_root")
    if data_root:
        candidates.append(Path(str(data_root)) / ann_file)
    candidates.append(args.ann_dir.resolve().parent / "ImageSets" / "Main" / ann_file.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    warnings.warn(f"Could not find configured ann_file {ann_file}; using all XML files.")
    return None


def class_names(dataset_cfg) -> tuple[str, ...]:
    classes = dataset_cfg.get("metainfo", {}).get("classes", ("Target",))
    return (classes,) if isinstance(classes, str) else tuple(classes)


def parse_annotation(root: ET.Element, classes: tuple[str, ...]):
    """Build the exact annotation shape consumed by VOCMetric.eval_map."""
    import numpy as np

    class_to_label = {name: index for index, name in enumerate(classes)}
    boxes, labels, ignored_boxes, ignored_labels = [], [], [], []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="").strip()
        if name not in class_to_label:
            continue
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        try:
            coords = [int(float(bbox.findtext(key))) for key in
                      ("xmin", "ymin", "xmax", "ymax")]
        except (TypeError, ValueError):
            warnings.warn("Malformed XML bndbox skipped.")
            continue
        difficult = int(obj.findtext("difficult", default="0") or 0)
        if difficult:
            ignored_boxes.append(coords)
            ignored_labels.append(class_to_label[name])
        else:
            boxes.append(coords)
            labels.append(class_to_label[name])
    return {
        "bboxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        "labels": np.asarray(labels, dtype=np.int64),
        "bboxes_ignore": np.asarray(ignored_boxes, dtype=np.float32).reshape(-1, 4),
        "labels_ignore": np.asarray(ignored_labels, dtype=np.int64),
    }


def parse_predictions(result, class_count: int, extra_score_thr: Optional[float]):
    """Convert model output to VOCMetric's per-class [x1,y1,x2,y2,score]."""
    import numpy as np

    pred = getattr(result, "pred_instances", None)
    if pred is None or not hasattr(pred, "bboxes"):
        return [np.empty((0, 5), dtype=np.float32) for _ in range(class_count)]
    bboxes = pred.bboxes.detach().cpu().numpy()
    scores = pred.scores.detach().cpu().numpy()
    labels = pred.labels.detach().cpu().numpy()
    output = []
    for label in range(class_count):
        keep = labels == label
        if extra_score_thr is not None:
            keep &= scores >= extra_score_thr
        output.append(np.hstack((bboxes[keep], scores[keep, None])).astype(np.float32))
    return output


def official_voc_metrics(predictions, annotations, classes, iou_thr, eval_mode,
                         scale_ranges):
    """Call exactly the function used by mmdet/evaluation/metrics/voc_metric.py."""
    from mmdet.evaluation.functional import eval_map

    mean_ap, class_results = eval_map(
        predictions,
        annotations,
        scale_ranges=scale_ranges,
        iou_thr=iou_thr,
        dataset=classes,
        logger="silent",
        eval_mode=eval_mode,
        use_legacy_coordinate=True,
    )
    gt_count = sum(len(item["bboxes"]) for item in annotations)
    ignored_gt_count = sum(len(item["bboxes_ignore"]) for item in annotations)
    prediction_count = sum(sum(len(cls) for cls in image) for image in predictions)

    # VOCMetric's eval_map result contains class-level cumulative curves. The
    # last point gives the official final recall at this IoU threshold.
    tp = 0
    fp = 0
    for result in class_results:
        recalls = result["recall"]
        precisions = result["precision"]
        final_recall = float(recalls[-1]) if getattr(recalls, "size", 0) else 0.0
        final_precision = float(precisions[-1]) if getattr(precisions, "size", 0) else 0.0
        class_gts = int(result["num_gts"])
        class_dets = int(result["num_dets"])
        class_tp = int(round(final_recall * class_gts))
        class_fp = int(round(class_tp / final_precision - class_tp)) if final_precision > 0 else class_dets
        tp += class_tp
        fp += max(0, class_fp)
    recall = tp / gt_count if gt_count else 0.0
    return {
        "AP50": round(float(mean_ap), 3),
        "Recall": round(float(recall), 3),
        "gt_count": gt_count,
        "ignored_gt_count": ignored_gt_count,
        "prediction_count": prediction_count,
        "tp": tp,
        "fp": fp,
        "fn": gt_count - tp,
    }


def main() -> None:
    args = parse_args()
    for path, name in ((args.config, "config"), (args.checkpoint, "checkpoint"),
                       (args.ann_dir, "annotation directory"),
                       (args.image_dir, "image directory")):
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path.resolve()}")
    if not 0.0 < args.iou_thr <= 1.0:
        raise ValueError("--iou-thr must be in (0, 1]")

    try:
        from mmengine.config import Config
        from mmdet.apis import init_detector
    except ImportError as exc:
        raise RuntimeError("Run this script in the MMDetection environment.") from exc

    config = Config.fromfile(str(args.config.resolve()))
    dataset_cfg = unwrap_dataset_cfg(config.test_dataloader.dataset)
    evaluator = resolve_evaluator(config)
    if evaluator.get("type", "VOCMetric") != "VOCMetric":
        raise ValueError("test_evaluator must be VOCMetric to match tools/test.py.")
    eval_mode = evaluator.get("eval_mode", "11points")
    scale_ranges = evaluator.get("scale_ranges", None)
    classes = class_names(dataset_cfg)
    id_list = resolve_id_list(args, dataset_cfg)
    selected_ids = None
    if id_list is not None:
        selected_ids = {Path(line.strip()).stem for line in
                        id_list.read_text(encoding="utf-8").splitlines() if line.strip()}

    helpers = load_helpers()
    model = init_detector(str(args.config.resolve()), str(args.checkpoint.resolve()),
                          device=args.device)
    groups = defaultdict(lambda: {"images": 0, "predictions": [], "annotations": []})
    tested, skipped = 0, 0
    for xml_path in sorted(args.ann_dir.resolve().rglob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            warnings.warn(f"Invalid XML {xml_path}: {exc}; skipped.")
            skipped += 1
            continue
        image_id = helpers.stem(root, xml_path)
        if selected_ids is not None and image_id not in selected_ids:
            continue
        image_path = helpers.find_image(args.image_dir.resolve(), image_id)
        if image_path is None:
            warnings.warn(f"Image not found for {image_id}; skipped.")
            skipped += 1
            continue
        try:
            result = helpers.detector_inference(model, image_path, root)
            annotation = parse_annotation(root, classes)
            prediction = parse_predictions(result, len(classes), args.score_thr)
        except Exception as exc:
            warnings.warn(f"Evaluation failed for {image_id}: {exc}")
            skipped += 1
            continue
        key = (helpers.text(root.find("view"), "Unknown"),
               helpers.text(root.find("band_type"), "Unknown"))
        groups[key]["images"] += 1
        groups[key]["predictions"].append(prediction)
        groups[key]["annotations"].append(annotation)
        tested += 1

    rows = []
    for (view, band_type), data in sorted(groups.items()):
        metrics = official_voc_metrics(data["predictions"], data["annotations"],
                                       classes, args.iou_thr, eval_mode, scale_ranges)
        rows.append({"view": view, "band_type": band_type,
                     "image_count": data["images"], **metrics})

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["view", "band_type", "image_count", "gt_count", "ignored_gt_count",
              "prediction_count", "tp", "fp", "fn", "AP50", "Recall"]
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Evaluation path: VOCMetric -> eval_map ({eval_mode})")
    print(f"Image ID list: {id_list if id_list else 'all existing XML files'}")
    print("Extra score filter: " +
          ("none (same as test.py)" if args.score_thr is None else str(args.score_thr)))
    print(f"Images evaluated: {tested}")
    print(f"Skipped: {skipped}")
    print(f"Groups: {len(rows)}")
    print(f"Report: {output}")
    for row in rows:
        print(f"{row['view']}/{row['band_type']}: AP50={row['AP50']:.3f}, "
              f"Recall={row['Recall']:.3f}")


if __name__ == "__main__":
    main()
