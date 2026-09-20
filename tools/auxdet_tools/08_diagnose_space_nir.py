#!/usr/bin/env python3
"""Diagnose Space/NIR resolution, detector caps, and near-miss false negatives.

The report answers the following questions for Space/NIR images by default:

1. What are the original image resolutions, and how close are they to the
   configured Resize target (normally 1024 x 1024)?
2. What are the configured RPN proposal and RCNN output caps, and how many
   images actually produce a number of final predictions equal to the RCNN
   cap?
3. How many strict IoU=0.50 false negatives become matches with IoU=0.25 or
   a center-distance rule?

The script writes a JSON summary and a per-image CSV. It uses the same
metadata-aware inference helper as scripts 03, 04, and 06.

Example (run from the repository root)::

    python tools/auxdet_tools/08_diagnose_space_nir.py \
        --config configs/auxdet/auxdet_r50_fpn_1x_voc.py \
        --checkpoint work_dirs/auxdet/latest.pth \
        --ann-dir data_root/VOC2007/Annotations \
        --image-dir data_root/VOC2007/PNGImages \
        --id-list data_root/VOC2007/ImageSets/Main/val.txt \
        --output-dir data_root/VOC2007/space_nir_diagnosis \
        --device cuda:0

To inspect every view/band combination instead, pass ``--all-types``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import warnings
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


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
        help="Optional train/val ID list. Omit to scan all existing XML files.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--view", default="Space", help="Focused view; default: Space."
    )
    parser.add_argument(
        "--band-type", default="NIR", help="Focused band; default: NIR."
    )
    parser.add_argument(
        "--all-types", action="store_true",
        help="Analyze all view/band combinations instead of only Space/NIR.",
    )
    parser.add_argument(
        "--near-1024-tol", type=int, default=128,
        help="Tolerance in pixels for calling width/height near 1024.",
    )
    parser.add_argument(
        "--relaxed-iou", type=float, default=0.25,
        help="Relaxed IoU threshold for near-miss analysis.",
    )
    parser.add_argument(
        "--center-distance-px", type=float, default=10.0,
        help="Center-distance threshold in original-image pixels.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data_root/VOC2007/space_nir_diagnosis"),
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


def get_classes(dataset_cfg) -> tuple[str, ...]:
    classes = dataset_cfg.get("metainfo", {}).get("classes", ("Target",))
    return (classes,) if isinstance(classes, str) else tuple(classes)


def find_resize_scales(value: Any) -> list[Any]:
    """Find every Resize.scale in a Config/ConfigDict/list structure."""
    found: list[Any] = []
    if isinstance(value, dict):
        if value.get("type") == "Resize" and "scale" in value:
            found.append(value["scale"])
        for child in value.values():
            found.extend(find_resize_scales(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.extend(find_resize_scales(child))
    return found


def get_model_caps(config) -> dict[str, Optional[int]]:
    """Read RPN proposal and RCNN output caps from the model test config."""
    test_cfg = config.get("model", {}).get("test_cfg", {})
    rpn_cfg = test_cfg.get("rpn", {})
    rcnn_cfg = test_cfg.get("rcnn", {})

    def as_int(value):
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "rpn_max_per_img": as_int(rpn_cfg.get("max_per_img")),
        "rcnn_max_per_img": as_int(rcnn_cfg.get("max_per_img")),
        "rcnn_score_thr": float(rcnn_cfg["score_thr"])
        if rcnn_cfg.get("score_thr") is not None
        else None,
    }


def image_index(directory: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    if not directory.is_dir():
        return result
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            result.setdefault(path.stem.lower(), path)
    return result


def read_image_size(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            return int(image.width), int(image.height)
    except ImportError:
        import cv2

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Cannot read image: {image_path}")
        return int(image.shape[1]), int(image.shape[0])


def xml_size(root: ET.Element) -> tuple[Optional[int], Optional[int]]:
    size = root.find("size")
    if size is None:
        return None, None
    try:
        return int(float(size.findtext("width"))), int(float(size.findtext("height")))
    except (TypeError, ValueError):
        return None, None


def read_gt(
    root: ET.Element, classes: tuple[str, ...]
) -> tuple[list[tuple[float, float, float, float]], int]:
    """Return non-difficult GT boxes and total difficult count."""
    boxes: list[tuple[float, float, float, float]] = []
    difficult_count = 0
    for obj in root.findall("object"):
        if obj.findtext("name", default="").strip() not in classes:
            continue
        difficult = int(obj.findtext("difficult", default="0") or 0)
        if difficult:
            difficult_count += 1
            continue
        node = obj.find("bndbox")
        if node is None:
            continue
        try:
            x1, y1, x2, y2 = [
                float(node.findtext(key))
                for key in ("xmin", "ymin", "xmax", "ymax")
            ]
        except (TypeError, ValueError):
            warnings.warn("A malformed bndbox was skipped.")
            continue
        boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    return boxes, difficult_count


def read_predictions(result) -> list[tuple[float, tuple[float, float, float, float]]]:
    pred = getattr(result, "pred_instances", None)
    if pred is None or not hasattr(pred, "bboxes"):
        return []
    bboxes = pred.bboxes.detach().cpu().numpy()
    scores = pred.scores.detach().cpu().numpy()
    return sorted(
        [
            (float(score), tuple(float(value) for value in box[:4]))
            for box, score in zip(bboxes, scores)
        ],
        key=lambda item: item[0],
        reverse=True,
    )


def iou_legacy(first, second) -> float:
    """IoU using MMDetection's VOC legacy (+1 width/height) convention."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left + 1.0) * max(0.0, bottom - top + 1.0)
    first_area = max(0.0, first[2] - first[0] + 1.0) * max(0.0, first[3] - first[1] + 1.0)
    second_area = max(0.0, second[2] - second[0] + 1.0) * max(0.0, second[3] - second[1] + 1.0)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def center_distance(first, second) -> float:
    first_center = ((first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0)
    second_center = ((second[0] + second[2]) / 2.0, (second[1] + second[3]) / 2.0)
    return math.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1])


def match_count(gt_boxes, prediction_boxes, quality, threshold, higher_better=True) -> int:
    """VOC-style confidence-ordered one-to-one matching.

    ``quality`` is IoU for IoU matching (larger is better), or center distance
    for center matching (smaller is better). This mirrors VOC's rule of
    assigning a detection to its best available GT rather than the first GT.
    """
    used_gt: set[int] = set()
    matched = 0
    for _, prediction in prediction_boxes:
        if not gt_boxes:
            continue
        # Match to the best-overlap/closest GT among *all* GT boxes first.
        # If that GT was already covered, VOC counts this detection as a false
        # positive; it does not fall through to a second-best GT.
        candidates = [
            (quality(prediction, gt), index)
            for index, gt in enumerate(gt_boxes)
        ]
        best_quality, best_index = (
            max(candidates) if higher_better else min(candidates)
        )
        if (higher_better and best_quality < threshold) or (
            not higher_better and best_quality > threshold
        ):
            continue
        if best_index in used_gt:
            continue
        used_gt.add(best_index)
        matched += 1
    return matched


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
    return next((path.resolve() for path in candidates if path.is_file()), None)


def install_rpn_probe(model):
    """Capture the number of proposals returned by the RPN per inference.

    ``tools/test.py`` does not expose proposals, so this lightweight wrapper
    observes the exact proposals passed from the RPN to the ROI head without
    changing model outputs. It returns ``None`` for detector types without an
    RPN ``predict`` method.
    """
    rpn_head = getattr(model, "rpn_head", None)
    original_predict = getattr(rpn_head, "predict", None)
    if original_predict is None:
        return None
    captured: list[int] = []

    def wrapped_predict(*args, **kwargs):
        proposals = original_predict(*args, **kwargs)
        captured.clear()
        for proposal in proposals:
            boxes = getattr(proposal, "bboxes", None)
            captured.append(int(len(boxes)) if boxes is not None else int(len(proposal)))
        return proposals

    rpn_head.predict = wrapped_predict
    return captured


def main() -> None:
    args = parse_args()
    for path, name in (
        (args.config, "config"),
        (args.checkpoint, "checkpoint"),
        (args.ann_dir, "annotation directory"),
        (args.image_dir, "image directory"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path.resolve()}")
    if args.near_1024_tol < 0 or args.relaxed_iou <= 0 or args.relaxed_iou > 1:
        raise ValueError("Invalid resolution tolerance or relaxed IoU threshold.")
    if args.center_distance_px <= 0:
        raise ValueError("--center-distance-px must be positive.")

    try:
        from mmengine.config import Config
        from mmdet.apis import init_detector
    except ImportError as exc:
        raise RuntimeError("Run this script in the MMDetection environment.") from exc

    config = Config.fromfile(str(args.config.resolve()))
    dataset_cfg = unwrap_dataset_cfg(config.test_dataloader.dataset)
    classes = get_classes(dataset_cfg)
    resize_scales = find_resize_scales(config.test_dataloader.dataset.pipeline)
    caps = get_model_caps(config)
    id_list = resolve_id_list(args, dataset_cfg)
    selected_ids = None
    if id_list is not None:
        selected_ids = {
            Path(line.strip()).stem
            for line in id_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    helpers = load_helpers()
    model = init_detector(
        str(args.config.resolve()), str(args.checkpoint.resolve()), device=args.device
    )
    rpn_probe = install_rpn_probe(model)
    image_files = image_index(args.image_dir.resolve())
    records: list[dict] = []
    for xml_path in sorted(args.ann_dir.resolve().rglob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            warnings.warn(f"Invalid XML {xml_path}: {exc}; skipped.")
            continue
        image_id = helpers.stem(root, xml_path)
        if selected_ids is not None and image_id not in selected_ids:
            continue
        view = helpers.text(root.find("view"), "Unknown")
        band_type = helpers.text(root.find("band_type"), "Unknown")
        if not args.all_types and (view, band_type) != (args.view, args.band_type):
            continue
        image_path = image_files.get(image_id.lower()) or helpers.find_image(
            args.image_dir.resolve(), image_id
        )
        if image_path is None:
            warnings.warn(f"Image not found for {image_id}; skipped.")
            continue
        width, height = read_image_size(image_path)
        xml_width, xml_height = xml_size(root)
        try:
            result = helpers.detector_inference(model, image_path, root)
            predictions = read_predictions(result)
            rpn_proposal_count = rpn_probe[0] if rpn_probe else None
        except Exception as exc:
            warnings.warn(f"Inference failed for {image_id}: {exc}")
            continue
        gt_boxes, difficult_count = read_gt(root, classes)
        strict_matches = match_count(
            gt_boxes,
            predictions,
            iou_legacy,
            0.50,
        )
        relaxed_matches = match_count(
            gt_boxes,
            predictions,
            iou_legacy,
            args.relaxed_iou,
        )
        center_matches = match_count(
            gt_boxes,
            predictions,
            center_distance,
            args.center_distance_px,
            higher_better=False,
        )
        prediction_count = len(predictions)
        rcnn_cap = caps["rcnn_max_per_img"]
        rpn_cap = caps["rpn_max_per_img"]
        records.append(
            {
                "image_id": image_id,
                "view": view,
                "band_type": band_type,
                "width": width,
                "height": height,
                "xml_width": xml_width,
                "xml_height": xml_height,
                "gt_count": len(gt_boxes),
                "difficult_gt_count": difficult_count,
                "prediction_count": prediction_count,
                "rcnn_cap": rcnn_cap,
                "rcnn_cap_hit": bool(rcnn_cap is not None and prediction_count >= rcnn_cap),
                "rpn_proposal_count": rpn_proposal_count,
                "rpn_cap": rpn_cap,
                "rpn_cap_hit": bool(
                    rpn_cap is not None
                    and rpn_proposal_count is not None
                    and rpn_proposal_count >= rpn_cap
                ),
                "strict_iou50_matches": strict_matches,
                "strict_iou50_fn": max(0, len(gt_boxes) - strict_matches),
                "relaxed_iou_matches": relaxed_matches,
                "relaxed_iou_fn": max(0, len(gt_boxes) - relaxed_matches),
                "center_matches": center_matches,
                "center_fn": max(0, len(gt_boxes) - center_matches),
            }
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "per_image.csv"
    csv_fields = [
        "image_id", "view", "band_type", "width", "height", "xml_width", "xml_height",
        "gt_count", "difficult_gt_count", "prediction_count", "rcnn_cap", "rcnn_cap_hit",
        "rpn_proposal_count", "rpn_cap", "rpn_cap_hit",
        "strict_iou50_matches", "strict_iou50_fn", "relaxed_iou_matches", "relaxed_iou_fn",
        "center_matches", "center_fn",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(records)

    groups: defaultdict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        groups[(record["view"], record["band_type"])].append(record)

    def summarize(items: list[dict]) -> dict:
        total_gt = sum(item["gt_count"] for item in items)
        strict_fn = sum(item["strict_iou50_fn"] for item in items)
        relaxed_fn = sum(item["relaxed_iou_fn"] for item in items)
        center_fn = sum(item["center_fn"] for item in items)
        near_relaxed = max(0, strict_fn - relaxed_fn)
        near_center = max(0, strict_fn - center_fn)
        cap_hits = sum(1 for item in items if item["rcnn_cap_hit"])
        rpn_cap_hits = sum(1 for item in items if item["rpn_cap_hit"])
        widths = [item["width"] for item in items]
        heights = [item["height"] for item in items]
        near_width = sum(abs(width - 1024) <= args.near_1024_tol for width in widths)
        near_height = sum(abs(height - 1024) <= args.near_1024_tol for height in heights)
        return {
            "image_count": len(items),
            "gt_count": total_gt,
            "strict_iou50_fn": strict_fn,
            "relaxed_iou_fn": relaxed_fn,
            "strict_fn_rescued_by_relaxed_iou": near_relaxed,
            "center_fn": center_fn,
            "strict_fn_rescued_by_center_distance": near_center,
            "rcnn_cap_hit_images": cap_hits,
            "rcnn_cap_hit_fraction": cap_hits / len(items) if items else 0.0,
            "rpn_cap_hit_images": rpn_cap_hits,
            "rpn_cap_hit_fraction": rpn_cap_hits / len(items) if items else 0.0,
            "width_min": min(widths) if widths else None,
            "width_median": sorted(widths)[len(widths) // 2] if widths else None,
            "width_max": max(widths) if widths else None,
            "height_min": min(heights) if heights else None,
            "height_median": sorted(heights)[len(heights) // 2] if heights else None,
            "height_max": max(heights) if heights else None,
            "width_near_1024_count": near_width,
            "height_near_1024_count": near_height,
            "width_near_1024_fraction": near_width / len(items) if items else 0.0,
            "height_near_1024_fraction": near_height / len(items) if items else 0.0,
        }

    group_summary = {
        f"{view}/{band_type}": summarize(items)
        for (view, band_type), items in sorted(groups.items())
    }
    summary = {
        "scope": "all view/band types" if args.all_types else f"{args.view}/{args.band_type}",
        "image_id_list": str(id_list) if id_list else None,
        "configured_resize_scales": [list(scale) if isinstance(scale, tuple) else scale
                                      for scale in resize_scales],
        "near_1024_tolerance_px": args.near_1024_tol,
        "model_caps": caps,
        "rpn_cap_direct_hit_observable": any(
            item["rpn_proposal_count"] is not None for item in records
        ),
        "rpn_cap_note": "rpn_proposal_count is captured immediately before the ROI head.",
        "relaxed_iou_threshold": args.relaxed_iou,
        "center_distance_threshold_px": args.center_distance_px,
        "groups": group_summary,
        "per_image_csv": str(csv_path),
    }
    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Images analyzed: {len(records)}")
    print(f"Per-image CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")
    print(f"Configured Resize scales: {summary['configured_resize_scales']}")
    print(f"RPN max_per_img: {caps['rpn_max_per_img']}")
    print(f"RCNN max_per_img: {caps['rcnn_max_per_img']}")
    for name, values in group_summary.items():
        print(
            f"{name}: images={values['image_count']}, "
            f"near-1024 width={values['width_near_1024_fraction']:.1%}, "
            f"height={values['height_near_1024_fraction']:.1%}, "
            f"RCNN-cap hits={values['rcnn_cap_hit_images']}, "
            f"RPN-cap hits={values['rpn_cap_hit_images']}, "
            f"strict FN={values['strict_iou50_fn']}, "
            f"rescued IoU{args.relaxed_iou:.2f}={values['strict_fn_rescued_by_relaxed_iou']}, "
            f"rescued center={values['strict_fn_rescued_by_center_distance']}"
        )


if __name__ == "__main__":
    main()
