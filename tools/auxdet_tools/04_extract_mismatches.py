#!/usr/bin/env python3
"""Test every annotated image and extract detector/GT mismatches.

One predicted box is considered matched when its IoU with one still-unmatched
ground-truth box is at least ``--iou-thr``.  The image is a mismatch if the
number of matched pairs is smaller than either the GT count or prediction
count.  Mismatched images are copied to ``--output-dir`` and details are saved
in ``mismatch.csv``.

Example:
    python tools/auxdet_tools/04_extract_mismatches.py \
        --config configs/auxdet/auxdet_r50_fpn_1x_voc.py \
        --checkpoint work_dirs/auxdet/latest.pth \
        --ann-dir data_root/VOC2007/Annotations \
        --image-dir data_root/VOC2007/PNGImages \
        --id-list data_root/VOC2007/ImageSets/Main/val.txt \
        --output-dir data_root/VOC2007/mismatches \
        --device cuda:0 --score-thr 0.30 --iou-thr 0.50

To copy the already annotated green+red images instead of raw PNGs, add
``--source-dir data_root/VOC2007/green_and_red_boxes``.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def text(node: ET.Element | None, default: str = "") -> str:
    if node is None or node.text is None or not node.text.strip():
        return default
    return node.text.strip()


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
        "--source-dir",
        type=Path,
        default=None,
        help="Directory to copy on mismatch; defaults to --image-dir.",
    )
    parser.add_argument(
        "--id-list",
        type=Path,
        default=None,
        help="Optional ImageSets list (for example val.txt) used as the test split.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data_root/VOC2007/mismatches")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-thr", type=float, default=0.30)
    parser.add_argument("--iou-thr", type=float, default=0.50)
    return parser.parse_args()


def stem(root: ET.Element, xml_path: Path) -> str:
    return Path(text(root.find("filename"), xml_path.stem)).stem or xml_path.stem


def read_gt(root: ET.Element) -> list[tuple[float, float, float, float]]:
    boxes = []
    for obj in root.findall("object"):
        node = obj.find("bndbox")
        if node is None:
            continue
        try:
            x1, y1, x2, y2 = [
                float(text(node.find(name), "0"))
                for name in ("xmin", "ymin", "xmax", "ymax")
            ]
        except ValueError:
            warnings.warn("A non-numeric bndbox was skipped.")
            continue
        boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    return boxes


def find_image(directory: Path, image_stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = directory / f"{image_stem}{extension}"
        if candidate.is_file():
            return candidate
    if directory.is_dir():
        for candidate in directory.iterdir():
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                if candidate.stem.lower() == image_stem.lower():
                    return candidate
    return None


def metadata(root: ET.Element, image_path: Path) -> tuple[str, str, int, int]:
    view = text(root.find("view"), "Unknown")
    band = text(root.find("band_type"), "Unknown")
    size = root.find("size")
    try:
        width = int(float(text(size.find("width"), "0"))) if size is not None else 0
        height = int(float(text(size.find("height"), "0"))) if size is not None else 0
    except ValueError:
        width, height = 0, 0
    if width <= 0 or height <= 0:
        import cv2

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Cannot read image to determine size: {image_path}")
        height, width = image.shape[:2]
    return view, band, width, height


def detector_inference(model, image_path: Path, root: ET.Element):
    import torch
    from mmcv.transforms import Compose
    from mmdet.utils import get_test_pipeline_cfg

    view, band, width, height = metadata(root, image_path)
    pipeline = Compose(get_test_pipeline_cfg(model.cfg.copy()))
    data = pipeline(
        {
            "img_path": str(image_path),
            "img_id": 0,
            "width": width,
            "height": height,
            "view": view,
            "band_type": band,
        }
    )
    data["inputs"] = [data["inputs"]]
    data["data_samples"] = [data["data_samples"]]
    with torch.no_grad():
        return model.test_step(data)[0]


def read_predictions(result, score_thr: float) -> list[tuple[float, float, float, float]]:
    pred = getattr(result, "pred_instances", None)
    if pred is None or not hasattr(pred, "bboxes"):
        return []
    bboxes = pred.bboxes.detach().cpu().numpy()
    scores = (
        pred.scores.detach().cpu().numpy()
        if hasattr(pred, "scores")
        else [1.0] * len(bboxes)
    )
    return [
        tuple(float(value) for value in box[:4])
        for box, score in zip(bboxes, scores)
        if float(score) >= score_thr
    ]


def iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(
    gt_boxes: list[tuple[float, float, float, float]],
    pred_boxes: list[tuple[float, float, float, float]],
    threshold: float,
) -> int:
    """Greedily form one-to-one pairs in descending IoU order."""
    candidates = sorted(
        (
            iou(gt, pred),
            gt_index,
            pred_index,
        )
        for gt_index, gt in enumerate(gt_boxes)
        for pred_index, pred in enumerate(pred_boxes)
        if iou(gt, pred) >= threshold
    )
    candidates.reverse()
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    for _, gt_index, pred_index in candidates:
        if gt_index not in used_gt and pred_index not in used_pred:
            used_gt.add(gt_index)
            used_pred.add(pred_index)
    return len(used_gt)


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
    source_dir = (args.source_dir or args.image_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    selected_ids = None
    if args.id_list is not None:
        if not args.id_list.is_file():
            raise FileNotFoundError(f"ID list does not exist: {args.id_list.resolve()}")
        selected_ids = {
            Path(line.strip()).stem
            for line in args.id_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    try:
        from mmdet.apis import init_detector
    except ImportError as exc:
        raise RuntimeError("Run this script in the MMDetection environment.") from exc
    model = init_detector(
        str(args.config.resolve()), str(args.checkpoint.resolve()), device=args.device
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "mismatch.csv"
    rows = []
    tested = 0
    mismatch_count = 0
    skipped = 0
    for xml_path in sorted(args.ann_dir.resolve().rglob("*.xml")):
        root = ET.parse(xml_path).getroot()
        image_stem = stem(root, xml_path)
        if selected_ids is not None and image_stem not in selected_ids:
            continue
        image_path = find_image(args.image_dir.resolve(), image_stem)
        source_path = find_image(source_dir, image_stem)
        if image_path is None or source_path is None:
            warnings.warn(f"Missing image/source for {image_stem}; skipped.")
            skipped += 1
            continue
        try:
            result = detector_inference(model, image_path, root)
            gt_boxes = read_gt(root)
            pred_boxes = read_predictions(result, args.score_thr)
            matched = match_boxes(gt_boxes, pred_boxes, args.iou_thr)
        except Exception as exc:
            warnings.warn(f"Testing failed for {image_stem}: {exc}")
            skipped += 1
            continue
        tested += 1
        is_mismatch = matched < max(len(gt_boxes), len(pred_boxes))
        if not is_mismatch:
            continue
        mismatch_count += 1
        destination = output_dir / source_path.name
        shutil.copy2(source_path, destination)
        rows.append(
            {
                "image_id": image_stem,
                "view": text(root.find("view"), "Unknown"),
                "band_type": text(root.find("band_type"), "Unknown"),
                "gt_count": len(gt_boxes),
                "prediction_count": len(pred_boxes),
                "matched_count": matched,
                "score_thr": args.score_thr,
                "iou_thr": args.iou_thr,
            }
        )

    with report_path.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "image_id",
            "view",
            "band_type",
            "gt_count",
            "prediction_count",
            "matched_count",
            "score_thr",
            "iou_thr",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Images tested: {tested}")
    print(f"Mismatched images: {mismatch_count}")
    print(f"Skipped: {skipped}")
    print(f"Mismatch images: {output_dir}")
    print(f"Mismatch report: {report_path}")


if __name__ == "__main__":
    main()
