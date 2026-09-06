#!/usr/bin/env python3
"""Run the detector and add red prediction boxes to green GT images.

The detector receives the original image, while the output canvas is read from
``--green-dir`` (the output of ``02_draw_green_gt.py``).  This matters for the
AuxFPN model in this repository: its forward pass needs ``view``, ``band_type``,
``width`` and ``height`` from each image's XML metadata.

Example:
    python tools/auxdet_tools/03_draw_red_predictions.py \
        --config configs/auxdet/auxdet_r50_fpn_1x_voc.py \
        --checkpoint work_dirs/auxdet/latest.pth \
        --image-dir data_root/VOC2007/PNGImages \
        --ann-dir data_root/VOC2007/Annotations \
        --green-dir data_root/VOC2007/green_boxes \
        --output-dir data_root/VOC2007/green_and_red_boxes \
        --device cuda:0 --score-thr 0.30
"""

from __future__ import annotations

import argparse
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
        "--green-dir", type=Path, default=Path("data_root/VOC2007/green_boxes")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_root/VOC2007/green_and_red_boxes"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-thr", type=float, default=0.30)
    parser.add_argument("--thickness", type=int, default=2)
    return parser.parse_args()


def image_stem(root: ET.Element, xml_path: Path) -> str:
    return Path(text(root.find("filename"), xml_path.stem)).stem or xml_path.stem


def metadata(root: ET.Element, image_path: Path) -> tuple[str, str, int, int]:
    """Return view, band, width and height for the AuxFPN data sample."""
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


def find_image(image_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    if image_dir.is_dir():
        for candidate in image_dir.iterdir():
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                if candidate.stem.lower() == stem.lower():
                    return candidate
    return None


def detector_inference(model, image_path: Path, root: ET.Element):
    """Infer one image while preserving the XML metadata required by AuxFPN."""
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


def predictions(result, score_thr: float) -> list[tuple[float, float, float, float, float, int]]:
    pred = getattr(result, "pred_instances", None)
    if pred is None or not hasattr(pred, "bboxes"):
        return []
    bboxes = pred.bboxes.detach().cpu().numpy()
    scores = (
        pred.scores.detach().cpu().numpy()
        if hasattr(pred, "scores")
        else [1.0] * len(bboxes)
    )
    labels = (
        pred.labels.detach().cpu().numpy()
        if hasattr(pred, "labels")
        else [0] * len(bboxes)
    )
    output = []
    for box, score, label in zip(bboxes, scores, labels):
        if float(score) >= score_thr:
            output.append(
                (
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                    float(score),
                    int(label),
                )
            )
    return output


def draw_predictions(
    image_path: Path,
    destination: Path,
    detections: list[tuple[float, float, float, float, float, int]],
    thickness: int,
) -> None:
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Cannot read green-box image: {image_path}")
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        canvas = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        canvas = image.copy()
    for index, (x1, y1, x2, y2, score, label) in enumerate(detections, start=1):
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(canvas, p1, p2, (0, 0, 255), thickness)
        cv2.putText(
            canvas,
            f"Pred {index} {score:.2f} (c{label})",
            (p1[0], max(14, p1[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), canvas):
        raise RuntimeError(f"Cannot write image: {destination}")


def main() -> None:
    args = parse_args()
    for path, name in (
        (args.config, "config"),
        (args.checkpoint, "checkpoint"),
        (args.ann_dir, "annotation directory"),
        (args.image_dir, "image directory"),
        (args.green_dir, "green-box directory"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path.resolve()}")

    try:
        from mmdet.apis import init_detector
    except ImportError as exc:
        raise RuntimeError("Run this script in the MMDetection environment.") from exc

    model = init_detector(
        str(args.config.resolve()), str(args.checkpoint.resolve()), device=args.device
    )
    args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
    done = 0
    skipped = 0
    for xml_path in sorted(args.ann_dir.resolve().rglob("*.xml")):
        root = ET.parse(xml_path).getroot()
        stem = image_stem(root, xml_path)
        original = find_image(args.image_dir.resolve(), stem)
        green = find_image(args.green_dir.resolve(), stem)
        if original is None or green is None:
            warnings.warn(
                f"Missing original/green image for {stem}; prediction skipped."
            )
            skipped += 1
            continue
        try:
            result = detector_inference(model, original, root)
            detections = predictions(result, args.score_thr)
            draw_predictions(
                green,
                args.output_dir.resolve() / green.name,
                detections,
                args.thickness,
            )
        except Exception as exc:
            warnings.warn(f"Prediction failed for {stem}: {exc}")
            skipped += 1
            continue
        done += 1

    print(f"Green+red images written: {done}")
    print(f"Skipped: {skipped}")
    print(f"Output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
