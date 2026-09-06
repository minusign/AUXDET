#!/usr/bin/env python3
"""Draw XML ground-truth boxes in green on the corresponding PNG files.

Example:
    python tools/auxdet_tools/02_draw_green_gt.py \
        --image-dir data_root/VOC2007/PNGImages \
        --ann-dir data_root/VOC2007/Annotations \
        --output-dir data_root/VOC2007/green_boxes
"""

from __future__ import annotations

import argparse
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def text(node: ET.Element | None, default: str = "") -> str:
    if node is None or node.text is None:
        return default
    return node.text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ann-dir", type=Path, default=Path("data_root/VOC2007/Annotations")
    )
    parser.add_argument(
        "--image-dir", type=Path, default=Path("data_root/VOC2007/PNGImages")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data_root/VOC2007/green_boxes")
    )
    parser.add_argument("--thickness", type=int, default=2)
    return parser.parse_args()


def image_id(root: ET.Element, xml_path: Path) -> str:
    return Path(text(root.find("filename"), xml_path.stem)).stem or xml_path.stem


def read_boxes(root: ET.Element) -> list[tuple[int, int, int, int, str]]:
    boxes = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            values = [
                int(round(float(text(box.find(name), "0"))))
                for name in ("xmin", "ymin", "xmax", "ymax")
            ]
        except ValueError:
            warnings.warn("A non-numeric bndbox was skipped.")
            continue
        x1, y1, x2, y2 = values
        boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), text(obj.find("name"), "GT")))
    return boxes


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


def main() -> None:
    args = parse_args()
    ann_dir = args.ann_dir.resolve()
    image_dir = args.image_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory does not exist: {ann_dir}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("This script requires opencv-python.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    skipped = 0
    for xml_path in sorted(ann_dir.rglob("*.xml")):
        root = ET.parse(xml_path).getroot()
        stem = image_id(root, xml_path)
        source = find_image(image_dir, stem)
        if source is None:
            warnings.warn(f"PNG not found for {xml_path.name}: {stem}")
            skipped += 1
            continue
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None:
            warnings.warn(f"Image cannot be read: {source}")
            skipped += 1
            continue
        if image.ndim == 2:
            canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            canvas = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            canvas = image.copy()

        for x1, y1, x2, y2, label in read_boxes(root):
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), args.thickness)
            cv2.putText(
                canvas,
                label,
                (x1, max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        destination = output_dir / source.name
        if not cv2.imwrite(str(destination), canvas):
            warnings.warn(f"Image cannot be written: {destination}")
            skipped += 1
            continue
        done += 1

    print(f"Green-box images written: {done}")
    print(f"Skipped (missing/unreadable): {skipped}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
