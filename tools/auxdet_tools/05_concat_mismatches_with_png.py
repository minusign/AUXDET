#!/usr/bin/env python3
"""Concatenate each mismatch image with its original PNG side by side.

The mismatch image is placed on the left and the corresponding image from
``PNGImages`` on the right.  Files are matched by filename stem, so this also
works when the two directories use different image extensions.

Example:
    python tools/auxdet_tools/05_concat_mismatches_with_png.py \
        --mismatch-dir data_root/VOC2007/mismatches \
        --png-dir data_root/VOC2007/PNGImages \
        --output-dir data_root/VOC2007/mismatch_side_by_side
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mismatch-dir",
        type=Path,
        default=Path("data_root/VOC2007/mismatches"),
        help="Directory containing mismatch images.",
    )
    parser.add_argument(
        "--png-dir",
        type=Path,
        default=Path("data_root/VOC2007/PNGImages"),
        help="Directory containing original PNG images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_root/VOC2007/mismatch_side_by_side"),
        help="Directory for concatenated images.",
    )
    parser.add_argument(
        "--no-label",
        action="store_true",
        help="Do not add the small MISMATCH/PNG labels at the top.",
    )
    return parser.parse_args()


def image_files(directory: Path) -> dict[str, Path]:
    """Return image paths indexed by lowercase filename stem."""
    result: dict[str, Path] = {}
    if not directory.is_dir():
        return result
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            result.setdefault(path.stem.lower(), path)
    return result


def to_bgr(image):
    """Convert grayscale/BGRA images to a drawable 3-channel BGR image."""
    import cv2

    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def resize_to_height(image, height: int):
    """Resize an image to height while preserving its aspect ratio."""
    import cv2

    if image.shape[0] == height:
        return image
    width = max(1, round(image.shape[1] * height / image.shape[0]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()
    mismatch_dir = args.mismatch_dir.resolve()
    png_dir = args.png_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not mismatch_dir.is_dir():
        raise FileNotFoundError(f"Mismatch directory does not exist: {mismatch_dir}")
    if not png_dir.is_dir():
        raise FileNotFoundError(f"PNG directory does not exist: {png_dir}")

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("This script requires opencv-python and numpy.") from exc

    mismatch_files = image_files(mismatch_dir)
    png_files = image_files(png_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    for stem, mismatch_path in sorted(mismatch_files.items()):
        png_path = png_files.get(stem)
        if png_path is None:
            warnings.warn(f"Original PNG not found for mismatch image: {mismatch_path.name}")
            skipped += 1
            continue

        mismatch = to_bgr(cv2.imread(str(mismatch_path), cv2.IMREAD_UNCHANGED))
        original = to_bgr(cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED))
        if mismatch is None or original is None:
            warnings.warn(f"Could not read pair: {mismatch_path.name} / {png_path.name}")
            skipped += 1
            continue

        # Match heights so the two panels align horizontally.
        target_height = max(mismatch.shape[0], original.shape[0])
        mismatch = resize_to_height(mismatch, target_height)
        original = resize_to_height(original, target_height)
        if not args.no_label:
            # Reserve a top strip and add labels without changing image scale.
            label_height = 28
            mismatch_panel = np.full(
                (target_height + label_height, mismatch.shape[1], 3),
                255,
                dtype=np.uint8,
            )
            original_panel = np.full(
                (target_height + label_height, original.shape[1], 3),
                255,
                dtype=np.uint8,
            )
            mismatch_panel[label_height:] = mismatch
            original_panel[label_height:] = original
            cv2.putText(
                mismatch_panel,
                "MISMATCH",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                original_panel,
                "PNG",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            mismatch, original = mismatch_panel, original_panel

        # A black one-pixel divider makes the left/right boundary unambiguous.
        divider = np.zeros((mismatch.shape[0], 2, 3), dtype=np.uint8)
        combined = np.concatenate((mismatch, divider, original), axis=1)
        destination = output_dir / f"{mismatch_path.stem}.png"
        if not cv2.imwrite(str(destination), combined):
            warnings.warn(f"Could not write: {destination}")
            skipped += 1
            continue
        written += 1

    print(f"Side-by-side images written: {written}")
    print(f"Skipped: {skipped}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
