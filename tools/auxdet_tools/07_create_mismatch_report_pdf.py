#!/usr/bin/env python3
"""Create a PDF report sampled by ``(view, band_type)``.

For every view/band combination, the script selects at most 10 mismatched
images.  Images with more unmatched GT/prediction boxes are selected first.
Each report entry contains the original image together with the corresponding
``mismatch_side_by_side`` image, plus its type and image ID.

Every selected sample gets its own dynamically sized PDF page.  Source raster
pixels are embedded without downsampling or JPEG re-encoding; page dimensions
are derived from the source image dimensions.

Run from the repository root::

    python tools/auxdet_tools/07_create_mismatch_report_pdf.py \
        --voc-root data_root/VOC2007

The default output is ``data_root/VOC2007/mismatch_report.pdf``.
"""

from __future__ import annotations

import argparse
import csv
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--voc-root",
        type=Path,
        default=Path("data_root/VOC2007"),
        help="VOC2007 directory containing PNGImages and Annotations.",
    )
    parser.add_argument(
        "--png-dir",
        type=Path,
        default=None,
        help="Original image directory; defaults to VOC_ROOT/PNGImages.",
    )
    parser.add_argument(
        "--side-by-side-dir",
        type=Path,
        default=None,
        help="Output directory of script 05; auto-detected when omitted.",
    )
    parser.add_argument(
        "--ann-dir",
        type=Path,
        default=None,
        help="XML directory; defaults to VOC_ROOT/Annotations.",
    )
    parser.add_argument(
        "--mismatch-csv",
        type=Path,
        default=None,
        help="Mismatch CSV from script 04; defaults to VOC_ROOT/mismatches/mismatch.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PDF path; defaults to VOC_ROOT/mismatch_report.pdf.",
    )
    parser.add_argument(
        "--per-type",
        type=int,
        default=10,
        help="Maximum number of images selected for each view/band type.",
    )
    parser.add_argument(
        "--image-dpi",
        type=float,
        default=72.0,
        help=(
            "Physical display DPI only; source pixels are never resampled. "
            "72 means one source pixel equals one PDF point."
        ),
    )
    parser.add_argument(
        "--title", default="AuxDet Mismatch Report", help="PDF report title."
    )
    return parser.parse_args()


def image_index(directory: Path) -> Dict[str, Path]:
    """Index common image files by lowercase filename stem."""
    result: Dict[str, Path] = {}
    if not directory.is_dir():
        return result
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            result.setdefault(path.stem.lower(), path)
    return result


def xml_index(directory: Path) -> Dict[str, Path]:
    if not directory.is_dir():
        return {}
    return {path.stem.lower(): path for path in directory.rglob("*.xml")}


def xml_text(root: ET.Element, name: str, default: str = "Unknown") -> str:
    node = root.find(name)
    if node is None or node.text is None or not node.text.strip():
        return default
    return node.text.strip()


def safe_int(value: object) -> Optional[int]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def read_csv_rows(csv_path: Path) -> Dict[str, dict]:
    if not csv_path.is_file():
        warnings.warn(
            f"Mismatch CSV not found: {csv_path}. "
            "All available images will have equal selection priority."
        )
        return {}
    rows: Dict[str, dict] = {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            image_id = (row.get("image_id") or "").strip()
            if image_id:
                rows[Path(image_id).stem.lower()] = row
    return rows


def detect_side_by_side_dir(voc_root: Path) -> Path:
    candidates = (
        voc_root / "mismatch_side_by_side",
        voc_root / "mismatchsidebyside",
        voc_root / "mismatch_sidebyside",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def build_records(
    png_dir: Path,
    side_dir: Path,
    ann_dir: Path,
    mismatch_csv: Path,
) -> Tuple[List[dict], int]:
    originals = image_index(png_dir)
    side_images = image_index(side_dir)
    annotations = xml_index(ann_dir)
    csv_rows = read_csv_rows(mismatch_csv)
    records: List[dict] = []
    skipped = 0

    # The side-by-side directory defines the set of mismatch images considered.
    for stem, side_path in sorted(side_images.items()):
        original_path = originals.get(stem)
        if original_path is None:
            warnings.warn(f"Original PNG not found for {side_path.name}; skipped.")
            skipped += 1
            continue

        row = csv_rows.get(stem, {})
        view = (row.get("view") or "").strip()
        band_type = (row.get("band_type") or "").strip()
        if not view or not band_type:
            xml_path = annotations.get(stem)
            if xml_path is not None:
                try:
                    root = ET.parse(xml_path).getroot()
                    view = view or xml_text(root, "view")
                    band_type = band_type or xml_text(root, "band_type")
                except ET.ParseError as exc:
                    warnings.warn(f"Invalid XML {xml_path}: {exc}")
        view = view or "Unknown"
        band_type = band_type or "Unknown"

        gt_count = safe_int(row.get("gt_count"))
        pred_count = safe_int(row.get("prediction_count"))
        matched_count = safe_int(row.get("matched_count"))
        if gt_count is not None and pred_count is not None and matched_count is not None:
            unmatched_gt = max(0, gt_count - matched_count)
            unmatched_pred = max(0, pred_count - matched_count)
            unmatched_total = unmatched_gt + unmatched_pred
            mismatch_ratio = unmatched_total / max(1, gt_count + pred_count)
            count_gap = abs(gt_count - pred_count)
        else:
            unmatched_total = -1
            mismatch_ratio = -1.0
            count_gap = -1

        records.append(
            {
                "image_id": side_path.stem,
                "view": view,
                "band_type": band_type,
                "original": original_path,
                "side": side_path,
                "gt_count": gt_count,
                "pred_count": pred_count,
                "matched_count": matched_count,
                "unmatched_total": unmatched_total,
                "mismatch_ratio": mismatch_ratio,
                "count_gap": count_gap,
            }
        )
    return records, skipped


def select_by_type(records: List[dict], per_type: int) -> List[Tuple[Tuple[str, str], List[dict], int]]:
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for record in records:
        groups[(record["view"], record["band_type"])].append(record)

    selected = []
    for key in sorted(groups):
        candidates = groups[key]
        # Larger unmatched count/ratio/count gap means a greater GT-prediction
        # discrepancy.  Image ID is the deterministic final tie-breaker.
        candidates.sort(
            key=lambda item: (
                -item["unmatched_total"],
                -item["mismatch_ratio"],
                -item["count_gap"],
                item["image_id"],
            )
        )
        selected.append((key, candidates[:per_type], len(candidates)))
    return selected


def raster_size(path: Path, points_per_pixel: float) -> Tuple[float, float]:
    """Return display size without changing the embedded raster dimensions."""
    from reportlab.lib.utils import ImageReader

    width, height = ImageReader(str(path)).getSize()
    return width * points_per_pixel, height * points_per_pixel


def create_pdf(
    groups: List[Tuple[Tuple[str, str], List[dict], int]],
    output: Path,
    report_title: str,
    image_dpi: float,
) -> int:
    try:
        from reportlab.lib.colors import HexColor, white
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise RuntimeError("This script requires reportlab: pip install reportlab") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    # Page content streams are left uncompressed and PNG pixels use lossless
    # Flate encoding. drawImage embeds the original raster instead of making a
    # resized intermediate image.
    pdf = canvas.Canvas(str(output), pagesize=(100, 100), pageCompression=0)
    pdf.setTitle(report_title)
    points_per_pixel = 72.0 / image_dpi
    margin = 32.0
    image_gap = 18.0
    top_area = 78.0
    caption_area = 20.0
    footer_area = 20.0
    minimum_page_width = 760.0
    maximum_pdf_dimension = 14000.0
    page_number = 0

    for (view, band_type), samples, available_count in groups:
        for sample_index, sample in enumerate(samples, start=1):
            original_width, original_height = raster_size(
                sample["original"], points_per_pixel
            )
            side_width, side_height = raster_size(sample["side"], points_per_pixel)
            image_height = max(original_height, side_height)
            content_width = original_width + image_gap + side_width
            page_width = max(minimum_page_width, content_width + 2 * margin)
            page_height = margin + top_area + image_height + caption_area + footer_area
            if max(page_width, page_height) > maximum_pdf_dimension:
                raise ValueError(
                    f"Page for {sample['image_id']} exceeds the PDF size limit. "
                    "Increase --image-dpi; this changes physical display size "
                    "without removing source pixels."
                )

            page_number += 1
            pdf.setPageSize((page_width, page_height))
            pdf.setFillColor(white)
            pdf.rect(0, 0, page_width, page_height, fill=1, stroke=0)
            pdf.setFillColor(HexColor("#17324D"))
            pdf.setFont("Helvetica-Bold", 17)
            pdf.drawString(margin, page_height - margin, report_title)
            pdf.setFont("Helvetica-Bold", 12)
            pdf.drawRightString(
                page_width - margin,
                page_height - margin,
                f"Type: {view} / {band_type}",
            )
            pdf.setFillColor(HexColor("#526372"))
            pdf.setFont("Helvetica", 9)
            pdf.drawString(
                margin,
                page_height - margin - 20,
                f"Image ID: {sample['image_id']}    Selected {sample_index}/{len(samples)} "
                f"from {available_count} available images",
            )
            if sample["gt_count"] is not None:
                details = (
                    f"GT={sample['gt_count']}   Pred={sample['pred_count']}   "
                    f"Matched={sample['matched_count']}   "
                    f"Unmatched={sample['unmatched_total']}"
                )
                pdf.setFillColor(HexColor("#B33A3A"))
                pdf.setFont("Helvetica-Bold", 9)
                pdf.drawRightString(
                    page_width - margin, page_height - margin - 20, details
                )

            image_bottom = footer_area + caption_area
            content_left = (page_width - content_width) / 2
            original_x = content_left
            side_x = original_x + original_width + image_gap
            for path, draw_x, draw_width, draw_height, caption in (
                (
                    sample["original"],
                    original_x,
                    original_width,
                    original_height,
                    "Original PNG - native pixels",
                ),
                (
                    sample["side"],
                    side_x,
                    side_width,
                    side_height,
                    "Mismatch side-by-side - native pixels",
                ),
            ):
                draw_y = image_bottom + (image_height - draw_height) / 2
                pdf.drawImage(
                    ImageReader(str(path)),
                    draw_x,
                    draw_y,
                    width=draw_width,
                    height=draw_height,
                    preserveAspectRatio=True,
                    mask="auto",
                )
                pdf.setStrokeColor(HexColor("#8797A5"))
                pdf.rect(draw_x, draw_y, draw_width, draw_height, fill=0, stroke=1)
                pdf.setFillColor(HexColor("#536575"))
                pdf.setFont("Helvetica", 8)
                pdf.drawCentredString(
                    draw_x + draw_width / 2, footer_area + 7, caption
                )
            pdf.setFillColor(HexColor("#667788"))
            pdf.setFont("Helvetica", 8)
            pdf.drawRightString(page_width - margin, 10, f"Page {page_number}")
            pdf.showPage()

    pdf.save()
    return page_number


def main() -> None:
    args = parse_args()
    if args.per_type < 1:
        raise ValueError("--per-type must be at least 1")
    if args.image_dpi <= 0:
        raise ValueError("--image-dpi must be greater than 0")
    voc_root = args.voc_root.resolve()
    png_dir = (args.png_dir or voc_root / "PNGImages").resolve()
    ann_dir = (args.ann_dir or voc_root / "Annotations").resolve()
    side_dir = (
        args.side_by_side_dir.resolve()
        if args.side_by_side_dir is not None
        else detect_side_by_side_dir(voc_root).resolve()
    )
    mismatch_csv = (
        args.mismatch_csv or voc_root / "mismatches" / "mismatch.csv"
    ).resolve()
    output = (args.output or voc_root / "mismatch_report.pdf").resolve()

    for directory, name in (
        (png_dir, "original PNG directory"),
        (side_dir, "mismatch side-by-side directory"),
        (ann_dir, "annotation directory"),
    ):
        if not directory.is_dir():
            raise FileNotFoundError(f"{name} does not exist: {directory}")

    records, skipped = build_records(png_dir, side_dir, ann_dir, mismatch_csv)
    groups = select_by_type(records, args.per_type)
    selected_count = sum(len(samples) for _, samples, _ in groups)
    if selected_count == 0:
        raise RuntimeError(
            "No matching image pairs were found in PNGImages and mismatch_side_by_side."
        )
    pages = create_pdf(
        groups,
        output,
        args.title,
        args.image_dpi,
    )

    print(f"Types included: {len(groups)}")
    print(f"Images included: {selected_count}")
    print(f"Skipped: {skipped}")
    print(f"PDF pages: {pages}")
    print(f"PDF: {output}")
    for (view, band), samples, available in groups:
        print(f"  {view}/{band}: selected {len(samples)} of {available}")


if __name__ == "__main__":
    main()
