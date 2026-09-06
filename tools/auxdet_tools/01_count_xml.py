#!/usr/bin/env python3
"""Count image-level ``view``/``band_type`` combinations in XML files.

Each XML file contributes one image count.  A pair such as ``Air/LWIR`` and
``Air/NIR`` is counted as two different groups.  The output CSV also contains
the number of annotated target boxes, which is useful when "quantity" means
objects rather than images.

Example:
    python tools/auxdet_tools/01_count_xml.py \
        --ann-dir data_root/VOC2007/Annotations
"""

from __future__ import annotations

import argparse
import csv
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def value(root: ET.Element, name: str) -> str:
    node = root.find(name)
    if node is None or node.text is None or not node.text.strip():
        return "Unknown"
    return node.text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ann-dir",
        type=Path,
        default=Path("data_root/VOC2007/Annotations"),
        help="Directory containing XML annotations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_root/VOC2007/view_band_type_counts.csv"),
        help="CSV path for the grouped report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ann_dir = args.ann_dir.resolve()
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory does not exist: {ann_dir}")

    xml_files = sorted(ann_dir.rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found under: {ann_dir}")

    groups: defaultdict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"xml_count": 0, "target_count": 0}
    )
    for xml_path in xml_files:
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            warnings.warn(f"Invalid XML skipped: {xml_path} ({exc})")
            continue
        key = (value(root, "view"), value(root, "band_type"))
        groups[key]["xml_count"] += 1
        groups[key]["target_count"] += len(root.findall("object"))

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["view", "band_type", "xml_count", "target_count"])
        for (view, band_type), counts in sorted(groups.items()):
            writer.writerow(
                [view, band_type, counts["xml_count"], counts["target_count"]]
            )

    print(f"XML files counted: {sum(c['xml_count'] for c in groups.values())}")
    print(f"Unique view/band_type groups: {len(groups)}")
    print(f"Report: {output}")
    print("\nview\tband_type\txml_count\ttarget_count")
    for (view, band_type), counts in sorted(groups.items()):
        print(f"{view}\t{band_type}\t{counts['xml_count']}\t{counts['target_count']}")


if __name__ == "__main__":
    main()
