#!/usr/bin/env python3
"""Compare two aligned official prediction caches and export interpretable cases."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from eval_common import (curves, groups, load_evaluation, match_image, read_csv,
                         require_aligned, write_csv, write_json)


def compare(first, second, first_name="B0", second_name="candidate", localization_iou=.1):
    require_aligned(first, second)
    threshold = first.metadata["protocol"]["iou_thr"]
    gt_rows, image_rows, prediction_rows, cases = [], [], [], []
    for a, b in zip(first.images, second.images):
        am, ah = match_image(a, threshold, localization_iou)
        bm, bh = match_image(b, threshold, localization_iou)
        counts = Counter()
        for index in range(len(a.gt_boxes)):
            if a.gt_ignore[index]:
                state = "ignored"
            elif ah[index] >= 0 and bh[index] >= 0:
                state = "both_hit"
            elif ah[index] >= 0:
                state = "lost"
            elif bh[index] >= 0:
                state = "gained"
            else:
                state = "both_missed"
            counts[state] += 1
            gt_rows.append(dict(image_id=a.image_id, view=a.view, band_type=a.band_type,
                                gt_index=index, label=int(a.gt_labels[index]), state=state,
                                baseline_prediction_index=int(ah[index]),
                                candidate_prediction_index=int(bh[index]),
                                **dict(zip(("x1", "y1", "x2", "y2"), a.gt_boxes[index].tolist()))))
        row = dict(image_id=a.image_id, view=a.view, band_type=a.band_type,
                   gt_count=int((~a.gt_ignore).sum()),
                   **{key: counts[key] for key in ("both_hit", "lost", "gained", "both_missed")})
        for name, matches in (("baseline", am), ("candidate", bm)):
            counter = Counter(m["status"] for m in matches)
            errors = Counter(m["error_type"] for m in matches)
            for key in ("TP", "FP", "ignored"):
                row[f"{name}_{key}"] = counter[key]
            for key in ("duplicate", "background", "localization"):
                row[f"{name}_{key}"] = errors[key]
            prediction_rows.extend(dict(model=first_name if name == "baseline" else second_name, **m)
                                   for m in matches)
        row["delta_TP"] = row["candidate_TP"]-row["baseline_TP"]
        row["delta_FP"] = row["candidate_FP"]-row["baseline_FP"]
        image_rows.append(row)
        categories = [key for key in ("lost", "gained", "both_missed") if counts[key]]
        if row["delta_FP"]:
            categories.append("fp_increase" if row["delta_FP"] > 0 else "fp_decrease")
        if categories:
            cases.append(dict(image_id=a.image_id, categories=";".join(categories),
                              view=a.view, band_type=a.band_type, lost=counts["lost"],
                              gained=counts["gained"], both_missed=counts["both_missed"],
                              delta_FP=row["delta_FP"], image_path=str(a.image_path)))
    return image_rows, gt_rows, prediction_rows, cases


def render_cases(first, second, ids, output, first_name, second_name, localization_iou):
    from PIL import Image, ImageDraw
    left, right = ({i.image_id: i for i in data.images} for data in (first, second))
    output.mkdir(parents=True, exist_ok=True)
    for image_id in ids:
        if image_id not in left:
            raise ValueError(f"Case image is outside the evaluated split: {image_id}")
        a, b = left[image_id], right[image_id]
        original = Image.open(a.image_path).convert("RGB")
        width, height = original.size
        canvas = Image.new("RGB", (width * 3, height + 32), "white")
        for column, (title, image) in enumerate((("Image + GT", a), (first_name, a), (second_name, b))):
            panel = original.copy()
            draw = ImageDraw.Draw(panel)
            if column == 0:
                for box, ignored in zip(image.gt_boxes, image.gt_ignore):
                    draw.rectangle(box.tolist(), outline="gray" if ignored else "lime", width=2)
            else:
                matches = match_image(image, first.metadata["protocol"]["iou_thr"], localization_iou)[0]
                for box, match in zip(image.boxes, matches):
                    color = {"TP": "lime", "FP": "red", "ignored": "gray"}[match["status"]]
                    draw.rectangle(box.tolist(), outline=color, width=2)
                    draw.text((float(box[0]), max(0, float(box[1])-12)),
                              f"{match['status']} {match['score']:.2f}", fill=color)
            canvas.paste(panel, (column * width, 32))
            ImageDraw.Draw(canvas).text((column * width + 5, 8), title, fill="black")
        # Flatten IDs safely when datasets have nested identifiers.
        canvas.save(output / (image_id.replace("/", "__") + ".png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--baseline-name", default="B0")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--id-list", type=Path)
    parser.add_argument("--score-thr", type=float)
    parser.add_argument("--localization-iou", type=float, default=.1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-list", type=Path, help="CSV with image_id column for rendering")
    parser.add_argument("--render-max", type=int, default=0, help="0 disables rendering unless --case-list")
    args = parser.parse_args()
    if args.render_max < 0 or args.baseline_name == args.candidate_name:
        parser.error("Use distinct model names and nonnegative --render-max")
    kwargs = dict(data_root=args.data_root, id_list=args.id_list, score_thr=args.score_thr)
    first = load_evaluation(args.baseline_config, args.baseline_predictions, **kwargs)
    second = load_evaluation(args.candidate_config, args.candidate_predictions, **kwargs)
    if len(first.classes) != 1:
        parser.error("This comparison's PR/FPPI output currently supports the single Target class")
    results = compare(first, second, args.baseline_name, args.candidate_name, args.localization_iou)
    names = ("per_image_comparison.csv", "per_gt_comparison.csv",
             "per_prediction_matches.csv", "case_candidates.csv")
    empty_fields = (
        ["image_id", "view", "band_type", "gt_count"],
        ["image_id", "view", "band_type", "gt_index", "label", "state",
         "baseline_prediction_index", "candidate_prediction_index", "x1", "y1", "x2", "y2"],
        ["model", "image_id", "prediction_index", "filtered_prediction_index",
         "label", "score", "matched_gt_index", "best_gt_index", "IoU", "status", "error_type"],
        ["image_id", "categories", "view", "band_type", "lost", "gained", "both_missed", "delta_FP", "image_path"])
    for name, rows, fields in zip(names, results, empty_fields):
        write_csv(args.output_dir / name, rows, fields=None if rows else fields)
    curve_rows = []
    for name, data in ((args.baseline_name, first), (args.candidate_name, second)):
        for scope, view, band, images in groups(data):
            curve_rows.extend(dict(model=name, scope=scope, view=view, band_type=band,
                                   image_count=len(images), **row)
                              for row in curves(images, data.metadata["protocol"]["iou_thr"]))
    write_csv(args.output_dir / "pr_curve.csv", [
        {k: v for k, v in row.items() if k != "FPPI"} for row in curve_rows])
    write_csv(args.output_dir / "fppi_curve.csv", [
        {k: v for k, v in row.items() if k != "precision"} for row in curve_rows])
    if args.case_list:
        ids = list(dict.fromkeys(row["image_id"] for row in read_csv(args.case_list)))
        if args.render_max:
            ids = ids[:args.render_max]
    else:
        # Interleave degradations and improvements to retain both case types.
        candidates = results[3]
        pools = [[r["image_id"] for r in candidates if category in r["categories"].split(";")]
                 for category in ("lost", "gained", "both_missed", "fp_increase", "fp_decrease")]
        ids = list(dict.fromkeys(item for index in range(max(map(len, pools), default=0))
                                 for pool in pools if index < len(pool) for item in [pool[index]]))
        ids = ids[:args.render_max]
    if ids:
        render_cases(first, second, ids, args.output_dir / "cases",
                     args.baseline_name, args.candidate_name, args.localization_iou)
        write_csv(args.output_dir / "rendered_cases.csv", [{"image_id": v} for v in ids])
    write_json(args.output_dir / "comparison_metadata.json", dict(
        baseline=first.metadata, candidate=second.metadata,
        localization_iou=args.localization_iou, iou_thr=.5,
        error_rule="best same-class IoU: [localization_iou,0.5) is localization; below is background",
        curves_scope="final detections retained in cache after configured score filtering/NMS",
        rendered_images=ids))
    print(f"Comparison reports: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
