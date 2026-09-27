#!/usr/bin/env python3
"""Fit fixed input-size bins from training annotations, then evaluate GT recall."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from eval_common import (SIZE_CONVENTION, bin_index, digest, gt_sizes, load_dataset,
                         load_evaluation, match_image, read_json, require_aligned,
                         validate_bins, write_csv, write_json)


def distribution(data):
    rows = []
    for image in data.images:
        raw, scaled = gt_sizes(data, image)
        for index, (before, after) in enumerate(zip(raw, scaled)):
            rows.append(dict(image_id=image.image_id, gt_index=index,
                             view=image.view, band_type=image.band_type,
                             ignored=bool(image.gt_ignore[index]),
                             raw_width=float(before[0]), raw_height=float(before[1]),
                             raw_area=float(np.prod(before)), input_width=float(after[0]),
                             input_height=float(after[1]), input_area=float(np.prod(after)),
                             input_sqrt_area=float(np.sqrt(np.prod(after)))))
    return rows


def fit_bins(data, rows, quantiles, edges=None):
    if data.metadata["split"] != "train":
        raise ValueError("Bin fitting is allowed only on training annotations")
    values = [r["input_sqrt_area"] for r in rows if not r["ignored"]]
    if not values:
        raise ValueError("No ordinary training GT")
    if edges is None:
        if not quantiles or any(not 0 < q < 1 for q in quantiles) or quantiles != sorted(set(quantiles)):
            raise ValueError("Quantiles must be distinct, increasing and inside (0,1)")
        edges = [0.0, *np.quantile(values, quantiles).tolist()]
    bins = dict(measure="sqrt_area", convention=SIZE_CONVENTION,
                edges=[*edges, None], labels=[f"size_{i}" for i in range(len(edges))],
                interval="[lower, upper)", fitted_from="train",
                dataset_sha256=data.metadata["dataset_sha256"],
                id_list=data.metadata["id_list"],
                transform=data.config.test_dataloader.dataset.pipeline,
                fit_method="explicit_edges" if quantiles is None else "training_quantiles",
                quantiles=quantiles, gt_count=len(values))
    validate_bins(bins)
    bins["bins_id"] = digest(bins)
    return bins


def size_metrics(data, bins, model, baseline=None):
    validate_bins(bins)
    if bins.get("fitted_from") != "train":
        raise ValueError("Use frozen bins fitted on training annotations")
    if digest(bins.get("transform")) != digest(data.config.test_dataloader.dataset.pipeline):
        raise ValueError("Bin fitting and evaluation have different resize pipelines")
    if baseline is not None:
        require_aligned(data, baseline)
    threshold = data.metadata["protocol"]["iou_thr"]
    baseline_hits = {} if baseline is None else {
        i.image_id: match_image(i, threshold)[1] for i in baseline.images}
    buckets = [("overall", "ALL", "ALL", data.images),
               ("domain", "Space", "NIR", [i for i in data.images if (i.view, i.band_type) == ("Space", "NIR")])]
    rows = []
    for scope, view, band, images in buckets:
        counters = [dict(gt_count=0, tp=0, baseline_tp=0, containing_images=set()) for _ in bins["labels"]]
        total_hits = 0
        for image in images:
            hits = match_image(image, threshold)[1]
            total_hits += int((hits[~image.gt_ignore] >= 0).sum())
            _, sizes = gt_sizes(data, image)
            for index in np.where(~image.gt_ignore)[0]:
                bucket = counters[bin_index(float(np.sqrt(np.prod(sizes[index]))), bins)]
                bucket["gt_count"] += 1
                bucket["tp"] += int(hits[index] >= 0)
                bucket["containing_images"].add(image.image_id)
                if baseline is not None:
                    bucket["baseline_tp"] += int(baseline_hits[image.image_id][index] >= 0)
        assert sum(c["gt_count"] for c in counters) == sum(int((~i.gt_ignore).sum()) for i in images)
        assert sum(c["tp"] for c in counters) == total_hits
        for label, values in zip(bins["labels"], counters):
            gt, tp, btp = values["gt_count"], values["tp"], values["baseline_tp"]
            recall = tp/gt if gt else None
            base_recall = btp/gt if gt and baseline is not None else None
            rows.append(dict(model=model, evaluation_id=data.metadata["evaluation_id"],
                             bins_id=bins["bins_id"], scope=scope, view=view, band_type=band,
                             size_bin=label, image_count=len(images),
                             images_with_gt_in_bin=len(values["containing_images"]),
                             gt_count=gt, tp=tp, fn=gt-tp, Recall=recall,
                             baseline_Recall=base_recall,
                             delta_Recall=None if base_recall is None else recall-base_recall))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--model", default="model")
    parser.add_argument("--fit-bins", action="store_true")
    parser.add_argument("--quantiles", type=float, nargs="+", default=[1/3, 2/3])
    parser.add_argument("--edges", type=float, nargs="+")
    parser.add_argument("--bins", type=Path)
    parser.add_argument("--baseline-predictions", type=Path)
    parser.add_argument("--baseline-config", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--id-list", type=Path)
    parser.add_argument("--score-thr", type=float)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    kwargs = dict(data_root=args.data_root, id_list=args.id_list)
    if args.fit_bins:
        if args.predictions or args.baseline_predictions:
            parser.error("Fit bins from training annotations without prediction caches")
        data = load_dataset(args.config, split="train", **kwargs)
        rows = distribution(data)
        bins = fit_bins(data, rows, None if args.edges is not None else args.quantiles, args.edges)
    else:
        if args.bins is None or args.predictions is None:
            parser.error("Evaluation requires --predictions and frozen --bins")
        data = load_evaluation(args.config, args.predictions, score_thr=args.score_thr, **kwargs)
        bins = validate_bins(read_json(args.bins))
        original = dict(bins)
        identity = original.pop("bins_id", None)
        if digest(original) != identity:
            raise ValueError("Size-bin file changed after fitting")
        rows = distribution(data)
        baseline = None
        if args.baseline_predictions:
            baseline = load_evaluation(args.baseline_config or args.config,
                                       args.baseline_predictions, score_thr=args.score_thr, **kwargs)
        metrics = size_metrics(data, bins, args.model, baseline)
        write_csv(args.output_dir / "target_size_metrics.csv", metrics)
    write_csv(args.output_dir / "size_distribution.csv", rows,
              fields=["image_id", "gt_index", "view", "band_type", "ignored", "raw_width",
                      "raw_height", "raw_area", "input_width", "input_height", "input_area", "input_sqrt_area"])
    write_json(args.output_dir / "size_bins.json", bins)
    write_json(args.output_dir / "size_metadata.json", dict(
        data.metadata, model=args.model, bins_id=bins["bins_id"], convention=SIZE_CONVENTION))
    summaries = []
    for view, band in sorted({(r["view"], r["band_type"]) for r in rows}):
        subset = [r for r in rows if (r["view"], r["band_type"]) == (view, band) and not r["ignored"]]
        for measure in ("raw_width", "raw_height", "raw_area", "input_width", "input_height", "input_area"):
            values = [r[measure] for r in subset]
            if values:
                summaries.append(dict(view=view, band_type=band, measure=measure, gt_count=len(values),
                                      minimum=min(values), median=float(np.median(values)),
                                      p90=float(np.quantile(values, .9)), maximum=max(values)))
    write_csv(args.output_dir / "size_distribution_summary.csv", summaries)
    print(f"Size reports: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
