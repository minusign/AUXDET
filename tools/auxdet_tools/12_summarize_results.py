#!/usr/bin/env python3
"""Validate experiment provenance and export paper tables/figures from results."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import statistics

from mmengine.config import Config

from eval_common import digest, file_info, read_csv, read_json, read_manifest, write_csv, write_json


def number(value):
    return None if value in (None, "") else float(value)


def training_signature(cfg):
    # Architecture and random seed are intentionally excluded. These settings
    # must be common across all ablation variants and seeds.
    return digest({key: cfg.get(key) for key in (
        "train_cfg", "optim_wrapper", "param_scheduler", "auto_scale_lr",
        "train_dataloader", "val_dataloader", "env_cfg", "default_hooks")})


def load_run(run):
    directory = Path(run["eval_dir"])
    metadata = read_json(directory / "evaluation_metadata.json")
    parity = read_json(directory / "parity_report.json")
    overall = read_csv(directory / "overall_metrics.csv")
    domains = read_csv(directory / "domain_metrics.csv")
    if len(overall) != 1 or overall[0]["scope"] != "overall":
        raise ValueError("Overall AP must come from one overall row, never averaged domain AP")
    if int(overall[0]["gt_count"]) == 0:
        raise ValueError("A formal detection summary requires at least one ordinary GT")
    for row in [*overall, *domains]:
        if row["evaluation_id"] != metadata["evaluation_id"] or row["model"] != run["model"]:
            raise ValueError(f"{run['experiment']}: mixed evaluation IDs or model names")
    if parity["status"] == "fail" or metadata["parity_status"] != parity["status"]:
        raise ValueError(f"{run['experiment']}: failed/inconsistent parity report")
    if metadata.get("checkpoint") is None:
        raise ValueError("Formal summaries require 06 --checkpoint to record checkpoint provenance")
    for field, source in (("checkpoint", "checkpoint"), ("predictions", "predictions"),
                          ("config", "config"), ("split", "id_list")):
        # Local files are required: verify content, not merely a matching path.
        if file_info(run[field])["sha256"] != metadata[source]["sha256"]:
            raise ValueError(f"{run['experiment']}: {field} differs from evaluated source")
    if not Path(run["training_log"]).is_file():
        raise FileNotFoundError(f"Training log is missing: {run['training_log']}")
    cfg = Config.fromfile(run["config"])
    if digest(cfg.to_dict()) != metadata["resolved_config_sha256"]:
        raise ValueError(f"{run['experiment']}: inherited config changed after evaluation")
    configured_seed = cfg.get("randomness", {}).get("seed")
    if configured_seed is not None and run["seed"] != configured_seed:
        raise ValueError(f"{run['experiment']}: manifest/config seed mismatch")
    for key in ("image_count", "gt_count", "tp", "fp", "fn"):
        if sum(int(row[key]) for row in domains) != int(overall[0][key]):
            raise ValueError(f"{run['experiment']}: domain sums disagree for {key}")
    info = dict(run, metadata=metadata, overall=overall[0], domains=domains,
                training_signature=training_signature(cfg),
                architecture_signature=digest(cfg.model),
                log_source=file_info(run["training_log"]), size=[], complexity=[])
    if run["size_dir"]:
        folder = Path(run["size_dir"])
        size_meta = read_json(folder / "size_metadata.json")
        size_rows = read_csv(folder / "target_size_metrics.csv")
        if size_meta["evaluation_id"] != metadata["evaluation_id"]:
            raise ValueError(f"{run['experiment']}: size analysis uses different predictions/protocol")
        for row in size_rows:
            if row["evaluation_id"] != metadata["evaluation_id"] or row["bins_id"] != size_meta["bins_id"]:
                raise ValueError("Size rows mix evaluation/bin identities")
        info["size"], info["bins_id"] = size_rows, size_meta["bins_id"]
    if run["profile_dir"]:
        folder = Path(run["profile_dir"])
        prof_meta = read_json(folder / "profiling_metadata.json")
        prof_rows = [row for row in read_csv(folder / "complexity_metrics.csv")
                     if row["experiment"] == run["experiment"]]
        if len(prof_rows) != 1 or prof_rows[0]["status"] != "pass":
            raise ValueError(f"{run['experiment']}: missing/failed complexity result")
        detail = prof_meta["experiments"][run["experiment"]]
        if detail["checkpoint"]["sha256"] != metadata["checkpoint"]["sha256"] or \
                detail["config"]["sha256"] != metadata["config"]["sha256"]:
            raise ValueError("Complexity measurements use a different checkpoint/config")
        if prof_rows[0]["protocol_sha256"] != prof_meta["protocol_sha256"]:
            raise ValueError("Profiling protocol identity mismatch")
        info["complexity"] = prof_rows
    return info


def validate_runs(runs, baseline="B0", min_seeds=3):
    if min_seeds < 2:
        raise ValueError("At least two seeds are needed for a sample standard deviation")
    for key in ("training_signature",):
        if len({r[key] for r in runs}) != 1:
            raise ValueError("Training configurations differ (optimizer/schedule/batch/data/hooks)")
    for key in ("dataset_sha256", "protocol_sha256"):
        if len({r["metadata"][key] for r in runs}) != 1:
            raise ValueError(f"Evaluations use different {key}")
    if len({digest(r["metadata"]["official_source"]) for r in runs}) != 1:
        raise ValueError("Runs were evaluated using different VOC evaluator source revisions")
    for key in ("training_protocol", "evaluation_protocol"):
        if len({r[key] for r in runs}) != 1:
            raise ValueError(f"Manifest contains different {key} labels")
    by_model = defaultdict(list)
    by_phase = defaultdict(list)
    for run in runs:
        by_model[run["model"]].append(run)
        by_phase[run["phase"]].append(run)
    for model, subset in by_model.items():
        if len({r["architecture_signature"] for r in subset}) != 1:
            raise ValueError(f"{model}: architecture changed between repeated runs")
    for phase, subset in by_phase.items():
        models = defaultdict(list)
        for run in subset:
            models[run["model"]].append(run)
        if baseline not in models:
            raise ValueError(f"Missing {baseline} in phase {phase}")
        if phase == "screening":
            if any(len(values) != 1 for values in models.values()):
                raise ValueError("Screening is one run per model; label repetitions as repeat")
        else:
            seeds = None
            for model, values in models.items():
                current = [r["seed"] for r in values]
                if any(not isinstance(seed, int) for seed in current):
                    raise ValueError("Repeated runs need explicit integer seeds")
                if len(current) != len(set(current)) or len(current) < min_seeds:
                    raise ValueError(f"{model}: duplicated seeds or fewer than {min_seeds} repetitions")
                if len({r["metadata"]["checkpoint"]["sha256"] for r in values}) != len(values):
                    raise ValueError(f"{model}: repeated runs reuse the same checkpoint")
                if seeds is not None and set(current) != seeds:
                    raise ValueError("Repeated models must use the same paired seed set")
                seeds = set(current)
    size_ids = {r["bins_id"] for r in runs if r["size"]}
    if len(size_ids) > 1:
        raise ValueError("Size results use different frozen bin files")
    protocols = {row["protocol_sha256"] for r in runs for row in r["complexity"]}
    if len(protocols) > 1:
        raise ValueError("Complexity results use different hardware/measurement protocols")


def aggregate(rows, group_keys, metrics):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    result = []
    for key, items in sorted(grouped.items()):
        output = dict(zip(group_keys, key))
        output["n"] = len(items)
        for metric in metrics:
            values = [number(row.get(metric)) for row in items]
            values = [v for v in values if v is not None]
            output[metric + "_mean"] = statistics.mean(values) if values else None
            output[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else None
        result.append(output)
    return result


def tables(runs, baseline):
    overall, domains, size, complexity, paired = [], [], [], [], []
    lookup = {(r["phase"], r["model"], r["seed"] if r["phase"] == "repeat" else ""): r for r in runs}
    for run in runs:
        identity = dict(experiment=run["experiment"], model=run["model"],
                        phase=run["phase"], seed=run["seed"], epoch=run["epoch"],
                        checkpoint_sha256=run["metadata"]["checkpoint"]["sha256"],
                        parity_status=run["metadata"]["parity_status"])
        base = lookup[(run["phase"], baseline, run["seed"] if run["phase"] == "repeat" else "")]
        overall.append({**run["overall"], **identity})
        paired.append(dict(identity, delta_AP50=number(run["overall"]["AP50"])-number(base["overall"]["AP50"]),
                           delta_Recall=None if number(run["overall"]["Recall"]) is None else
                           number(run["overall"]["Recall"])-number(base["overall"]["Recall"])))
        domain_base = {(r["view"], r["band_type"]): r for r in base["domains"]}
        for row in run["domains"]:
            reference = domain_base[(row["view"], row["band_type"])]
            rec, ref_rec = number(row["Recall"]), number(reference["Recall"])
            domains.append(dict(row, **{k: v for k, v in identity.items() if k not in row},
                                delta_AP50=number(row["AP50"])-number(reference["AP50"]),
                                delta_Recall=rec-ref_rec if rec is not None and ref_rec is not None else None))
        base_size = {(r["scope"], r["view"], r["band_type"], r["size_bin"]): r for r in base["size"]}
        for row in run["size"]:
            key = tuple(row[k] for k in ("scope", "view", "band_type", "size_bin"))
            if key not in base_size:
                raise ValueError("Size comparison requires corresponding baseline size results")
            current, reference = number(row["Recall"]), number(base_size[key]["Recall"])
            size.append({**row, **identity, "delta_Recall": current-reference
                         if current is not None and reference is not None else None})
        complexity.extend({**row, **identity} for row in run["complexity"])
    output = dict(overall_runs=overall, domain_runs=domains, size_runs=size,
                  complexity_runs=complexity, paired_seed_deltas=paired)
    output["overall_summary"] = aggregate(overall, ["phase", "model"], ["AP50", "Recall"])
    output["domain_summary"] = aggregate(domains, ["phase", "model", "view", "band_type"],
                                         ["AP50", "Recall", "delta_AP50", "delta_Recall"])
    output["size_summary"] = aggregate(size, ["phase", "model", "scope", "view", "band_type", "size_bin"],
                                       ["Recall", "delta_Recall"])
    for destination, source, keys in (
            ("overall_summary", overall, ("phase", "model")),
            ("domain_summary", domains, ("phase", "model", "view", "band_type")),
            ("size_summary", size, ("phase", "model", "scope", "view", "band_type", "size_bin"))):
        for row in output[destination]:
            selected = [item for item in source if all(item[k] == row[k] for k in keys)]
            for column in ("image_count", "gt_count"):
                counts = {int(item[column]) for item in selected}
                if len(counts) != 1:
                    raise ValueError("Repeated runs have different image/GT group sizes")
                row[column] = counts.pop()
    output["paired_delta_summary"] = aggregate(paired, ["phase", "model"], ["delta_AP50", "delta_Recall"])
    output["complexity_summary"] = aggregate(complexity, ["phase", "model"],
                                             ["params", "trainable_params", "latency_mean_ms",
                                              "peak_allocated_mb", "flops"])
    for row in output["complexity_summary"]:
        row["flops_coverage"] = ";".join(sorted({r["flops_status"] for r in complexity
                                                if (r["phase"], r["model"]) == (row["phase"], row["model"])}))
    return output


def make_figures(output, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    folder.mkdir(parents=True, exist_ok=True)
    for phase in sorted({r["phase"] for r in output["overall_summary"]}):
        rows = [r for r in output["overall_summary"] if r["phase"] == phase]
        names = [r["model"] for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, metric in zip(axes, ("AP50", "Recall")):
            means = [100*r[metric+"_mean"] for r in rows]
            ax.bar(names, means)
            for index, row in enumerate(rows):
                if row[metric+"_std"] is not None:
                    ax.errorbar(index, means[index], yerr=100*row[metric+"_std"],
                                color="black", capsize=4, fmt="none")
            ax.set(ylabel=f"{metric} (%)", title=f"{phase}: {metric}", ylim=(0, 105))
        for ext in ("png", "pdf"):
            fig.savefig(folder / f"{phase}_overall.{ext}", dpi=300)
        plt.close(fig)
        rows = [r for r in output["domain_summary"] if r["phase"] == phase]
        domains = sorted({(r["view"], r["band_type"]) for r in rows})
        lookup = {(r["model"], r["view"], r["band_type"]): r["delta_AP50_mean"] for r in rows}
        matrix = np.array([[100*lookup[(model, *domain)] for domain in domains] for model in names])
        fig, ax = plt.subplots(figsize=(max(6, len(domains)*1.3), max(3, len(names)*.6)), constrained_layout=True)
        bound = max(float(np.max(np.abs(matrix))), .01)
        plot = ax.imshow(matrix, cmap="RdBu", vmin=-bound, vmax=bound, aspect="auto")
        ax.set_xticks(range(len(domains)), ["/".join(d) for d in domains], rotation=25, ha="right")
        ax.set_yticks(range(len(names)), names)
        for y in range(len(names)):
            for x in range(len(domains)):
                red, green, blue, _ = plot.cmap(plot.norm(matrix[y, x]))
                luminance = .2126*red + .7152*green + .0722*blue
                ax.text(x, y, f"{matrix[y,x]:+.2f}", ha="center", va="center",
                        color="white" if luminance < .5 else "black")
        fig.colorbar(plot, ax=ax, label="AP50 change (percentage points)")
        for ext in ("png", "pdf"):
            fig.savefig(folder / f"{phase}_domain_delta.{ext}", dpi=300)
        plt.close(fig)
        for scope in ("overall", "domain"):
            rows = [r for r in output["size_summary"] if r["phase"] == phase and r["scope"] == scope]
            if not rows:
                continue
            fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
            for model in names:
                values = sorted([r for r in rows if r["model"] == model], key=lambda r: r["size_bin"])
                ax.plot([r["size_bin"] for r in values],
                        [100*r["Recall_mean"] if r["Recall_mean"] is not None else np.nan for r in values],
                        marker="o", label=model)
            ax.set(ylabel="Recall (%)", xlabel="Frozen size bin", ylim=(0, 105),
                   title="All images" if scope == "overall" else "Space/NIR")
            ax.legend()
            for ext in ("png", "pdf"):
                fig.savefig(folder / f"{phase}_size_{scope}.{ext}", dpi=300)
            plt.close(fig)
        cost = [r for r in output["complexity_summary"] if r["phase"] == phase]
        if cost:
            fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
            for row in cost:
                x, y = row["params_mean"]/1e6, row["latency_mean_ms_mean"]
                ax.scatter(x, y)
                ax.annotate(row["model"], (x, y))
            ax.set(xlabel="Parameters (millions)", ylabel="Latency per batch (ms)")
            for ext in ("png", "pdf"):
                fig.savefig(folder / f"{phase}_complexity.{ext}", dpi=300)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+")
    parser.add_argument("--baseline", default="B0")
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    runs = [load_run(row) for row in read_manifest(args.manifest, args.experiments)]
    validate_runs(runs, args.baseline, args.min_seeds)
    output = tables(runs, args.baseline)
    sections = ["# Ablation results", "",
                "AP50/Recall are fractions. Standard deviations use ddof=1; blank means a single run.",
                "Screening and repeated runs are separate. FLOPs only count supported traced operations; see coverage.",
                "Size image_count is the full group denominator; images_with_gt_in_bin is not additive across bins.", ""]
    for name, rows in output.items():
        if not rows:
            continue
        write_csv(args.output_dir / f"{name}.csv", rows)
        if name.endswith("_summary"):
            columns = list(rows[0])
            sections += [f"## {name}", "", "| " + " | ".join(columns) + " |",
                         "| " + " | ".join("---" for _ in columns) + " |"]
            for row in rows:
                sections.append("| " + " | ".join(
                    "" if row[key] is None else f"{row[key]:.6f}" if isinstance(row[key], float)
                    else str(row[key]) for key in columns) + " |")
            sections.append("")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tables.md").write_text("\n".join(sections) + "\n")
    write_json(args.output_dir / "summary_metadata.json", dict(
        manifest=file_info(args.manifest), baseline=args.baseline, min_seeds=args.min_seeds,
        runs=[dict(experiment=r["experiment"], seed=r["seed"], epoch=r["epoch"],
                   evaluation_id=r["metadata"]["evaluation_id"], checkpoint=r["metadata"]["checkpoint"],
                   training_log=r["log_source"], parity=r["metadata"]["parity_status"])
              for r in runs]))
    if not args.no_plots:
        make_figures(output, args.output_dir / "figures")
    print(f"Paper tables and figures: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
