"""Shared offline evaluation for this repository's VSBWILDVOCDetDataset.

Official prediction dumps contain original-image coordinates and real img_id.
XML coordinates are intentionally NOT shifted by one (see voc_aux.py).
The repository's actual eval_map/tpfp_default code is used for official metrics.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import pickle
import sys
import types
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path
from typing import Optional

import numpy as np
from mmengine.config import Config


PROJECT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_info(path):
    path = Path(path).resolve()
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return {"path": str(path), "sha256": checksum.hexdigest(),
            "bytes": path.stat().st_size}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


@lru_cache(maxsize=1)
def official_voc():
    """Import the original CPU evaluator without importing CUDA model packages.

Only its Pool factory changes to a thread pool. Matching, sorting, AP and
coordinate calculations are the exact checked-out MMDetection implementation.
"""
    name = "_auxdet_official_voc"
    package = types.ModuleType(name)
    package.__path__ = [str(PROJECT / "mmdet/evaluation/functional")]
    sys.modules[name] = package
    module = importlib.import_module(name + ".mean_ap")
    module.Pool = ThreadPool
    return module


@dataclass
class ImageRecord:
    image_id: str
    image_path: Path
    width: int
    height: int
    view: str
    band_type: str
    gt_boxes: np.ndarray
    gt_labels: np.ndarray
    gt_ignore: np.ndarray
    boxes: np.ndarray = field(default_factory=lambda: np.empty((0, 4), np.float32))
    scores: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    labels: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    scale_factor: Optional[tuple] = None
    img_shape: Optional[tuple] = None
    prediction_indices: Optional[np.ndarray] = None

    def annotation(self):
        return dict(bboxes=self.gt_boxes[~self.gt_ignore],
                    labels=self.gt_labels[~self.gt_ignore],
                    bboxes_ignore=self.gt_boxes[self.gt_ignore],
                    labels_ignore=self.gt_labels[self.gt_ignore])

    def predictions(self, class_count):
        return [np.column_stack((self.boxes[self.labels == label],
                                 self.scores[self.labels == label])).astype(np.float32)
                for label in range(class_count)]


@dataclass
class EvaluationData:
    images: list
    classes: tuple
    config: Config
    metadata: dict


def _array(value, dtype):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _path(root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def test_resize_settings(cfg):
    """Reject transforms whose inverse cannot be established from this cache."""
    pipeline = cfg.test_dataloader.dataset.get("pipeline", [])
    allowed = {"LoadImageFromFile", "LoadAnnotations", "Resize", "PackDetInputs"}
    resize = [i for i, step in enumerate(pipeline) if step.get("type") == "Resize"]
    if len(resize) != 1 or any(step.get("type") not in allowed for step in pipeline):
        raise ValueError("Offline evaluation requires a simple, single-Resize test pipeline; "
                         "crop, flip and TTA are unsupported")
    if any(step.get("type") == "LoadAnnotations" for step in pipeline[:resize[0]]):
        raise ValueError("Test LoadAnnotations must follow Resize to preserve original-image GT")
    settings = pipeline[resize[0]]
    scale = settings.get("scale")
    if not isinstance(scale, (tuple, list)) or len(scale) != 2 or any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            for value in scale):
        raise ValueError("Test Resize must have one fixed positive (width, height) scale")
    return settings


def dataset_settings(config, split="test"):
    cfg = Config.fromfile(str(Path(config).resolve()))
    dataset = cfg[f"{split}_dataloader"]["dataset"]
    # Repeat/concat wrappers change index semantics. Require a direct dataset.
    if dataset.get("type") != "VSBWILDVOCDetDataset":
        raise ValueError("Use a direct VSBWILDVOCDetDataset configuration")
    if dataset.get("backend_args"):
        raise ValueError("Offline tools currently require local dataset files")
    evaluator = cfg.get("test_evaluator", {})
    if isinstance(evaluator, (tuple, list)):
        candidates = [v for v in evaluator if v.get("type") == "VOCMetric"]
        if len(candidates) != 1:
            raise ValueError("Exactly one VOCMetric is required")
        evaluator = candidates[0]
    if evaluator.get("type") != "VOCMetric" or evaluator.get("scale_ranges") is not None:
        raise ValueError("Use VOCMetric without scale_ranges for these tools")
    if evaluator.get("eval_mode", "11points") not in ("area", "11points"):
        raise ValueError("Unsupported VOC eval_mode")
    test_resize_settings(cfg)
    return cfg, dataset, evaluator


def parse_xml(path, classes):
    root = ET.parse(path).getroot()
    size = root.find("size")
    width, height = (int(size.findtext(key)) for key in ("width", "height"))
    if min(width, height) <= 0:
        raise ValueError(f"Invalid image dimensions in {path}")
    view, band = (root.findtext(key, "").strip() for key in ("view", "band_type"))
    if not view or not band or "Unknown" in (view, band):
        raise ValueError(f"Missing view/band_type in {path}")
    boxes, labels, ignored = [], [], []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in classes:
            continue
        node = obj.find("bndbox")
        if node is None:
            raise ValueError(f"Missing bbox in {path}")
        box = [int(float(node.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax")]
        if box[2] < box[0] or box[3] < box[1]:
            raise ValueError(f"Reversed bbox in {path}: {box}")
        difficult = int(obj.findtext("difficult", "0"))
        if difficult not in (0, 1):
            raise ValueError(f"Invalid difficult flag in {path}")
        boxes.append(box)
        labels.append(classes.index(name))
        ignored.append(bool(difficult))
    return width, height, view, band, np.asarray(boxes, np.float32).reshape(-1, 4), \
        np.asarray(labels, np.int64), np.asarray(ignored, bool)


def load_dataset(config, *, split="test", data_root=None, id_list=None,
                 ann_dir=None, image_dir=None):
    cfg, settings, evaluator = dataset_settings(config, split)
    root = Path(data_root or settings.get("data_root", ".")).resolve()
    prefix = _path(root, settings.get("data_prefix", {}).get("sub_data_root", ""))
    ids_path = Path(id_list).resolve() if id_list else _path(root, settings["ann_file"])
    annotations = Path(ann_dir).resolve() if ann_dir else prefix / settings.get("ann_subdir", "Annotations")
    images_dir = Path(image_dir).resolve() if image_dir else prefix / settings.get("img_subdir", "PNGImages")
    ids = [v.strip() for v in ids_path.read_text(encoding="utf-8-sig").splitlines() if v.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Split list is empty or contains duplicate image IDs")
    if any(Path(v).is_absolute() or ".." in Path(v).parts for v in ids):
        raise ValueError("Image IDs must be relative dataset identifiers")
    if split == "train" and id_list:
        configured_ids = set(_path(root, settings["ann_file"]).read_text(
            encoding="utf-8-sig").split())
        if not set(ids).issubset(configured_ids):
            raise ValueError("Training size-bin IDs must belong to the configured training split")
    classes = tuple(settings.get("metainfo", {}).get("classes", ("Target",)))
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("Invalid class list")
    records, annotation_hashes = [], []
    from PIL import Image
    for image_id in ids:
        xml_path = annotations / f"{image_id}.xml"
        image_path = images_dir / f"{image_id}.png"
        if not xml_path.is_file() or not image_path.is_file():
            raise FileNotFoundError(f"Missing image/XML for {image_id}")
        values = parse_xml(xml_path, classes)
        with Image.open(image_path) as image:
            if image.size != values[:2]:
                raise ValueError(f"Image/XML dimensions differ for {image_id}")
        records.append(ImageRecord(image_id, image_path, *values))
        annotation_hashes.append([image_id, file_info(xml_path)["sha256"]])
    protocol = dict(iou_thr=0.5, eval_mode=evaluator.get("eval_mode", "11points"),
                    legacy_coordinate=True, xml_coordinate_shift=0,
                    classes=list(classes), extra_score_thr=None,
                    test_cfg=cfg.model.get("test_cfg", {}),
                    test_pipeline=cfg.test_dataloader.dataset.get("pipeline", []),
                    data_preprocessor=cfg.model.get("data_preprocessor", {}))
    metadata = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
                    config=file_info(config), resolved_config_sha256=digest(cfg.to_dict()),
                    split=split, id_list=file_info(ids_path), image_count=len(records),
                    dataset_sha256=digest(annotation_hashes), classes=list(classes),
                    data_root=str(root), ann_dir=str(annotations), image_dir=str(images_dir),
                    image_ids=ids, protocol=protocol, predictions=None, checkpoint=None,
                    prediction_coordinates="original_image", official_source={
                        name: file_info(PROJECT / "mmdet/evaluation/functional" / name)["sha256"]
                        for name in ("mean_ap.py", "bbox_overlaps.py")})
    metadata["runtime"] = dict(python=sys.version.split()[0], numpy=np.__version__,
                               mmengine=importlib.metadata.version("mmengine"))
    return EvaluationData(records, classes, cfg, metadata)


def load_prediction_file(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = read_json(path)
    else:
        # Official DumpDetResults produces a trusted local pickle, usually
        # with CPU torch tensors. Do not use torch.load on this pickle.
        with path.open("rb") as stream:
            payload = pickle.load(stream)
    if not isinstance(payload, list):
        raise ValueError("Expected the list produced by tools/test.py --out")
    return payload


def attach_predictions(data, payload, source=None, *, score_thr=None, iou_thr=0.5,
                       checkpoint=None):
    if not 0 < iou_thr <= 1 or (score_thr is not None and not 0 <= score_thr <= 1):
        raise ValueError("Invalid IoU/score threshold")
    indexed = {}
    for item in payload:
        if hasattr(item, "to_dict"):
            item = item.to_dict()
        if not isinstance(item, dict) or "img_id" not in item:
            raise ValueError("Each prediction needs its real img_id")
        image_id = str(item["img_id"])
        if image_id in indexed:
            raise ValueError(f"Duplicate prediction img_id: {image_id}")
        indexed[image_id] = item
    expected = {image.image_id for image in data.images}
    if expected != set(indexed):
        raise ValueError(f"Prediction/split ID mismatch; missing={sorted(expected-set(indexed))[:10]}, "
                         f"extra={sorted(set(indexed)-expected)[:10]}")
    for image in data.images:
        item = indexed[image.image_id]
        for key in ("ori_shape", "img_shape", "scale_factor", "view", "band_type"):
            if key not in item:
                raise ValueError(f"{image.image_id}: cache is missing {key}")
        if tuple(item["ori_shape"][:2]) != (image.height, image.width):
            raise ValueError(f"{image.image_id}: ori_shape disagrees with XML")
        if item["view"] != image.view or item["band_type"] != image.band_type:
            raise ValueError(f"{image.image_id}: cache/XML domain metadata mismatch")
        if item.get("flip", False):
            raise ValueError("Use unflipped official test predictions (no TTA)")
        if "img_path" in item and Path(item["img_path"]).name != image.image_path.name:
            raise ValueError(f"{image.image_id}: img_path disagrees with ID")
        scale = _array(item["scale_factor"], np.float64).reshape(-1)
        if len(scale) not in (2, 4) or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError(f"{image.image_id}: invalid scale_factor")
        if len(scale) == 4 and not np.allclose(scale[:2], scale[2:]):
            raise ValueError(f"{image.image_id}: inconsistent four-element scale_factor")
        shape = tuple(int(v) for v in item["img_shape"][:2])
        if min(shape) <= 0 or not np.allclose(
                scale[:2], (shape[1] / image.width, shape[0] / image.height),
                rtol=2e-5, atol=2e-6):
            raise ValueError(f"{image.image_id}: scale_factor/img_shape inconsistent; "
                             "cropped or otherwise transformed caches are unsupported")
        prediction = item.get("pred_instances")
        if hasattr(prediction, "to_dict"):
            prediction = prediction.to_dict()
        if not isinstance(prediction, dict):
            raise ValueError(f"{image.image_id}: missing pred_instances")
        boxes = _array(prediction["bboxes"], np.float32).reshape(-1, 4)
        scores = _array(prediction["scores"], np.float32).reshape(-1)
        raw_labels = _array(prediction["labels"], np.float64).reshape(-1)
        if not np.isfinite(raw_labels).all() or (raw_labels != np.floor(raw_labels)).any():
            raise ValueError("Prediction labels must be finite integers")
        labels = raw_labels.astype(np.int64)
        if not len(boxes) == len(scores) == len(labels):
            raise ValueError(f"{image.image_id}: inconsistent prediction lengths")
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise ValueError(f"{image.image_id}: non-finite predictions")
        if ((boxes[:, 2:] - boxes[:, :2]) < 0).any() or ((scores < 0) | (scores > 1)).any():
            raise ValueError(f"{image.image_id}: invalid boxes/scores")
        if ((labels < 0) | (labels >= len(data.classes))).any():
            raise ValueError(f"{image.image_id}: invalid class IDs")
        keep = np.ones(len(scores), bool) if score_thr is None else scores >= score_thr
        image.boxes, image.scores, image.labels = boxes[keep], scores[keep], labels[keep]
        image.prediction_indices = np.flatnonzero(keep)
        image.scale_factor, image.img_shape = tuple(scale[:2]), shape
    data.metadata["predictions"] = file_info(source) if source else {"kind": "online_official_dataloader"}
    data.metadata["checkpoint"] = file_info(checkpoint) if checkpoint else None
    data.metadata["protocol"].update(iou_thr=iou_thr, extra_score_thr=score_thr)
    data.metadata["protocol_sha256"] = digest(data.metadata["protocol"])
    data.metadata["evaluation_id"] = digest({
        key: data.metadata[key] for key in ("dataset_sha256", "predictions",
                                            "protocol_sha256", "resolved_config_sha256")})
    return data


def load_evaluation(config, predictions, **kwargs):
    attach_keys = ("score_thr", "iou_thr", "checkpoint")
    attach_kwargs = {key: kwargs.pop(key) for key in attach_keys if key in kwargs}
    data = load_dataset(config, **kwargs)
    return attach_predictions(data, load_prediction_file(predictions), predictions, **attach_kwargs)


def require_aligned(first, second):
    for key in ("dataset_sha256", "image_ids", "classes", "protocol_sha256"):
        if first.metadata[key] != second.metadata[key]:
            raise ValueError(f"Models do not share the same {key}")
    for a, b in zip(first.images, second.images):
        if a.scale_factor != b.scale_factor or a.img_shape != b.img_shape:
            raise ValueError(f"Different resize transforms for {a.image_id}")


def match_image(image, iou_thr=0.5, localization_iou=0.1):
    """VOC score-ordered, max-IoU matching, with original GT/prediction indices.

An overlapping second-best uncovered GT is NOT used when the best GT was
already covered. Matches to difficult GT are ignored, including duplicates.
"""
    if not 0 <= localization_iou < iou_thr <= 1:
        raise ValueError("Require 0 <= localization_iou < iou_thr <= 1")
    matches = [None] * len(image.boxes)
    hits = np.full(len(image.gt_boxes), -1, np.int64)
    overlap_fn = official_voc().bbox_overlaps
    for label in np.unique(image.labels):
        pred_ids = np.where(image.labels == label)[0]
        normal = np.where((image.gt_labels == label) & ~image.gt_ignore)[0]
        ignored = np.where((image.gt_labels == label) & image.gt_ignore)[0]
        gt_ids = np.concatenate((normal, ignored))
        ious = overlap_fn(image.boxes[pred_ids], image.gt_boxes[gt_ids],
                          use_legacy_coordinate=True)
        for local in np.argsort(-image.scores[pred_ids]):
            pi = int(pred_ids[local])
            cache_index = pi if image.prediction_indices is None else int(image.prediction_indices[pi])
            gi = int(gt_ids[ious[local].argmax()]) if len(gt_ids) else -1
            overlap = float(ious[local].max()) if len(gt_ids) else 0.0
            status, error, matched = "FP", "background", -1
            if overlap >= iou_thr:
                matched = gi
                if image.gt_ignore[gi]:
                    status, error = "ignored", "difficult"
                elif hits[gi] < 0:
                    status, error, hits[gi] = "TP", "", cache_index
                else:
                    error = "duplicate"
            elif overlap >= localization_iou:
                error = "localization"
            matches[pi] = dict(image_id=image.image_id, prediction_index=cache_index,
                               filtered_prediction_index=pi,
                               label=int(label), score=float(image.scores[pi]),
                               matched_gt_index=matched, best_gt_index=gi, IoU=overlap,
                               status=status, error_type=error)
    return matches, hits


def evaluate(images, classes, iou_thr=0.5, eval_mode="area"):
    if not images:
        raise ValueError("Cannot evaluate an empty image group")
    official = official_voc()
    ap, classes_result = official.eval_map(
        [image.predictions(len(classes)) for image in images],
        [image.annotation() for image in images], dataset=classes,
        iou_thr=iou_thr, eval_mode=eval_mode, use_legacy_coordinate=True,
        logger="silent", nproc=1)
    gt_count = sum(int((~image.gt_ignore).sum()) for image in images)
    rows = [row for image in images for row in match_image(image, iou_thr)[0]]
    tp = sum(row["status"] == "TP" for row in rows)
    fp = sum(row["status"] == "FP" for row in rows)
    # Independent check of the detailed matcher against actual official arrays.
    official_tp = 0
    for label, result in enumerate(classes_result):
        for image in images:
            det = image.predictions(len(classes))[label]
            ann = image.annotation()
            t, f = official.tpfp_default(det, ann["bboxes"][ann["labels"] == label],
                                       ann["bboxes_ignore"][ann["labels_ignore"] == label],
                                       iou_thr=iou_thr, use_legacy_coordinate=True)
            detailed = [r for r in match_image(image, iou_thr)[0] if r["label"] == label]
            if not np.array_equal(t[0], [r["status"] == "TP" for r in detailed]) or \
                    not np.array_equal(f[0], [r["status"] == "FP" for r in detailed]):
                raise AssertionError(f"Detailed matcher differs from official: {image.image_id}")
        if len(result["recall"]):
            official_tp += int(round(float(result["recall"][-1]) * result["num_gts"]))
    if official_tp != tp or sum(int(r["num_gts"]) for r in classes_result) != gt_count:
        raise AssertionError("Official and detailed GT/TP counts differ")
    return dict(image_count=len(images), gt_count=gt_count,
                ignored_gt_count=sum(int(image.gt_ignore.sum()) for image in images),
                prediction_count=len(rows), ignored_prediction_count=sum(r["status"] == "ignored" for r in rows),
                tp=tp, fp=fp, fn=gt_count-tp, AP50=float(ap),
                Recall=tp/gt_count if gt_count else None)


def groups(data):
    yield "overall", "ALL", "ALL", data.images
    for view, band in sorted({(i.view, i.band_type) for i in data.images}):
        yield "domain", view, band, [i for i in data.images if (i.view, i.band_type) == (view, band)]


def curves(images, iou_thr=0.5):
    """Micro curves at attainable score thresholds, grouping all equal scores."""
    rows = [r for image in images for r in match_image(image, iou_thr)[0]]
    rows.sort(key=lambda r: -r["score"])
    total_gt = sum(int((~i.gt_ignore).sum()) for i in images)
    output = [dict(score_threshold=None, tp=0, fp=0, precision=1.0,
                   Recall=0.0 if total_gt else None, FPPI=0.0)]
    tp = fp = 0
    for index, row in enumerate(rows):
        tp += row["status"] == "TP"
        fp += row["status"] == "FP"
        if index + 1 < len(rows) and rows[index+1]["score"] == row["score"]:
            continue
        output.append(dict(score_threshold=row["score"], tp=tp, fp=fp,
                           precision=tp/(tp+fp) if tp+fp else 1.0,
                           Recall=tp/total_gt if total_gt else None, FPPI=fp/len(images)))
    return output


def resize_scale(data, image):
    """Actual cached resize scale, or the deterministic test Resize for train bins."""
    if image.scale_factor is not None:
        return image.scale_factor
    settings = test_resize_settings(data.config)
    scale = settings["scale"]
    if settings.get("keep_ratio", False):
        from mmcv.image.geometric import rescale_size
        width, height = rescale_size((image.width, image.height), tuple(scale))
    else:
        width, height = scale
    return width/image.width, height/image.height


SIZE_CONVENTION = "scaled_legacy_pixel_extent: (x2-x1+1)*sx, (y2-y1+1)*sy"


def gt_sizes(data, image):
    raw = image.gt_boxes[:, 2:] - image.gt_boxes[:, :2] + 1
    scaled = raw * np.asarray(resize_scale(data, image))[None, :]
    return raw, scaled


def validate_bins(bins):
    edges = bins["edges"]
    if bins.get("measure") != "sqrt_area" or bins.get("convention") != SIZE_CONVENTION:
        raise ValueError("Size-bin measure/convention is incompatible")
    if len(edges) < 3 or edges[0] != 0 or edges[-1] is not None:
        raise ValueError("Bins must start at zero and end with null (positive infinity)")
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in edges[:-1]) or \
            any(b <= a for a, b in zip(edges[:-2], edges[1:-1])):
        raise ValueError("Bin edges must be finite and strictly increasing")
    if len(bins["labels"]) != len(edges)-1 or len(set(bins["labels"])) != len(bins["labels"]):
        raise ValueError("Invalid bin labels")
    return bins


def bin_index(value, bins):
    validate_bins(bins)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Invalid size value")
    return int(np.searchsorted(bins["edges"][1:-1], value, side="right"))


def add_data_arguments(parser):
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--id-list", type=Path)
    parser.add_argument("--ann-dir", type=Path)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--score-thr", type=float)
    parser.add_argument("--iou-thr", type=float, default=0.5)


def data_arguments(args):
    return {key: getattr(args, key) for key in
            ("data_root", "id_list", "ann_dir", "image_dir", "score_thr", "iou_thr")}


MANIFEST_FIELDS = ("experiment", "model", "phase", "config", "checkpoint", "epoch",
                   "seed", "training_log", "predictions", "split",
                   "training_protocol", "evaluation_protocol", "eval_dir",
                   "size_dir", "profile_dir")
MANIFEST_PATHS = ("config", "checkpoint", "training_log", "predictions", "split",
                  "eval_dir", "size_dir", "profile_dir")


def read_manifest(path, experiments=None):
    path = Path(path).resolve()
    rows = read_csv(path)
    names = [row.get("experiment") for row in rows]
    if not rows or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Manifest needs unique nonempty experiment names")
    if experiments and set(experiments) - set(names):
        raise ValueError("Requested experiment is missing from manifest")
    result = []
    for row in rows:
        if experiments and row["experiment"] not in experiments:
            continue
        missing = set(MANIFEST_FIELDS) - set(row)
        if missing:
            raise ValueError(f"Missing manifest columns: {sorted(missing)}")
        if row["phase"] not in ("screening", "repeat") or not row["model"]:
            raise ValueError("Manifest phase must be screening/repeat and model must be set")
        for key in ("config", "checkpoint", "predictions", "split", "training_log",
                    "training_protocol", "evaluation_protocol", "eval_dir"):
            if not row[key]:
                raise ValueError(f"{row['experiment']}: missing {key}")
        if row["phase"] == "repeat" and (not row["seed"] or not row["epoch"]):
            raise ValueError("Repeated runs require explicit seed and checkpoint epoch")
        for key in ("seed", "epoch"):
            if row[key]:
                row[key] = int(row[key])
                if row[key] < (1 if key == "epoch" else 0):
                    raise ValueError(f"Invalid manifest {key}")
        for key in MANIFEST_PATHS:
            if row[key]:
                row[key] = str(_path(path.parent, row[key]))
        result.append(row)
    return result
