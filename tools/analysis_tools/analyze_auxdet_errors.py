"""Attribute AuxDet detection errors without changing model or evaluator.

The script runs the configured detector on one validation split and writes
per-image, per-GT, per-domain, size, proposal-recall, RPN-to-RoI attribution,
score-threshold, coordinate-audit, and visualization artifacts.  Predictions
are requested with ``rescale=True``. GT boxes are
mapped back with ``scale_factor`` only when the configured pipeline loaded
annotations before resizing; AuxDet's actual validation pipeline loads them
after resize, so those XML-coordinate GT boxes remain unchanged. IoU uses the
same legacy VOC coordinate convention as :class:`VOCMetric`.

An optional ``--compare-config/--compare-checkpoint`` pair enables paired
case visualizations for two models.  It never changes either model or the
dataset/evaluator configuration.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import os.path as osp
import subprocess
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


IOU_TP = 0.5
IOU_LOCALIZATION = 0.3
DEFAULT_SCORE_THR = 1e-3
DEFAULT_SIZE_EDGES = (0.0, 16.0, 64.0, 256.0, float('inf'))
SCORE_BINS = ((0.001, 0.01, '[0.001,0.01)'),
              (0.01, 0.1, '[0.01,0.1)'),
              (0.1, 0.3, '[0.1,0.3)'),
              (0.3, 0.5, '[0.3,0.5)'),
              (0.5, 0.75, '[0.5,0.75)'),
              (0.75, 1.000001, '[0.75,1.0]'))


def iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray,
               legacy_coordinate: bool = True) -> np.ndarray:
    """Return pairwise IoU using mmdet's VOC-compatible box convention."""
    a = np.asarray(boxes1, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(boxes2, dtype=np.float32).reshape(-1, 4)
    out = np.zeros((len(a), len(b)), dtype=np.float32)
    if not len(a) or not len(b):
        return out
    extra = 1.0 if legacy_coordinate else 0.0
    area_a = np.maximum(a[:, 2] - a[:, 0] + extra, 0) * np.maximum(
        a[:, 3] - a[:, 1] + extra, 0)
    area_b = np.maximum(b[:, 2] - b[:, 0] + extra, 0) * np.maximum(
        b[:, 3] - b[:, 1] + extra, 0)
    xx1 = np.maximum(a[:, None, 0], b[None, :, 0])
    yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 2], b[None, :, 2])
    yy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(xx2 - xx1 + extra, 0) * np.maximum(
        yy2 - yy1 + extra, 0)
    union = np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)
    return inter / union


def _size_group(area: float, edges: Sequence[float]) -> str:
    labels = ('tiny', 'small', 'medium', 'large')
    for i in range(len(edges) - 1):
        if edges[i] <= area < edges[i + 1]:
            return labels[i]
    return 'unknown'


def parse_size_edges(value: str) -> Tuple[float, ...]:
    values = [float(x.strip()) for x in value.split(',')]
    if len(values) != 5 or values[0] != 0 or any(
            values[i] >= values[i + 1] for i in range(4)):
        raise ValueError('size bins must be five increasing edges beginning '
                         'with 0, e.g. 0,16,64,256,inf')
    return tuple(values)


def _as_numpy(value: Any, shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    if value is None:
        if shape is None:
            arr = np.empty((0,), dtype=np.float32)
        else:
            arr = np.empty((0,) + tuple(x for x in shape if x != -1),
                           dtype=np.float32)
    elif hasattr(value, 'detach'):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def _get_field(instance: Any, name: str, default: Any = None) -> Any:
    if instance is None:
        return default
    if hasattr(instance, name):
        return getattr(instance, name)
    if hasattr(instance, 'get'):
        return instance.get(name, default)
    return default


def _gt_arrays(data_sample: Any, edges: Sequence[float],
               gt_was_resized: bool = False) -> List[dict]:
    """Convert a DetDataSample to original-coordinate GT records."""
    normal = getattr(data_sample, 'gt_instances', None)
    ignored = getattr(data_sample, 'ignored_instances', None)
    meta = getattr(data_sample, 'metainfo', {})
    scale = np.asarray(meta.get('scale_factor', 1), dtype=np.float32).reshape(-1)
    if scale.size == 1:
        scale = np.repeat(scale, 4)
    if scale.size != 4:
        scale = np.ones(4, dtype=np.float32)

    records = []
    for inst, ignored_flag in ((normal, False), (ignored, True)):
        if inst is None:
            continue
        boxes = _as_numpy(_get_field(inst, 'bboxes'), ( -1, 4))
        labels = _as_numpy(_get_field(inst, 'labels'), (-1,)).astype(np.int64)
        if gt_was_resized:
            boxes = boxes / scale[None, :]
        for index, (box, label) in enumerate(zip(boxes, labels)):
            width = max(float(box[2] - box[0]), 0.0)
            height = max(float(box[3] - box[1]), 0.0)
            area = width * height
            records.append(dict(index=index, bbox=box.astype(np.float32),
                                label=int(label), ignore=bool(ignored_flag),
                                width=width, height=height, area=area,
                                size_group=_size_group(area, edges)))
    return records


def _prediction_records(data_sample: Any,
                        score_thr: Optional[float] = None) -> List[dict]:
    pred = getattr(data_sample, 'pred_instances', None)
    boxes = _as_numpy(_get_field(pred, 'bboxes'), (-1, 4))
    scores = _as_numpy(_get_field(pred, 'scores'), (-1,)).astype(np.float32)
    labels = _as_numpy(_get_field(pred, 'labels'), (-1,)).astype(np.int64)
    records = []
    for index, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if score_thr is not None and float(score) < score_thr:
            continue
        records.append(dict(index=index, bbox=box.astype(np.float32),
                            score=float(score), label=int(label)))
    records.sort(key=lambda item: (-item['score'], item['index']))
    return records


def match_image(gt: Sequence[dict], predictions: Sequence[dict],
                score_thr: float = DEFAULT_SCORE_THR,
                iou_tp: float = IOU_TP,
                iou_localization: float = IOU_LOCALIZATION) -> dict:
    """Greedily classify one image, preserving one-to-one TP matching."""
    preds = sorted([p for p in predictions if p['score'] >= score_thr],
                   key=lambda item: (-item['score'], item['index']))
    normal = [g for g in gt if not g.get('ignore', False)]
    ignored = [g for g in gt if g.get('ignore', False)]
    normal_ious = iou_matrix([p['bbox'] for p in preds],
                             [g['bbox'] for g in normal])
    ignore_ious = iou_matrix([p['bbox'] for p in preds],
                             [g['bbox'] for g in ignored])
    matched = set()
    matched_pred_by_gt = {}
    pred_rows = []
    for pi, pred in enumerate(preds):
        same = np.asarray([g['label'] == pred['label'] for g in normal],
                          dtype=bool)
        available = np.asarray([j not in matched for j in range(len(normal))],
                                dtype=bool)
        candidates = np.where(same & available)[0]
        best_available = (int(candidates[np.argmax(normal_ious[pi, candidates])])
                          if candidates.size else -1)
        best_available_iou = (float(normal_ious[pi, best_available])
                              if best_available >= 0 else 0.0)
        same_all = np.where(same)[0]
        best_any = (int(same_all[np.argmax(normal_ious[pi, same_all])])
                    if same_all.size else -1)
        best_any_iou = (float(normal_ious[pi, best_any])
                        if best_any >= 0 else 0.0)
        ignore_match = np.where(np.asarray(
            [g['label'] == pred['label'] for g in ignored], dtype=bool) &
            (ignore_ious[pi] >= iou_tp))[0]
        if best_available_iou >= iou_tp:
            matched.add(best_available)
            matched_pred_by_gt[best_available] = pred['index']
            category = 'tp'
            matched_gt = best_available
        elif ignore_match.size:
            category = 'ignored_prediction'
            matched_gt = -1
        elif best_any_iou >= iou_tp:
            category = 'duplicate'
            matched_gt = best_any
        elif best_any_iou >= iou_localization:
            category = 'localization_fp'
            matched_gt = best_any
        else:
            category = 'background_fp'
            matched_gt = -1
        pred_rows.append(dict(pred_index=pred['index'], label=pred['label'],
                              bbox=pred['bbox'], score=pred['score'],
                              category=category, matched_gt=matched_gt,
                              best_iou=best_any_iou))

    gt_rows = []
    for gi, record in enumerate(normal):
        same_pred = [i for i, p in enumerate(preds)
                     if p['label'] == record['label']]
        best_pi = (max(same_pred, key=lambda i: normal_ious[i, gi])
                   if same_pred else -1)
        best_iou = float(normal_ious[best_pi, gi]) if best_pi >= 0 else 0.0
        if gi in matched:
            status = 'matched'
        elif best_iou >= iou_localization:
            status = 'localization_miss'
        else:
            status = 'missed'
        gt_rows.append(dict(gt_index=record['index'], label=record['label'],
                            bbox=record['bbox'], area=record['area'],
                            width=record['width'], height=record['height'],
                            ignore=False,
                            size_group=record['size_group'], status=status,
                            best_iou=best_iou,
                            matched_pred=matched_pred_by_gt.get(gi, -1),
                            best_pred=(preds[best_pi]['index']
                                       if best_pi >= 0 else -1)))
    ignored_rows = []
    for ignored_index, record in enumerate(ignored):
        same_pred = [i for i, p in enumerate(preds)
                     if p['label'] == record['label']]
        best_pi = (max(same_pred, key=lambda i: ignore_ious[i, ignored_index])
                   if same_pred else -1)
        best_iou = (float(ignore_ious[best_pi, ignored_index])
                    if best_pi >= 0 else 0.0)
        ignored_rows.append(dict(gt_index=record['index'], label=record['label'],
                                 bbox=record['bbox'], area=record['area'],
                                 width=record['width'], height=record['height'],
                                 ignore=True, size_group=record['size_group'],
                                 status='ignored_gt', best_iou=best_iou,
                                 matched_pred=-1,
                                 best_pred=(preds[best_pi]['index']
                                            if best_pi >= 0 else -1)))
    counts = Counter(row['category'] for row in pred_rows)
    counts.update('fn' for row in gt_rows if row['status'] != 'matched')
    tp = counts['tp']
    fp = counts['background_fp'] + counts['localization_fp'] + counts['duplicate']
    fn = counts['fn']
    return dict(predictions=pred_rows, gts=gt_rows,
                ignored_gts=ignored_rows, counts=counts, tp=tp,
                fp=fp, fn=fn, ignored_pred=counts['ignored_prediction'],
                precision=(tp / (tp + fp) if tp + fp else 0.0),
                recall=(tp / (tp + fn) if tp + fn else 0.0))


def _voc_ap(scores: Sequence[float], tp: Sequence[int], num_gt: int) -> Optional[float]:
    if num_gt <= 0:
        return None
    order = np.argsort(-np.asarray(scores), kind='mergesort')
    tp_arr = np.asarray(tp, dtype=np.float32)[order]
    fp_arr = 1.0 - tp_arr
    recall = np.cumsum(tp_arr) / float(num_gt)
    precision = np.cumsum(tp_arr) / np.maximum(np.cumsum(tp_arr + fp_arr), 1e-12)
    mrec = np.r_[0.0, recall, 1.0]
    mpre = np.r_[0.0, precision, 0.0]
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    changes = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]))


def ap50_subset(image_records: Sequence[dict], group: Optional[str] = None,
                domain_type: Optional[str] = None,
                domain_value: Optional[str] = None) -> Optional[float]:
    """Compute an explicitly labelled area-under-PR AP50 subset metric."""
    scores, tp = [], []
    num_gt = 0
    for image in image_records:
        if domain_type == 'view' and image['view'] != domain_value:
            continue
        if domain_type == 'band_type' and image['band_type'] != domain_value:
            continue
        gt = [g for g in image['gt'] if not g.get('ignore', False)
              and (group is None or g['size_group'] == group)]
        num_gt += len(gt)
        preds = [p for p in image['predictions'] if p['score'] >= image['score_thr']]
        ious = iou_matrix([p['bbox'] for p in preds], [g['bbox'] for g in gt])
        covered = set()
        for i, pred in sorted(enumerate(preds), key=lambda x: -x[1]['score']):
            candidates = [j for j, g in enumerate(gt)
                          if g['label'] == pred['label'] and j not in covered]
            best = max(candidates, key=lambda j: ious[i, j]) if candidates else -1
            is_tp = best >= 0 and ious[i, best] >= IOU_TP
            if is_tp:
                covered.add(best)
            scores.append(pred['score'])
            tp.append(int(is_tp))
    return _voc_ap(scores, tp, num_gt)


def proposal_recall_rows(proposals: Sequence[dict], gt: Sequence[dict],
                         image_id: str) -> List[dict]:
    normal = [g for g in gt if not g.get('ignore', False)]
    boxes = np.asarray([p['bbox'] for p in proposals], dtype=np.float32).reshape(-1, 4)
    scores = np.asarray([p.get('score', 0.0) for p in proposals], dtype=np.float32)
    order = np.argsort(-scores, kind='mergesort')
    rows = []
    for topk in (100, 1000):
        selected = boxes[order[:topk]]
        ious = iou_matrix([g['bbox'] for g in normal], selected)
        for thr in (0.3, 0.5, 0.7):
            hits = (ious.max(axis=1) >= thr) if len(selected) else np.zeros(len(normal), bool)
            rows.append(dict(image_id=image_id, topk=topk, iou_thr=thr,
                             gt_count=len(normal), hit_count=int(hits.sum()),
                             size_group='all'))
            for group in ('tiny', 'small', 'medium', 'large'):
                indexes = [i for i, g in enumerate(normal) if g['size_group'] == group]
                hit = int(hits[indexes].sum()) if indexes else 0
                rows.append(dict(image_id=image_id, topk=topk, iou_thr=thr,
                                 gt_count=len(indexes), hit_count=hit,
                                 size_group=group))
    return rows


def scale_factor_array(sample_or_meta: Any) -> np.ndarray:
    """Return MMDetection's xyxy scale factor as four finite values.

    ``scale_factor`` maps original image coordinates to the resized model
    input.  The inverse operation used by :func:`proposal_boxes_original`
    therefore divides by this array.  A missing or malformed value is treated
    as an identity transform so that the audit remains explicit rather than
    silently applying a partial scale.
    """
    meta = (sample_or_meta if isinstance(sample_or_meta, dict)
            else _metainfo(sample_or_meta))
    value = np.asarray(meta.get('scale_factor', 1), dtype=np.float32).reshape(-1)
    if value.size == 1:
        value = np.repeat(value, 4)
    if value.size != 4 or not np.all(np.isfinite(value)) or np.any(value == 0):
        return np.ones(4, dtype=np.float32)
    return value


def proposal_boxes_original(proposal: Any, sample_or_meta: Any) -> np.ndarray:
    """Copy RPN boxes from resized/input coordinates to original coordinates."""
    boxes = _as_numpy(_get_field(proposal, 'bboxes'), (-1, 4)).astype(np.float32)
    return boxes / scale_factor_array(sample_or_meta)[None, :]


def rpn_to_roi_rows(gt: Sequence[dict], proposals: Sequence[dict],
                    final_result: dict, image_id: str,
                    view: str = 'Unknown', band_type: str = 'Unknown',
                    image_path: str = '') -> List[dict]:
    """Join original-coordinate RPN IoU with the actual one-to-one TP result.

    ``best_final_iou`` is diagnostic (the best final box for the GT), while
    ``final_hit_05`` is deliberately taken from ``match_image``'s matched
    status.  This prevents a second, many-to-one IoU test from overstating
    final recall when several predictions compete for one GT.
    """
    normal = [g for g in gt if not g.get('ignore', False)]
    proposal_boxes = np.asarray([p['bbox'] for p in proposals], dtype=np.float32).reshape(-1, 4)
    proposal_ious = iou_matrix([g['bbox'] for g in normal], proposal_boxes)
    final_gt_rows = list(final_result.get('gts', []))
    final_preds = list(final_result.get('predictions', []))
    rows = []
    for gi, record in enumerate(normal):
        rpn_pi = (int(np.argmax(proposal_ious[gi]))
                  if proposal_ious.shape[1] else -1)
        best_rpn = (float(proposal_ious[gi, rpn_pi])
                    if rpn_pi >= 0 else 0.0)
        final_row = final_gt_rows[gi] if gi < len(final_gt_rows) else {}
        best_final = float(final_row.get('best_iou', 0.0))
        final_hit = final_row.get('status') == 'matched'
        same_final = [pi for pi, pred in enumerate(final_preds)
                      if pred['label'] == record['label']]
        final_pi = (max(same_final,
                        key=lambda pi: iou_matrix(
                            [final_preds[pi]['bbox']], [record['bbox']])[0, 0])
                    if same_final else -1)
        matched_pred_index = int(final_row.get('matched_pred', -1))
        matched_pi = next((pi for pi, pred in enumerate(final_preds)
                           if pred.get('pred_index', pred.get('index', -1))
                           == matched_pred_index), -1)

        def fmt_box(box: Any) -> str:
            if box is None:
                return ''
            return ' '.join(f'{float(x):.4f}' for x in box)

        def pred_index(item: Optional[dict]) -> int:
            return (int(item.get('pred_index', item.get('index', -1)))
                    if item is not None else -1)

        rpn_pred = proposals[rpn_pi] if rpn_pi >= 0 else None
        final_pred = final_preds[final_pi] if final_pi >= 0 else None
        matched_pred = final_preds[matched_pi] if matched_pi >= 0 else None
        if best_rpn >= 0.5 and final_hit:
            category = 'A_rpn05_final_tp'
        elif 0.3 <= best_rpn < 0.5 and final_hit:
            category = 'B_rpn03_05_final_tp'
        elif best_rpn < 0.3 and final_hit:
            category = 'C_rpn_lt03_final_tp'
        elif best_rpn >= 0.5 and not final_hit:
            category = 'D_rpn05_final_miss'
        elif best_rpn < 0.3 and not final_hit:
            category = 'E_rpn_lt03_final_miss'
        else:
            # [0.3, 0.5) misses are outside the requested A-E bins; retain
            # them explicitly instead of silently folding them into E.
            category = 'F_rpn03_05_final_miss'
        rows.append(dict(image_id=image_id, image_path=image_path,
                         gt_index=record.get('index', gi),
                         gt_bbox=' '.join(f'{float(x):.4f}' for x in record['bbox']),
                         gt_area=float(record['area']), best_rpn_iou=best_rpn,
                         best_final_iou=best_final,
                         rpn_best_index=rpn_pi,
                         rpn_best_score=(float(rpn_pred.get('score', 0.0))
                                         if rpn_pred else 0.0),
                         rpn_best_bbox=fmt_box(rpn_pred['bbox']
                                               if rpn_pred else None),
                         final_best_index=pred_index(final_pred),
                         final_best_score=(float(final_pred['score'])
                                           if final_pred else 0.0),
                         final_best_bbox=fmt_box(final_pred['bbox']
                                                 if final_pred else None),
                         final_matched_index=matched_pred_index,
                         final_matched_score=(float(matched_pred['score'])
                                              if matched_pred else 0.0),
                         final_matched_bbox=fmt_box(
                             matched_pred['bbox'] if matched_pred else None),
                         rpn_hit_03=int(best_rpn >= 0.3),
                         rpn_hit_05=int(best_rpn >= 0.5),
                         final_hit_05=int(final_hit), category=category,
                         size_group=record.get('size_group', 'unknown'),
                         view=view, band_type=band_type))
    return rows


def threshold_summary(image_records: Sequence[dict], threshold: float) -> dict:
    """Re-score unchanged post-NMS predictions at one fixed threshold."""
    counts = Counter()
    for image in image_records:
        result = match_image(image['gt'], image.get('raw_predictions',
                                                      image['predictions']),
                             score_thr=threshold)
        counts.update(result['counts'])
    tp = counts['tp']
    background = counts['background_fp']
    localization = counts['localization_fp']
    duplicate = counts['duplicate']
    fp = background + localization + duplicate
    fn = counts['fn']
    return dict(score_threshold=threshold, tp=tp,
                background_fp=background, localization_fp=localization,
                duplicate_fp=duplicate, fp=fp, fn=fn,
                precision=(tp / (tp + fp) if tp + fp else 0.0),
                recall=(tp / (tp + fn) if tp + fn else 0.0))


def select_c_cases(rows: Sequence[dict], limit: int = 20) -> List[dict]:
    """Select deterministic C-class cases with image-level diversity first."""
    candidates = [row for row in rows
                  if row.get('category') == 'C_rpn_lt03_final_tp']
    candidates.sort(key=lambda row: (-float(row.get('final_matched_score', 0.0)),
                                    -float(row.get('best_final_iou', 0.0)),
                                    row.get('image_id', ''),
                                    int(row.get('gt_index', 0))))
    selected, used_images = [], set()
    for row in candidates:
        if len(selected) >= limit:
            break
        if row.get('image_id') in used_images:
            continue
        selected.append(row)
        used_images.add(row.get('image_id'))
    if len(selected) < limit:
        selected_ids = {id(row) for row in selected}
        selected.extend(row for row in candidates
                        if id(row) not in selected_ids)
    return selected[:limit]


def _metainfo(sample: Any) -> dict:
    return dict(getattr(sample, 'metainfo', {}) or {})


def _write_csv(path: str, rows: Sequence[dict],
               fieldnames: Optional[Sequence[str]] = None) -> None:
    keys = list(fieldnames or dict.fromkeys(k for row in rows for k in row.keys()))
    if not keys:
        with open(path, 'w', newline='', encoding='utf-8'):
            return
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            clean = {}
            for key in keys:
                value = row.get(key, '')
                clean[key] = value.tolist() if isinstance(value, np.ndarray) else value
            writer.writerow(clean)


def _draw_case(path: str, out_path: str, gt: Sequence[dict],
               pred_rows: Sequence[dict], title: str) -> None:
    try:
        from PIL import Image, ImageDraw
        image = Image.open(path).convert('RGB')
        draw = ImageDraw.Draw(image)
        for record in gt:
            box = tuple(float(x) for x in record['bbox'])
            color = 'yellow' if record.get('ignore') else 'lime'
            draw.rectangle(box, outline=color, width=2)
            draw.text((box[0], max(0, box[1] - 12)), 'GT', fill=color)
        colors = {'tp': 'blue', 'background_fp': 'red',
                  'localization_fp': 'orange', 'duplicate': 'magenta',
                  'ignored_prediction': 'gray'}
        for row in pred_rows:
            box = tuple(float(x) for x in row['bbox'])
            color = colors.get(row['category'], 'red')
            draw.rectangle(box, outline=color, width=2)
            draw.text((box[0], box[1]),
                      f"{row['category']} {row['score']:.3f} "
                      f"IoU={row['best_iou']:.2f}", fill=color)
        draw.text((5, 5), title, fill='white')
        image.save(out_path)
    except Exception as exc:
        with open(out_path + '.error.txt', 'w', encoding='utf-8') as handle:
            handle.write(f'visualization unavailable: {exc}\n')


def _run_model(model, raw_batch: dict, device: str):
    import torch
    data = model.data_preprocessor(deepcopy(raw_batch), training=False)
    inputs, samples = data['inputs'], data['data_samples']
    model.eval()
    with torch.no_grad():
        features = model.extract_feat(inputs, samples)
        # The detector's real path keeps proposals in resized/input space for
        # RoI extraction, then only rescales final RoI boxes.  Preserve that
        # contract; proposal recall converts a copy below for analysis.
        rpn_results = model.rpn_head.predict(features, samples, rescale=False)
        roi_results = model.roi_head.predict(
            features, rpn_results, samples, rescale=True)
        model.add_pred_to_datasample(samples, roi_results)
    return samples, rpn_results


def _load_model(cfg_path: str, checkpoint: str, device: str):
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(cfg_path)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint, map_location='cpu', strict=True)
    model.to(device).eval()
    return cfg, model


def _set_data_root(cfg, data_root: str, split: str) -> None:
    loader = cfg[f'{split}_dataloader']
    dataset = loader.dataset
    while 'dataset' in dataset:
        dataset = dataset.dataset
    dataset.data_root = data_root


def _pipeline_resizes_loaded_boxes(cfg, split: str) -> bool:
    """Check transform order instead of guessing from ``scale_factor``.

    AuxDet's validation pipeline resizes before ``LoadAnnotations``, so its
    GT boxes stay in the XML/original coordinate system while predictions are
    returned in that same system with ``rescale=True``.  Standard pipelines
    that load annotations first need the inverse scale here.
    """
    dataset = cfg[f'{split}_dataloader'].dataset
    while 'dataset' in dataset:
        dataset = dataset.dataset
    pipeline = dataset.pipeline
    resize_index = next((i for i, item in enumerate(pipeline)
                         if item.get('type') in ('Resize', 'FixScaleResize')),
                        None)
    load_index = next((i for i, item in enumerate(pipeline)
                       if item.get('type') == 'LoadAnnotations'), None)
    return resize_index is not None and load_index is not None and load_index < resize_index


def _dataset_normal_gt_count(dataset) -> Optional[int]:
    """Count unignored annotations from a built dataset when available."""
    current = dataset
    while hasattr(current, 'dataset'):
        current = current.dataset
    data_list = getattr(current, 'data_list', None)
    if data_list is None:
        return None
    return sum(sum(not instance.get('ignore_flag', 0)
                   for instance in item.get('instances', []))
               for item in data_list)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--split', default='val', choices=('val', 'test'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--max-vis', type=int, default=20)
    parser.add_argument('--score-thr', type=float, default=DEFAULT_SCORE_THR)
    parser.add_argument('--size-bins', default='0,16,64,256,inf')
    parser.add_argument('--expected-gt', type=int, default=4884)
    parser.add_argument('--compare-config')
    parser.add_argument('--compare-checkpoint')
    args = parser.parse_args()
    if bool(args.compare_config) != bool(args.compare_checkpoint):
        parser.error('--compare-config and --compare-checkpoint must be supplied together')
    edges = parse_size_edges(args.size_bins)
    os.makedirs(args.out_dir, exist_ok=True)

    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    cfg, model = _load_model(args.config, args.checkpoint, args.device)
    _set_data_root(cfg, args.data_root, args.split)
    gt_was_resized = _pipeline_resizes_loaded_boxes(cfg, args.split)
    loader = Runner.build_dataloader(cfg[f'{args.split}_dataloader'])
    preflight_gt_count = _dataset_normal_gt_count(loader.dataset)
    preflight_comparison_allowed = (preflight_gt_count is None or
                                    preflight_gt_count == args.expected_gt)
    compare_cfg = compare_model = None
    if args.compare_config and preflight_comparison_allowed:
        compare_cfg, compare_model = _load_model(
            args.compare_config, args.compare_checkpoint, args.device)
        _set_data_root(compare_cfg, args.data_root, args.split)
        compare_gt_was_resized = _pipeline_resizes_loaded_boxes(
            compare_cfg, args.split)
    else:
        compare_gt_was_resized = gt_was_resized
        if args.compare_config:
            print(f'GT preflight count {preflight_gt_count} differs from '
                  f'expected {args.expected_gt}; paired comparison stopped.')

    image_records, gt_rows, pred_rows, proposal_rows = [], [], [], []
    rpn_roi_rows, coordinate_audit_rows = [], []
    scale_factors_seen = []
    visualization_pool = defaultdict(list)
    expected = args.expected_gt
    for batch in loader:
        samples, proposals = _run_model(model, batch, args.device)
        compare_samples = None
        if compare_model is not None:
            compare_samples, _ = _run_model(compare_model, batch, args.device)
        for index, sample in enumerate(samples):
            meta = _metainfo(sample)
            sample_scale = scale_factor_array(sample)
            scale_factors_seen.append(tuple(float(x) for x in sample_scale))
            image_id = str(meta.get('img_id', osp.basename(meta.get('img_path', index))))
            gt = _gt_arrays(sample, edges, gt_was_resized)
            raw_preds = _prediction_records(sample)
            preds = [p for p in raw_preds if p['score'] >= args.score_thr]
            result = match_image(gt, preds, args.score_thr)
            image = dict(image_id=image_id, view=meta.get('view', 'Unknown'),
                         band_type=meta.get('band_type', 'Unknown'), gt=gt,
                         predictions=preds, raw_predictions=raw_preds,
                         score_thr=args.score_thr,
                         result=result, image_path=meta.get('img_path'))
            image_records.append(image)
            for row in result['gts']:
                row.update(image_id=image_id, view=image['view'], band_type=image['band_type'])
                gt_rows.append(row)
            for row in result['ignored_gts']:
                row.update(image_id=image_id, view=image['view'], band_type=image['band_type'])
                gt_rows.append(row)
            for row in result['predictions']:
                row.update(image_id=image_id, view=image['view'], band_type=image['band_type'])
                pred_rows.append(row)
                if row['category'] == 'background_fp' and row['score'] >= 0.5:
                    visualization_pool['high_conf_background'].append((image, row))
            if any(row['status'] == 'missed' for row in result['gts']):
                visualization_pool['missed'].append(image)
            if any(row['status'] == 'localization_miss' for row in result['gts']):
                visualization_pool['localization'].append(image)
            if proposals is not None:
                proposal = proposals[index]
                raw_proposal_boxes = _as_numpy(_get_field(proposal, 'bboxes'), (-1, 4))
                scores = _as_numpy(_get_field(proposal, 'scores'), (-1,))
                boxes = proposal_boxes_original(proposal, sample)
                proposal_records = [dict(bbox=box, score=float(score))
                                    for box, score in zip(boxes, scores)]
                image['rpn_proposals'] = proposal_records
                proposal_rows.extend(proposal_recall_rows(
                    proposal_records, gt, image_id))
                rpn_roi_rows.extend(rpn_to_roi_rows(
                    gt, proposal_records, result, image_id,
                    image['view'], image['band_type'], image['image_path']))
                coordinate_audit_rows.append(dict(
                    image_id=image_id,
                    scale_factor=' '.join(f'{float(x):.6g}'
                                          for x in sample_scale),
                    rpn_input_box_count=len(raw_proposal_boxes),
                    rpn_input_min=(np.min(raw_proposal_boxes, axis=0).tolist()
                                   if len(raw_proposal_boxes) else []),
                    rpn_input_max=(np.max(raw_proposal_boxes, axis=0).tolist()
                                   if len(raw_proposal_boxes) else []),
                    rpn_original_min=(np.min(boxes, axis=0).tolist()
                                      if len(boxes) else []),
                    rpn_original_max=(np.max(boxes, axis=0).tolist()
                                      if len(boxes) else []),
                    gt_min=(np.min([g['bbox'] for g in gt], axis=0).tolist()
                            if gt else []),
                    gt_max=(np.max([g['bbox'] for g in gt], axis=0).tolist()
                            if gt else []),
                    final_min=(np.min([p['bbox'] for p in raw_preds], axis=0).tolist()
                               if raw_preds else []),
                    final_max=(np.max([p['bbox'] for p in raw_preds], axis=0).tolist()
                               if raw_preds else [])))
            else:
                image['rpn_proposals'] = []

            if compare_samples is not None:
                compare_gt = _gt_arrays(compare_samples[index], edges,
                                        compare_gt_was_resized)
                compare_preds = _prediction_records(compare_samples[index], args.score_thr)
                compare_result = match_image(compare_gt, compare_preds, args.score_thr)
                primary_good = result['tp'] > 0 and result['fp'] == 0 and result['fn'] == 0
                compare_good = compare_result['tp'] > 0 and compare_result['fp'] == 0 and compare_result['fn'] == 0
                if primary_good and not compare_good:
                    visualization_pool['primary_correct_compare_wrong'].append(
                        (image, compare_result, compare_samples[index]))
                if compare_good and not primary_good:
                    visualization_pool['compare_correct_primary_wrong'].append(
                        (image, compare_result, compare_samples[index]))

    total_gt = sum(sum(not g.get('ignore', False) for g in image['gt'])
                   for image in image_records)
    comparison_allowed = total_gt == expected
    rpn_max_per_img = None
    try:
        rpn_max_per_img = cfg.model.test_cfg.rpn.get('max_per_img', None)
    except (AttributeError, KeyError, TypeError):
        pass
    counts = Counter()
    for image in image_records:
        counts.update(image['result']['counts'])
    tp, fp, fn = counts['tp'], counts['background_fp'] + counts['localization_fp'] + counts['duplicate'], counts['fn']
    ap50_custom = ap50_subset(image_records)
    summary = dict(git_commit=subprocess.run(
        ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
        check=False).stdout.strip(), config=osp.abspath(args.config),
                    checkpoint=osp.abspath(args.checkpoint), data_root=args.data_root,
                    split=args.split, image_count=len(image_records), gt_count=total_gt,
                    gt_preflight_count=preflight_gt_count,
                    expected_gt=expected, expected_gt_match=comparison_allowed,
                    score_threshold=args.score_thr, iou_tp=IOU_TP,
                    iou_localization=IOU_LOCALIZATION, legacy_coordinate=True,
                    gt_was_resized_before_model=gt_was_resized,
                    rpn_max_per_img=rpn_max_per_img,
                    rpn_proposal_images=sum(bool(x.get('rpn_proposals'))
                                            for x in image_records),
                    rpn_roi_rows=len(rpn_roi_rows),
                    c_case_count=min(20, sum(row['category'] ==
                                             'C_rpn_lt03_final_tp'
                                             for row in rpn_roi_rows)),
                    scale_factor_identity_images=sum(
                        all(abs(x - 1.0) < 1e-6 for x in factor)
                        for factor in scale_factors_seen),
                    scale_factor_nonidentity_images=sum(
                        any(abs(x - 1.0) >= 1e-6 for x in factor)
                        for factor in scale_factors_seen),
                    scale_factor_unique=[' '.join(f'{x:.6g}' for x in factor)
                                         for factor in sorted(
                                             set(scale_factors_seen))],
                    tp=tp, fp=fp, fn=fn,
                    precision=(tp / (tp + fp) if tp + fp else 0.0),
                    recall=(tp / (tp + fn) if tp + fn else 0.0),
                    ap50_custom=ap50_custom)
    if args.compare_config:
        summary.update(compare_config=osp.abspath(args.compare_config),
                       compare_checkpoint=osp.abspath(args.compare_checkpoint),
                       comparison_status=('enabled' if comparison_allowed
                                           else 'stopped_gt_count_mismatch'))
    if args.compare_config and not comparison_allowed:
        summary['comparison_status'] = 'stopped_gt_count_mismatch'
        visualization_pool['primary_correct_compare_wrong'].clear()
        visualization_pool['compare_correct_primary_wrong'].clear()
    _write_csv(osp.join(args.out_dir, 'per_image.csv'), [
        dict(image_id=x['image_id'], view=x['view'], band_type=x['band_type'],
             gt_count=sum(not g.get('ignore', False) for g in x['gt']),
             **{k: x['result'][k] for k in ('tp', 'fp', 'fn', 'precision', 'recall')})
        for x in image_records])
    _write_csv(osp.join(args.out_dir, 'per_gt.csv'), gt_rows)
    error_rows = [dict(category=category, score_bin='all', count=count)
                  for category, count in sorted(counts.items())]
    for category in ('background_fp', 'localization_fp', 'duplicate',
                     'ignored_prediction', 'tp'):
        category_preds = [p for p in pred_rows if p['category'] == category]
        for low, high, label in SCORE_BINS:
            count = sum(low <= p['score'] < high for p in category_preds)
            error_rows.append(dict(category=category, score_bin=label,
                                   count=count))
    _write_csv(osp.join(args.out_dir, 'error_summary.csv'), error_rows)
    size_rows = []
    for group in ('tiny', 'small', 'medium', 'large'):
        group_gt = [g for image in image_records for g in image['gt']
                    if not g.get('ignore', False) and g['size_group'] == group]
        matched = sum(row['status'] == 'matched' for row in gt_rows
                      if row['size_group'] == group)
        size_rows.append(dict(size_group=group, gt_count=len(group_gt),
                              matched_count=matched,
                              recall=(matched / len(group_gt) if group_gt else 0),
                              ap50_custom=ap50_subset(image_records, group)))
    _write_csv(osp.join(args.out_dir, 'size_analysis.csv'), size_rows)
    domain_rows = []
    for key, value in [('view', 'Air'), ('view', 'Land'), ('view', 'Space'),
                       ('band_type', 'LWIR'), ('band_type', 'NIR'), ('band_type', 'SWIR')]:
        subset = [x for x in image_records if x[key] == value]
        gt_count = sum(sum(not g.get('ignore', False) for g in x['gt']) for x in subset)
        tp_count = sum(x['result']['tp'] for x in subset)
        fn_count = sum(x['result']['fn'] for x in subset)
        domain_rows.append(dict(domain_type=key, domain=value,
                                image_count=len(subset), gt_count=gt_count,
                                recall=(tp_count / (tp_count + fn_count)
                                        if tp_count + fn_count else 0),
                                ap50_custom=ap50_subset(
                                    image_records, domain_type=key,
                                    domain_value=value)))
    _write_csv(osp.join(args.out_dir, 'per_domain.csv'), domain_rows)
    if proposal_rows:
        aggregate = defaultdict(lambda: [0, 0])
        for row in proposal_rows:
            key = (row['topk'], row['iou_thr'], row['size_group'])
            aggregate[key][0] += row['gt_count']
            aggregate[key][1] += row['hit_count']
        proposal_output = list(proposal_rows)
        proposal_output.extend(
            dict(row_type='aggregate', topk=topk, iou_thr=iou_thr,
                 size_group=group, gt_count=values[0],
                 hit_count=values[1],
                 recall=(values[1] / values[0] if values[0] else 0.0))
            for (topk, iou_thr, group), values in sorted(aggregate.items()))
        _write_csv(osp.join(args.out_dir, 'proposal_recall.csv'), proposal_output)
    else:
        open(osp.join(args.out_dir, 'proposal_recall.csv'), 'w').close()

    # RPN-to-RoI attribution uses the same one-to-one final TP assignment as
    # the main error report.  It is intentionally emitted separately so the
    # historical per-GT/per-image CSVs remain backward compatible.
    _write_csv(osp.join(args.out_dir, 'rpn_to_roi_analysis.csv'), rpn_roi_rows,
               fieldnames=('image_id', 'image_path', 'gt_index', 'gt_bbox', 'gt_area',
                           'best_rpn_iou', 'best_final_iou', 'rpn_hit_03',
                           'rpn_best_index', 'rpn_best_score', 'rpn_best_bbox',
                           'final_best_index', 'final_best_score',
                           'final_best_bbox', 'final_matched_index',
                           'final_matched_score', 'final_matched_bbox',
                           'rpn_hit_05', 'final_hit_05', 'category',
                           'size_group', 'view', 'band_type'))
    c_cases = select_c_cases(rpn_roi_rows, limit=20)
    _write_csv(osp.join(args.out_dir, 'c_rpn_roi_cases.csv'), c_cases,
               fieldnames=('image_id', 'image_path', 'gt_index', 'gt_bbox',
                           'gt_area', 'best_rpn_iou', 'rpn_best_index',
                           'rpn_best_score', 'rpn_best_bbox',
                           'best_final_iou', 'final_best_index',
                           'final_best_score', 'final_best_bbox',
                           'final_matched_index', 'final_matched_score',
                           'final_matched_bbox', 'rpn_hit_03', 'rpn_hit_05',
                           'final_hit_05', 'category', 'size_group', 'view',
                           'band_type'))
    category_counts = Counter(row['category'] for row in rpn_roi_rows)
    rpn_summary_rows = [dict(category=category, gt_count=count,
                             fraction=(count / len(rpn_roi_rows)
                                       if rpn_roi_rows else 0.0))
                        for category, count in sorted(category_counts.items())]
    rpn_summary_rows.append(dict(category='all_normal_gt',
                                 gt_count=len(rpn_roi_rows), fraction=1.0
                                 if rpn_roi_rows else 0.0))
    _write_csv(osp.join(args.out_dir, 'rpn_to_roi_summary.csv'),
               rpn_summary_rows,
               fieldnames=('category', 'gt_count', 'fraction'))
    _write_csv(osp.join(args.out_dir, 'score_threshold_analysis.csv'),
               [threshold_summary(image_records, threshold)
                for threshold in (0.1, 0.3, 0.5, 0.75)],
               fieldnames=('score_threshold', 'tp', 'background_fp',
                           'localization_fp', 'duplicate_fp', 'fp', 'fn',
                           'precision', 'recall'))

    # Keep coordinate evidence concise but inspectable.  The complete
    # per-image ranges are retained in this audit table; JSON-like lists are
    # serialized by _write_csv without changing their numeric values.
    _write_csv(osp.join(args.out_dir, 'coordinate_audit.csv'),
               coordinate_audit_rows,
               fieldnames=('image_id', 'scale_factor', 'rpn_input_box_count',
                           'rpn_input_min', 'rpn_input_max',
                           'rpn_original_min', 'rpn_original_max', 'gt_min',
                           'gt_max', 'final_min', 'final_max'))
    with open(osp.join(args.out_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)

    vis_count = defaultdict(int)
    for category, values in visualization_pool.items():
        for item in values:
            if vis_count[category] >= args.max_vis:
                break
            image = item[0] if isinstance(item, tuple) else item
            if not image.get('image_path'):
                continue
            out = osp.join(args.out_dir, 'errors', category,
                           f"{vis_count[category]:03d}_{image['image_id']}.png")
            os.makedirs(osp.dirname(out), exist_ok=True)
            rows = image['result']['predictions']
            if category in ('primary_correct_compare_wrong',
                            'compare_correct_primary_wrong') and len(item) > 1:
                rows = item[1]['predictions']
            _draw_case(image['image_path'], out, image['gt'], rows,
                       f'{category}: {image["image_id"]}')
            vis_count[category] += 1

    lines = [f'# AuxDet error analysis', '', f'- Git Commit: `{summary["git_commit"]}`',
             f'- Config: `{summary["config"]}`', f'- Checkpoint: `{summary["checkpoint"]}`',
             f'- Split: `{args.split}`', f'- Images analyzed: {len(image_records)}',
             f'- Normal GT: {total_gt} (expected {expected})',
             f'- Size bins (original image coordinates): {args.size_bins}',
             f'- Score threshold: {args.score_thr}', '- TP IoU: 0.5',
             '- Localization interval: [0.3, 0.5)',
             '- IoU: legacy VOC coordinate convention; predictions use `rescale=True` and GT scaling follows transform order.',
             f'- Cascade RPN max_per_img: {rpn_max_per_img}',
             '', '## Summary', '',
             f'- TP={tp}, FP={fp}, FN={fn}',
             f'- Precision={summary["precision"]:.6f}, Recall={summary["recall"]:.6f}',
             f'- C-class cases exported: {summary["c_case_count"]}',
             f'- Custom AP50 (area PR): {summary["ap50_custom"]:.6f}'
             if summary['ap50_custom'] is not None
             else '- Custom AP50 (area PR): unavailable (no normal GT)',
             '', '## Interpretation', '',
             'Background FP, localization FP, duplicate detections, and ignored predictions are separated in `per_gt.csv` and `error_summary.csv`.',
             'Size/domain AP values are explicitly named `ap50_custom`; they are subset area-under-PR diagnostics and are not official VOCMetric AP50.',
             'Proposal recall uses final Cascade RPN proposals returned by `rpn_head.predict` at top-100/top-1000. If max_per_img is below 1000, Recall@1000 is explicitly capped by the unchanged model test configuration.',
             'Platform and band labels are read from `DetDataSample.metainfo`; unknown labels are retained and not guessed.']
    if args.compare_config:
        lines += ['Paired visualization folders use `primary` for `--config` '
                  'and `compare` for `--compare-config`.']
    if not comparison_allowed:
        warning = (f'WARNING: normal GT count {total_gt} differs from expected '
                   f'{expected}.')
        if args.compare_config:
            warning += ' Paired cross-model comparison was stopped.'
        lines += ['', warning]
    with open(osp.join(args.out_dir, 'report.md'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')

    audit_lines = [
        '# RPN to RoI coordinate audit', '',
        f'- Git Commit: `{summary["git_commit"]}`',
        f'- Config: `{summary["config"]}`',
        f'- Checkpoint: `{summary["checkpoint"]}`',
        f'- Split: `{args.split}`; images analyzed: {len(image_records)}',
        f'- Normal GT: {total_gt}; expected: {expected}; '
        f'count_match={comparison_allowed}', '',
        f'- scale_factor identity images: '
        f'{summary["scale_factor_identity_images"]}/{len(scale_factors_seen)}; '
        f'non-identity images: {summary["scale_factor_nonidentity_images"]}',
        f'- unique scale_factor values: {summary["scale_factor_unique"]}', '',
        '## Coordinate contracts', '',
        '- Cascade RPN is called with `rescale=False`; its proposals are in '
        'the resized/model-input coordinate system.',
        '- The analysis copies those proposal boxes and divides xyxy by the '
        'four-value `scale_factor` (original to resized). It never mutates '
        'the proposal object used by the RoI head.',
        '- Cascade RoI prediction is called with `rescale=True`; final '
        'detection boxes are already returned in original-image coordinates '
        'and are not divided again.',
        '- GT boxes follow the configured transform order. '
        f'`gt_was_resized_before_model={gt_was_resized}`. For the current '
        'AuxDet validation pipeline, Resize precedes LoadAnnotations, so XML '
        'GT coordinates remain in the original coordinate system.',
        '- `coordinate_audit.csv` records the scale factor and coordinate '
        'ranges for raw RPN, restored RPN, GT, and final boxes per image.', '',
        '## Why final recall can exceed RPN proposal recall', '',
        'RPN recall asks whether a proposal already overlaps a GT at the '
        'specified IoU before RoI refinement. Final detection recall asks '
        'whether a one-to-one matched, class-consistent final box reaches '
        'IoU 0.5 after the Cascade RoI bbox heads decode and regress it. '
        'A proposal below 0.5 can therefore become a correct final box. '
        'The RPN and final metrics are different stages and must not be '
        'compared as if they were the same prediction set.', '',
        '## Attribution files', '',
        '- `rpn_to_roi_analysis.csv` contains one row per non-ignored GT. '
        '`best_final_iou` is diagnostic, while `final_hit_05` is taken from '
        'the existing one-to-one `match_image` assignment.',
        '- `rpn_to_roi_summary.csv` counts categories A-E: RPN>=0.5 and '
        'final TP; RPN in [0.3,0.5) and final TP; RPN<0.3 and final TP; '
        'RPN>=0.5 and final miss; and RPN<0.3 and final miss. The additional '
        'F bin retains [0.3,0.5) final misses outside the requested A-E '
        'partition.',
        '- Observed RPN-to-RoI category counts: '
        + ', '.join(f'{key}={category_counts[key]}'
                    for key in sorted(category_counts))
        + ('.' if category_counts else 'none (no proposals or normal GT).'),
        '- `score_threshold_analysis.csv` reuses the unchanged post-NMS '
        'prediction list at thresholds 0.1, 0.3, 0.5, and 0.75. It does '
        'not rerun NMS or modify model/evaluator settings.', '',
        '- `c_rpn_roi_cases.csv` contains up to 20 deterministic C-class '
        'cases, preferring one case per image. It includes the nearest RPN '
        'Proposal, the highest-IoU final prediction, and the actual matched '
        'TP prediction when those differ.', '',
        '## Limits', '',
        '- This report does not establish that the 68% versus 87% gap is a '
        'coordinate bug; the server output and coordinate ranges must be '
        'inspected first.',
        '- Proposal recall is limited by the configured RPN `max_per_img`; '
        f'the loaded value is `{rpn_max_per_img}`.',
        '- No numerical A-E or threshold result is claimed until this script '
        'is run on the server with the real dataset and checkpoint.',
    ]
    with open(osp.join(args.out_dir, 'rpn_roi_audit.md'), 'w',
              encoding='utf-8') as handle:
        handle.write('\n'.join(audit_lines) + '\n')


if __name__ == '__main__':
    main()
