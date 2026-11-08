"""Synthetic tests for the standalone AuxDet error attribution logic."""
import numpy as np

from tools.analysis_tools.analyze_auxdet_errors import (
    iou_matrix, match_image, proposal_boxes_original, rpn_to_roi_rows,
    threshold_summary)


def gt(box, label=0, ignore=False):
    x1, y1, x2, y2 = box
    return dict(bbox=np.asarray(box, dtype=np.float32), label=label,
                ignore=ignore, width=x2 - x1, height=y2 - y1,
                area=(x2 - x1) * (y2 - y1), size_group='small', index=0)


def pred(box, score=.9, label=0, index=0):
    return dict(bbox=np.asarray(box, dtype=np.float32), score=score,
                label=label, index=index)


def test_exact_match_and_one_to_one_duplicate():
    truth = [gt((0, 0, 9, 9))]
    result = match_image(truth, [pred((0, 0, 9, 9), index=0),
                                 pred((0, 0, 9, 9), score=.8, index=1)])
    assert result['tp'] == 1
    assert result['fp'] == 1
    assert result['counts']['duplicate'] == 1


def test_localization_and_missed_gt():
    truth = [gt((0, 0, 9, 9))]
    localization = match_image(truth, [pred((0, 0, 3, 9))])
    assert localization['counts']['localization_fp'] == 1
    assert localization['gts'][0]['status'] == 'localization_miss'
    missed = match_image(truth, [pred((30, 30, 39, 39))])
    assert missed['counts']['background_fp'] == 1
    assert missed['gts'][0]['status'] == 'missed'


def test_empty_gt_and_empty_predictions():
    assert match_image([], [pred((0, 0, 4, 4))])['counts']['background_fp'] == 1
    assert match_image([gt((0, 0, 4, 4))], [])['fn'] == 1


def test_multiple_targets_and_labels_are_not_cross_matched():
    truth = [gt((0, 0, 9, 9), label=0), gt((20, 20, 29, 29), label=0)]
    result = match_image(truth, [pred((0, 0, 9, 9), index=0),
                                 pred((20, 20, 29, 29), index=1)])
    assert result['tp'] == 2 and result['fn'] == 0
    wrong_label = match_image([gt((0, 0, 9, 9), label=0)],
                              [pred((0, 0, 9, 9), label=1)])
    assert wrong_label['tp'] == 0 and wrong_label['fn'] == 1


def test_ignored_gt_is_not_tp_or_fp():
    result = match_image([gt((0, 0, 9, 9), ignore=True)],
                         [pred((0, 0, 9, 9))])
    assert result['tp'] == 0 and result['fp'] == 0
    assert result['ignored_pred'] == 1


def test_legacy_and_nonlegacy_coordinate_modes_are_explicit():
    boxes = np.asarray([[0, 0, 0, 0]], dtype=np.float32)
    assert iou_matrix(boxes, boxes, legacy_coordinate=True)[0, 0] == 1
    assert iou_matrix(boxes, boxes, legacy_coordinate=False)[0, 0] == 0


def test_scaled_gt_is_converted_before_matching():
    # A 2x transformed GT is converted by the caller before matching; the
    # matcher itself never silently mixes resized and original coordinates.
    original = gt((10, 10, 19, 19))
    resized_box = np.asarray(original['bbox']) * 2
    converted = gt(resized_box / 2)
    result = match_image([converted], [pred((10, 10, 19, 19))])
    assert result['tp'] == 1 and result['fn'] == 0


def test_rpn_scale_direction_and_final_regression_attribution():
    # RPN is in 2x resized coordinates; the audit converts a copy back to XML
    # coordinates before comparing it with GT.
    proposal = {'bboxes': np.asarray([[20, 20, 40, 40]], dtype=np.float32)}
    restored = proposal_boxes_original(
        proposal, {'scale_factor': np.asarray([2, 2, 2, 2], dtype=np.float32)})
    np.testing.assert_allclose(restored, [[10, 10, 20, 20]])

    truth = [gt((10, 10, 20, 20))]
    # Proposal IoU is exactly 1 here, while the final regression is a TP.
    result = match_image(truth, [pred((10, 10, 20, 20))])
    rows = rpn_to_roi_rows(truth,
                           [dict(bbox=restored[0], score=.9)], result,
                           'img-1')
    assert rows[0]['rpn_hit_05'] and rows[0]['final_hit_05']
    assert rows[0]['category'] == 'A_rpn05_final_tp'


def test_rpn_low_iou_can_become_final_tp_and_high_iou_can_miss():
    truth = [gt((0, 0, 9, 9)), gt((30, 30, 39, 39))]
    # First proposal is disjoint (<.3), but the final box is a TP (category C).
    # Second proposal is a good RPN box, while its final prediction misses (D).
    proposals = [dict(bbox=np.asarray([[50, 50, 59, 59]], dtype=np.float32)[0], score=.9),
                 dict(bbox=np.asarray([[30, 30, 39, 39]], dtype=np.float32)[0], score=.8)]
    result = match_image(truth, [pred((0, 0, 9, 9), index=0)])
    rows = rpn_to_roi_rows(truth, proposals, result, 'img-2')
    assert rows[0]['category'] == 'C_rpn_lt03_final_tp'
    assert rows[1]['category'] == 'D_rpn05_final_miss'


def test_score_threshold_summary_reuses_predictions_without_new_nms():
    truth = [gt((0, 0, 9, 9))]
    image = dict(gt=truth,
                 raw_predictions=[pred((0, 0, 9, 9), score=.8, index=0),
                                  pred((80, 80, 89, 89), score=.2, index=1)],
                 predictions=[], score_thr=.001)
    low = threshold_summary([image], .1)
    high = threshold_summary([image], .75)
    assert low['tp'] == high['tp'] == 1
    assert low['background_fp'] == 1 and high['background_fp'] == 0
    assert low['recall'] == high['recall'] == 1.0
