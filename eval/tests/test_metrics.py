"""Synthetic-mask unit tests for metrics.py.

Run with:
    cd /scratch/users/sciget/isles26challenge
    source .venv-eval/bin/activate
    PYTHONPATH=. python -m pytest eval/tests -v
or simply:
    PYTHONPATH=. python eval/tests/test_metrics.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# Allow running as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.metrics import (
    boundary_metrics,
    evaluate_case,
    lesion_wise_metrics,
    overlap_metrics,
    size_binned_recall_voxels,
    volume_metrics,
)


VOXEL = (1.0, 1.0, 1.0)


def _empty(shape=(20, 20, 20)) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


def _cube(shape=(20, 20, 20), origin=(2, 2, 2), size=(4, 4, 4)) -> np.ndarray:
    a = np.zeros(shape, dtype=np.uint8)
    o = origin; s = size
    a[o[0]:o[0]+s[0], o[1]:o[1]+s[1], o[2]:o[2]+s[2]] = 1
    return a


def approx(x, y, tol=1e-6) -> bool:
    if math.isnan(x) and math.isnan(y):
        return True
    return abs(x - y) <= tol


# --- overlap ---

def test_overlap_perfect_match():
    m = _cube()
    om = overlap_metrics(m, m)
    assert om.dice == 1.0
    assert om.iou == 1.0
    assert om.sensitivity == 1.0
    assert om.precision == 1.0
    assert om.fp == 0 and om.fn == 0


def test_overlap_disjoint():
    a = _cube(origin=(2, 2, 2))
    b = _cube(origin=(12, 12, 12))
    om = overlap_metrics(a, b)
    assert om.dice == 0.0
    assert om.iou == 0.0
    assert om.tp == 0
    # Sensitivity = 0/64 = 0, Precision = 0/64 = 0
    assert om.sensitivity == 0.0
    assert om.precision == 0.0


def test_overlap_partial():
    # Two unit cubes of 4^3 = 64 voxels each, overlapping at a 2^3=8 corner.
    a = _cube(origin=(2, 2, 2), size=(4, 4, 4))
    b = _cube(origin=(4, 4, 4), size=(4, 4, 4))
    om = overlap_metrics(a, b)
    # |A∩B| = 2*2*2=8, |A|=|B|=64
    expected_dice = 2 * 8 / (64 + 64)
    expected_iou = 8 / (64 + 64 - 8)
    assert approx(om.dice, expected_dice)
    assert approx(om.iou, expected_iou)
    assert om.tp == 8
    assert om.fp == 56
    assert om.fn == 56


def test_overlap_empty_both():
    om = overlap_metrics(_empty(), _empty())
    assert om.dice == 1.0
    assert om.iou == 1.0
    assert om.tp == om.fp == om.fn == 0


def test_overlap_empty_pred_only():
    om = overlap_metrics(_empty(), _cube())
    assert om.dice == 0.0
    assert om.iou == 0.0
    assert om.sensitivity == 0.0
    assert math.isnan(om.precision)  # no predicted positives


def test_overlap_empty_gt_only():
    om = overlap_metrics(_cube(), _empty())
    assert om.dice == 0.0
    assert math.isnan(om.sensitivity)  # no GT positives
    assert om.precision == 0.0


# --- volume ---

def test_volume_match():
    m = _cube()
    vm = volume_metrics(m, m, VOXEL)
    assert approx(vm.abs_volume_diff_ml, 0.0)
    assert approx(vm.signed_volume_diff_ml, 0.0)
    assert approx(vm.relative_abs_volume_diff, 0.0)


def test_volume_scaling():
    a = _cube(size=(4, 4, 4))    # 64 voxels = 0.064 mL at 1 mm iso
    b = _cube(size=(2, 4, 4))    # 32 voxels
    vm = volume_metrics(a, b, (2.0, 2.0, 2.0))  # 8 mm^3 / voxel
    assert approx(vm.pred_vol_ml, 64 * 8 / 1000)
    assert approx(vm.gt_vol_ml, 32 * 8 / 1000)
    assert approx(vm.signed_volume_diff_ml, (64 - 32) * 8 / 1000)


# --- boundary ---

def test_boundary_perfect_match():
    m = _cube()
    bm = boundary_metrics(m, m, VOXEL)
    assert approx(bm.hd95_mm, 0.0)
    assert approx(bm.assd_mm, 0.0)
    assert approx(bm.surface_dice_3mm, 1.0)


def test_boundary_offset_by_one():
    a = _cube(origin=(2, 2, 2), size=(4, 4, 4))
    b = _cube(origin=(3, 2, 2), size=(4, 4, 4))  # shifted by 1 mm in x
    bm = boundary_metrics(a, b, VOXEL)
    # Distances should be on the order of 1 mm for the shifted boundary slab.
    assert bm.hd95_mm > 0.0
    assert bm.hd95_mm <= 1.5
    # 3-mm-tolerance surface dice should be 1.0 for a 1-voxel shift at 1mm iso.
    assert bm.surface_dice_3mm == 1.0


def test_boundary_empty_returns_nan():
    bm = boundary_metrics(_empty(), _cube(), VOXEL)
    assert math.isnan(bm.hd95_mm)
    assert math.isnan(bm.assd_mm)


# --- lesion-wise ---

def test_lesionwise_perfect_detection():
    a = _cube(origin=(2, 2, 2))
    b = _cube(origin=(2, 2, 2))
    lm = lesion_wise_metrics(a, b)
    assert lm.gt_lesion_count == 1
    assert lm.pred_lesion_count == 1
    assert lm.tp_lesions == 1
    assert lm.fp_lesions == 0
    assert lm.fn_lesions == 0
    assert lm.lesion_f1 == 1.0
    assert lm.signed_lesion_count_diff == 0


def test_lesionwise_two_gt_one_missed():
    gt = _empty((30, 30, 30))
    gt[2:4, 2:4, 2:4] = 1   # GT 1: 2x2x2
    gt[20:22, 20:22, 20:22] = 1  # GT 2: 2x2x2
    pred = _empty((30, 30, 30))
    pred[2:4, 2:4, 2:4] = 1  # matches GT 1, misses GT 2
    lm = lesion_wise_metrics(pred, gt)
    assert lm.gt_lesion_count == 2
    assert lm.pred_lesion_count == 1
    assert lm.tp_lesions == 1
    assert lm.fp_lesions == 0
    assert lm.fn_lesions == 1
    assert math.isclose(lm.lesion_recall, 0.5)
    assert math.isclose(lm.lesion_precision, 1.0)
    assert math.isclose(lm.lesion_f1, 2 * 0.5 * 1.0 / 1.5)


def test_lesionwise_one_gt_three_pred_fp():
    gt = _cube()
    pred = _empty()
    pred[2:6, 2:6, 2:6] = 1     # TP
    pred[10:12, 10:12, 10:12] = 1  # FP
    pred[15:17, 15:17, 15:17] = 1  # FP
    lm = lesion_wise_metrics(pred, gt)
    assert lm.gt_lesion_count == 1
    assert lm.pred_lesion_count == 3
    assert lm.tp_lesions == 1
    assert lm.fp_lesions == 2
    assert math.isclose(lm.lesion_recall, 1.0)
    assert math.isclose(lm.lesion_precision, 1/3)


def test_lesionwise_empty_both():
    lm = lesion_wise_metrics(_empty(), _empty())
    assert lm.gt_lesion_count == 0
    assert lm.pred_lesion_count == 0
    assert lm.lesion_f1 == 1.0


def test_lesionwise_empty_pred_only():
    lm = lesion_wise_metrics(_empty(), _cube())
    assert lm.lesion_recall == 0.0


def test_lesionwise_empty_gt_only():
    lm = lesion_wise_metrics(_cube(), _empty())
    assert lm.lesion_precision == 0.0


def test_lesionwise_one_pred_overlaps_two_gt():
    """Many-to-many caveat (Codex review #2): one predicted CC that overlaps
    two GT CCs gets recall credit on both. Both GT lesions count as detected;
    the single prediction counts as one TP."""
    gt = _empty((30, 30, 30))
    gt[2:5, 2:5, 2:5] = 1            # GT 1
    gt[6:9, 2:5, 2:5] = 1            # GT 2 (separated by 1 voxel gap on x axis)
    # Confirm GT is two components
    pred = _empty((30, 30, 30))
    pred[2:9, 2:5, 2:5] = 1          # one big pred bridging GT 1 and GT 2
    lm = lesion_wise_metrics(pred, gt)
    assert lm.gt_lesion_count == 2
    assert lm.pred_lesion_count == 1
    # Both GTs detected with any-overlap rule:
    assert lm.tp_lesions == 2
    assert lm.fn_lesions == 0
    # Pred is 1 TP (no FPs):
    assert lm.fp_lesions == 0
    assert math.isclose(lm.lesion_recall, 1.0)
    assert math.isclose(lm.lesion_precision, 1.0)
    # |count diff| surfaces the merge:
    assert lm.signed_lesion_count_diff == -1  # pred missed one component
    assert lm.abs_lesion_count_diff == 1


def test_lesionwise_two_pred_overlap_one_gt():
    """Inverse: one big GT lesion split into two predictions. Recall on the
    single GT is 1; both predictions count as TPs (precision = 1). |count
    diff| surfaces the split."""
    gt = _empty((30, 30, 30))
    gt[2:9, 2:5, 2:5] = 1            # one big GT
    pred = _empty((30, 30, 30))
    pred[2:5, 2:5, 2:5] = 1          # pred 1
    pred[6:9, 2:5, 2:5] = 1          # pred 2
    lm = lesion_wise_metrics(pred, gt)
    assert lm.gt_lesion_count == 1
    assert lm.pred_lesion_count == 2
    assert lm.tp_lesions == 1
    assert lm.fp_lesions == 0        # both predictions overlap the GT
    assert math.isclose(lm.lesion_recall, 1.0)
    assert math.isclose(lm.lesion_precision, 1.0)
    assert lm.signed_lesion_count_diff == 1
    assert lm.abs_lesion_count_diff == 1


def test_lesionwise_rejects_zero_min_overlap():
    """Codex review #1: min_overlap_voxels < 1 silently credits disjoint
    components and must be rejected at the API."""
    raised = False
    try:
        lesion_wise_metrics(_cube(), _cube(), min_overlap_voxels=0)
    except ValueError:
        raised = True
    assert raised, "expected ValueError for min_overlap_voxels=0"


def test_validates_shape_mismatch():
    a = np.zeros((10, 10, 10), dtype=np.uint8)
    b = np.zeros((10, 10, 11), dtype=np.uint8)
    for fn in (overlap_metrics, volume_metrics):
        raised = False
        try:
            if fn is volume_metrics:
                fn(a, b, VOXEL)
            else:
                fn(a, b)
        except ValueError:
            raised = True
        assert raised, f"{fn.__name__} did not raise on shape mismatch"


def test_rejects_nan_in_float_mask():
    a = np.zeros((10, 10, 10), dtype=np.float32)
    a[0, 0, 0] = np.nan
    b = np.zeros((10, 10, 10), dtype=np.uint8)
    raised = False
    try:
        overlap_metrics(a, b)
    except ValueError:
        raised = True
    assert raised, "expected ValueError on NaN-bearing float mask"


# --- size-binned recall ---

def test_size_binned_recall_small_missed():
    gt = _empty((40, 40, 40))
    # one tiny lesion (1 voxel) + one big lesion (10x10x10 = 1000 voxels)
    gt[1, 1, 1] = 1
    gt[20:30, 20:30, 20:30] = 1
    pred = _empty((40, 40, 40))
    pred[20:30, 20:30, 20:30] = 1  # only the big one detected
    lm = lesion_wise_metrics(pred, gt)
    bins = size_binned_recall_voxels(lm)
    # tiny bin should have n=1, detected=0
    assert bins["tiny_<10"]["n_gt"] == 1
    assert bins["tiny_<10"]["n_detected"] == 0
    # the big lesion falls into the medium 100..1000 bin (1000 voxels boundary
    # is exclusive at the high end). Let's check the bin that contains it.
    detected_bin = next(name for name, info in bins.items() if info["n_detected"] == 1)
    assert detected_bin in {"medium_100-1k", "large_1k-10k"}
    assert bins[detected_bin]["recall"] == 1.0


# --- evaluate_case end-to-end ---

def test_evaluate_case_smoke():
    gt = _cube()
    pred = _cube()
    cm = evaluate_case(pred, gt, VOXEL, subject_id="test")
    d = cm.to_dict()
    assert d["overlap"]["dice"] == 1.0
    assert d["lesionwise"]["lesion_f1"] == 1.0
    assert d["volume"]["abs_volume_diff_ml"] == 0.0


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERR  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(_run_all())
