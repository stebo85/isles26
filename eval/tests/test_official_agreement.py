"""Pin our diagnostic harness against the ISLES'26 OFFICIAL scorer.

`eval/metrics.py` and `eval/official_metrics.py` answer different questions and
they are *supposed* to disagree on some phantoms. Until now nothing in the test
suite said where the line is, so `eval/tests/test_metrics.py` looked like it was
validating the challenge metric when it was only validating our own lenient
diagnostic. This module is the missing arbiter: for every phantom it states
either "these must agree" or "these must differ, by exactly this much".

The official numbers are never recomputed here -- they come from
`eval/official_metrics.py`, which calls the organizers' `utils/eval_utils.py`
verbatim. So a change in panoptica, in the organizers' code, or in our wrapper
all show up as a test failure rather than as a leaderboard surprise.

Three official semantics that are cheap to get wrong are pinned explicitly,
because postprocessing decisions elsewhere in the repo depend on them:

  * matching is ONE-TO-ONE at IoU >= 0.25 (not Dice >= 0.25, not any-overlap),
    so merging/splitting lesions costs half a lesion each way in RQ;
  * `global_bin_dsc` is plain binary Dice only when at least one instance pair
    matched -- with zero matched pairs panoptica's edge-case handler returns
    0.0, discarding real voxel overlap;
  * PR-AUC returns 1.0 on a lesion-free case only if the soft map is perfectly
    constant, and on a lesioned case a constant soft map still scores
    (1 + prevalence) / 2, not 0.

panoptica and scikit-learn live only in `.venv-official`, so this module SKIPS
(exit 0, loud message) when those two packages are simply absent -- but it FAILS
loudly when they are installed and the import still breaks (broken upstream
clone or wrapper), and of course when the numbers moved. Run it with:

    cd /scratch/users/sciget/isles26challenge
    PYTHONPATH=. .venv-official/bin/python eval/tests/test_official_agreement.py
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

# Allow running as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.metrics import (
    connected_components,
    lesion_wise_metrics,
    overlap_metrics,
    volume_metrics,
)


def _official_stack_is_installed() -> bool:
    """True when panoptica AND scikit-learn are importable in this interpreter.

    This is what separates the two ways the import below can fail: "wrong venv"
    (skip, exit 0) from "the official stack is right here but the upstream clone
    or our wrapper is broken" (fail). Without the distinction, deleting
    work/isles26_upstream or breaking eval/official_metrics.py would turn this
    whole module green -- the exact silent-pass failure it exists to prevent.
    """
    try:
        return all(importlib.util.find_spec(m) is not None for m in ("panoptica", "sklearn"))
    except (ImportError, ValueError):
        return False


# `eval.official_metrics` imports the organizers' code at module import time
# (it inserts work/isles26_upstream on sys.path), so a missing clone or a
# missing panoptica/sklearn shows up right here.
OFFICIAL_IMPORT_ERROR: Exception | None = None
try:
    from eval.official_metrics import (  # noqa: E402
        OFFICIAL_SOURCE,
        flat_soft_pr_auc,
        score_masks,
        score_soft,
    )
except Exception as exc:  # pragma: no cover - exercised only without .venv-official
    OFFICIAL_IMPORT_ERROR = exc
    if _official_stack_is_installed():
        raise RuntimeError(
            "eval.official_metrics failed to import even though panoptica and "
            "scikit-learn are installed, so this is not a wrong-venv problem: the "
            f"upstream clone or the wrapper is broken ({type(exc).__name__}: {exc})"
        ) from exc
    # `_run_all` handles the skip for the plain-script path; this handles the
    # pytest path, where every test would otherwise error on a missing name.
    try:
        import pytest

        pytest.skip(
            f"official evaluation stack not importable ({exc}); run this module "
            "with .venv-official/bin/python",
            allow_module_level=True,
        )
    except ImportError:
        pass


TOL = 1e-9


def _cube(shape=(20, 20, 20), origin=(2, 2, 2), size=(4, 4, 4)) -> np.ndarray:
    a = np.zeros(shape, dtype=np.uint8)
    o, s = origin, size
    a[o[0]:o[0]+s[0], o[1]:o[1]+s[1], o[2]:o[2]+s[2]] = 1
    return a


def _empty(shape=(20, 20, 20)) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


def _merge_phantom() -> tuple[np.ndarray, np.ndarray]:
    """Two 3^3 GT lesions one voxel apart, bridged by a single 7x3x3 prediction."""
    gt = _empty((30, 30, 30))
    gt[2:5, 2:5, 2:5] = 1
    gt[6:9, 2:5, 2:5] = 1  # 1-voxel gap on x, so two components under 26-conn
    pred = _empty((30, 30, 30))
    pred[2:9, 2:5, 2:5] = 1
    return gt, pred


def _low_iou_high_dice_pair() -> tuple[np.ndarray, np.ndarray]:
    """Two 100-voxel slabs overlapping in 30 voxels: IoU 0.176, Dice 0.30.

    The only phantom that separates "threshold on IoU" from "threshold on Dice".
    """
    gt = _empty((40, 40, 40))
    pred = _empty((40, 40, 40))
    gt[10:20, 10:20, 10] = 1
    pred[17:27, 10:20, 10] = 1  # shifted by 7 -> 3 rows of 10x10 overlap
    return gt, pred


def approx(x: float, y: float, tol: float = TOL) -> bool:
    if math.isnan(x) and math.isnan(y):
        return True
    return abs(x - y) <= tol


def _voxel_volume_ml(voxel_size_mm: tuple[float, float, float]) -> float:
    return float(np.prod(voxel_size_mm)) / 1000.0


VOXEL = (1.0, 1.0, 1.0)
VOXEL_ML = _voxel_volume_ml(VOXEL)


# --- where the two harnesses MUST agree -------------------------------------

def test_official_source_is_the_upstream_clone():
    assert OFFICIAL_SOURCE.endswith("work/isles26_upstream/utils/eval_utils.py"), OFFICIAL_SOURCE


def test_connectivity_is_26_on_both_sides():
    """Lesion COUNTS can only agree if both sides call the same blob one blob.
    Two corner-touching voxels are one component at 26-connectivity, two at 6."""
    vol = _empty((16, 16, 16))
    vol[4, 4, 4] = 1
    vol[5, 5, 5] = 1  # corner-adjacent only
    _lab, n_ours = connected_components(vol)
    assert n_ours == 1, f"our labelling is not 26-connected: {n_ours} components"
    # With an empty prediction, |num_ref - num_pred| == num_ref_instances.
    _dice, _avd, count_diff, _rq = score_masks(vol, _empty((16, 16, 16)), VOXEL_ML)
    assert count_diff == 1, f"panoptica is not 26-connected: {count_diff} reference instances"


def test_perfect_prediction_agrees():
    m = _cube()
    dice, avd, count_diff, rq = score_masks(m, m, VOXEL_ML)
    ours_overlap = overlap_metrics(m, m)
    ours_lesion = lesion_wise_metrics(m, m)
    assert approx(dice, 1.0) and approx(ours_overlap.dice, 1.0)
    assert approx(avd, 0.0)
    assert count_diff == 0 == ours_lesion.abs_lesion_count_diff
    assert approx(rq, 1.0) and approx(ours_lesion.lesion_f1, 1.0)


def test_disjoint_prediction_agrees():
    gt = _cube(origin=(2, 2, 2))
    pred = _cube(origin=(12, 12, 12))
    dice, _avd, count_diff, rq = score_masks(gt, pred, VOXEL_ML)
    ours_overlap = overlap_metrics(pred, gt)
    ours_lesion = lesion_wise_metrics(pred, gt)
    assert approx(dice, 0.0) and approx(ours_overlap.dice, 0.0)
    assert count_diff == 0 == ours_lesion.abs_lesion_count_diff  # 1 vs 1
    # No overlap at all: lenient any-overlap and strict IoU>=0.25 both say 0.
    assert approx(rq, 0.0) and approx(ours_lesion.lesion_f1, 0.0)


def test_partial_overlap_above_iou_threshold_agrees():
    """Two 4^3 cubes offset by one voxel: IoU 48/80 = 0.6, comfortably matched."""
    gt = _cube(origin=(2, 2, 2))
    pred = _cube(origin=(3, 2, 2))
    dice, _avd, count_diff, rq = score_masks(gt, pred, VOXEL_ML)
    ours_overlap = overlap_metrics(pred, gt)
    ours_lesion = lesion_wise_metrics(pred, gt)
    assert approx(dice, 2 * 48 / (64 + 64))
    assert approx(ours_overlap.dice, dice)
    assert count_diff == 0 == ours_lesion.abs_lesion_count_diff
    assert approx(rq, 1.0) and approx(ours_lesion.lesion_f1, 1.0)


def test_empty_gt_and_empty_pred_agree():
    z = _empty()
    dice, avd, count_diff, rq = score_masks(z, z, VOXEL_ML)
    ours_overlap = overlap_metrics(z, z)
    ours_lesion = lesion_wise_metrics(z, z)
    # eval_utils short-circuits both-empty to empty_value=1.0 for Dice and F1.
    assert approx(dice, 1.0) and approx(ours_overlap.dice, 1.0)
    assert approx(avd, 0.0)
    assert count_diff == 0 == ours_lesion.abs_lesion_count_diff
    assert approx(rq, 1.0) and approx(ours_lesion.lesion_f1, 1.0)


def test_empty_gt_nonempty_pred_agrees():
    gt = _empty()
    pred = _cube()
    dice, avd, count_diff, rq = score_masks(gt, pred, VOXEL_ML)
    ours_overlap = overlap_metrics(pred, gt)
    ours_lesion = lesion_wise_metrics(pred, gt)
    assert approx(dice, 0.0) and approx(ours_overlap.dice, 0.0)
    assert approx(avd, 64 * VOXEL_ML)
    assert count_diff == 1 == ours_lesion.abs_lesion_count_diff
    assert approx(rq, 0.0) and approx(ours_lesion.lesion_f1, 0.0)


def test_nonempty_gt_empty_pred_agrees():
    gt = _cube()
    pred = _empty()
    dice, avd, count_diff, rq = score_masks(gt, pred, VOXEL_ML)
    ours_overlap = overlap_metrics(pred, gt)
    ours_lesion = lesion_wise_metrics(pred, gt)
    assert approx(dice, 0.0) and approx(ours_overlap.dice, 0.0)
    assert approx(avd, 64 * VOXEL_ML)
    assert count_diff == 1 == ours_lesion.abs_lesion_count_diff
    assert approx(rq, 0.0) and approx(ours_lesion.lesion_f1, 0.0)


def test_anisotropic_voxel_volume_agrees():
    """Volume is the one metric that depends on the header, and every phantom
    above is 1 mm isotropic -- where a mm^3-vs-mL or a per-axis-vs-product slip
    is invisible. 0.5x0.5x3 mm (0.75 mm^3, a realistic clinical anisotropic
    acquisition) makes both visible."""
    voxel = (0.5, 0.5, 3.0)
    vox_ml = _voxel_volume_ml(voxel)
    assert approx(vox_ml, 0.00075), vox_ml
    gt = _cube(size=(4, 4, 4))       # 64 voxels
    pred = _cube(size=(2, 4, 4))     # 32 voxels
    _dice, avd, _count_diff, _rq = score_masks(gt, pred, vox_ml)
    ours = volume_metrics(pred, gt, voxel)
    assert approx(avd, 32 * vox_ml), avd
    assert approx(ours.abs_volume_diff_ml, avd), (ours.abs_volume_diff_ml, avd)
    # Sanity: the same 32-voxel difference at 1 mm iso must scale by exactly
    # the voxel-volume ratio 1 / 0.75.
    _d, avd_iso, _c, _r = score_masks(gt, pred, VOXEL_ML)
    assert approx(avd_iso / avd, 4.0 / 3.0), (avd_iso, avd)


# --- where the two harnesses MUST diverge (documented, intentional) ----------

def test_merge_phantom_lenient_f1_beats_official_rq():
    """One predicted component bridging two GT lesions.

    Ours (any-overlap, many-to-many): both GT lesions are "detected" and the
    single prediction is a TP -> precision = recall = F1 = 1.0.
    Official (one-to-one at IoU >= 0.25): the bridge matches ONE reference,
    the other reference is an unmatched FN -> TP=1, FP=0, FN=1,
    RQ = 1 / (1 + 0.5*0 + 0.5*1) = 2/3.

    The 0.333 gap is the whole reason this file exists: our harness scores a
    merge failure as perfect detection, the leaderboard does not.
    """
    gt, pred = _merge_phantom()
    dice, _avd, count_diff, rq = score_masks(gt, pred, VOXEL_ML)
    ours_lesion = lesion_wise_metrics(pred, gt)
    ours_overlap = overlap_metrics(pred, gt)

    assert ours_lesion.gt_lesion_count == 2 and ours_lesion.pred_lesion_count == 1
    assert approx(ours_lesion.lesion_recall, 1.0)
    assert approx(ours_lesion.lesion_precision, 1.0)
    assert approx(ours_lesion.lesion_f1, 1.0)
    assert approx(rq, 2.0 / 3.0)
    assert approx(ours_lesion.lesion_f1 - rq, 1.0 / 3.0)
    # Both IoUs (27/63 = 0.43) clear the threshold; only one match is allowed.
    assert 27 / 63 > 0.25
    # Dice and |count diff| do agree -- they are matching-independent here
    # (at least one instance pair matched, see the tp==0 test below).
    assert approx(dice, ours_overlap.dice)
    assert count_diff == 1 == ours_lesion.abs_lesion_count_diff


def test_split_phantom_lenient_f1_beats_official_rq():
    """The mirror image: one GT lesion covered by two predicted components.

    Ours: the GT is detected and both predictions overlap it -> F1 = 1.0.
    Official: one prediction matches, the other is an unmatched FP ->
    TP=1, FP=1, FN=0, RQ = 1 / (1 + 0.5) = 2/3.

    Same 0.333 gap, and it is why min-connected-component pruning can pay off
    on the official scorer while looking neutral on ours.
    """
    gt, pred = _merge_phantom()
    gt, pred = pred, gt  # swap: the bridge becomes the GT, the two blobs the pred
    dice, _avd, count_diff, rq = score_masks(gt, pred, VOXEL_ML)
    ours_lesion = lesion_wise_metrics(pred, gt)
    ours_overlap = overlap_metrics(pred, gt)

    assert ours_lesion.gt_lesion_count == 1 and ours_lesion.pred_lesion_count == 2
    assert approx(ours_lesion.lesion_recall, 1.0)
    assert approx(ours_lesion.lesion_precision, 1.0)
    assert approx(ours_lesion.lesion_f1, 1.0)
    assert approx(rq, 2.0 / 3.0)
    assert approx(ours_lesion.lesion_f1 - rq, 1.0 / 3.0)
    assert approx(dice, ours_overlap.dice)
    assert count_diff == 1 == ours_lesion.abs_lesion_count_diff


def test_official_threshold_is_on_iou_not_dice():
    """IoU 0.176 < 0.25 < Dice 0.30. If the threshold were on Dice this pair
    would match and RQ would be 1.0; it is 0.0, so the threshold is on IoU."""
    gt, pred = _low_iou_high_dice_pair()
    inter = int(np.logical_and(gt, pred).sum())
    iou = inter / (int(gt.sum()) + int(pred.sum()) - inter)
    ours_overlap = overlap_metrics(pred, gt)
    assert approx(iou, 30 / 170) and iou < 0.25
    assert approx(ours_overlap.dice, 0.30) and ours_overlap.dice > 0.25

    _dice, _avd, count_diff, rq = score_masks(gt, pred, VOXEL_ML)
    assert approx(rq, 0.0), "pair matched => threshold is on Dice, not IoU"
    # Our lenient rule credits the same pair fully.
    ours_lesion = lesion_wise_metrics(pred, gt)
    assert approx(ours_lesion.lesion_f1, 1.0)
    assert count_diff == 0 == ours_lesion.abs_lesion_count_diff


def test_official_dice_collapses_to_zero_when_no_instance_matches():
    """`global_bin_dsc` is plain binary Dice ONLY when at least one instance
    pair matched. panoptica's zero-TP edge-case handler short-circuits it to
    0.0 otherwise, so 30 voxels of real overlap score the same as none.

    Consequence for us: the official Dice is not a purely voxel-wise metric,
    and any threshold/min-CC sweep tuned on our voxel Dice can be misleading
    for exactly the near-miss cases we most want to fix.
    """
    gt, pred = _low_iou_high_dice_pair()
    dice, _avd, _count_diff, rq = score_masks(gt, pred, VOXEL_ML)
    assert approx(rq, 0.0)
    assert approx(dice, 0.0), "expected the zero-TP edge case to zero out Dice"
    assert approx(overlap_metrics(pred, gt).dice, 0.30)

    # Add a second, well-matched lesion pair far away: now tp == 1, the edge
    # case does not fire, and Dice is plain binary Dice over BOTH pairs again.
    gt2, pred2 = gt.copy(), pred.copy()
    gt2[10:20, 10:20, 30] = 1
    pred2[10:20, 10:20, 30] = 1
    dice2, _avd2, _cd2, rq2 = score_masks(gt2, pred2, VOXEL_ML)
    assert approx(dice2, overlap_metrics(pred2, gt2).dice)
    assert approx(rq2, 0.5)  # TP=1, FP=1, FN=1 -> 1 / (1 + 0.5 + 0.5)


# --- PR-AUC (the only official metric computed on the soft map) --------------

def test_pr_auc_perfect_ranking_is_one():
    gt = _cube()
    rng = np.random.default_rng(0)
    # Every lesion voxel scores above every background voxel.
    soft = np.where(gt > 0, 0.6 + 0.4 * rng.random(gt.shape), 0.4 * rng.random(gt.shape))
    assert approx(score_soft(gt, soft.astype(np.float32)), 1.0)


def test_pr_auc_is_invariant_under_a_strictly_monotone_transform():
    """PR-AUC only sees the ranking, so recalibrating the soft map cannot buy
    (or cost) anything -- only the binary threshold moves the other four."""
    gt = _cube()
    rng = np.random.default_rng(1)
    soft = (0.1 + 0.8 * rng.random(gt.shape) + 0.1 * gt).astype(np.float32)
    base = score_soft(gt, soft)
    for name, transformed in (
        ("affine", 2.0 * soft + 1.0),
        ("exp", np.exp(soft)),
        ("logit-ish", np.log(soft / (1.0 - soft))),
    ):
        assert approx(score_soft(gt, transformed.astype(np.float32)), base, tol=1e-6), name


def test_pr_auc_empty_gt_varying_map_is_zero():
    """The expensive branch: on a lesion-free case ANY variation in the soft
    map scores 0.0, even if the binary prediction after thresholding is empty."""
    gt = _empty()
    rng = np.random.default_rng(2)
    assert approx(score_soft(gt, rng.random(gt.shape).astype(np.float32)), 0.0)
    # Even a single dissenting voxel is enough.
    almost_flat = np.zeros(gt.shape, dtype=np.float32)
    almost_flat[0, 0, 0] = 1e-6
    assert approx(score_soft(gt, almost_flat), 0.0)


def test_pr_auc_empty_gt_constant_map_is_one():
    gt = _empty()
    for value in (0.0, 0.7, 1.0):
        soft = np.full(gt.shape, value, dtype=np.float32)
        assert approx(score_soft(gt, soft), 1.0), value


def test_flat_soft_map_on_a_lesioned_case_scores_half_of_one_plus_prevalence():
    """What the "flatten the soft map when the binary prediction is empty"
    trick actually costs on a case that DOES have a lesion: the PR curve
    degenerates to (recall=1, precision=prevalence) and (recall=0,
    precision=1), whose trapezoid is (1 + prevalence) / 2 -- i.e. ~0.5, far
    better than the 0.0 an unlucky varying map would get on a lesion-free case.
    """
    gt = _cube()  # 64 of 8000 voxels
    prevalence = float(gt.sum()) / gt.size
    expected = (1.0 + prevalence) / 2.0
    assert approx(flat_soft_pr_auc(gt), expected, tol=1e-6)
    assert approx(expected, 0.504, tol=1e-9)


def _run_all() -> int:
    if OFFICIAL_IMPORT_ERROR is not None:
        print(
            "SKIP eval/tests/test_official_agreement.py: the official stack is not "
            f"importable here ({type(OFFICIAL_IMPORT_ERROR).__name__}: {OFFICIAL_IMPORT_ERROR}).\n"
            "      panoptica + scikit-learn live only in .venv-official; run with\n"
            "      PYTHONPATH=. .venv-official/bin/python eval/tests/test_official_agreement.py"
        )
        return 0
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
