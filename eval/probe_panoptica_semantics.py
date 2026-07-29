#!/usr/bin/env python3
"""Pin down the exact semantics of the ISLES'26 official evaluation code.

The audit had to *assume* several things about
`work/isles26_upstream/utils/eval_utils.py` that materially change every
lesion-count and lesion-F1 number in this repo. This script measures them
instead of assuming, using synthetic phantoms with known answers:

  1. Connected-component CONNECTIVITY used by
     `ConnectedComponentsInstanceApproximator` -- 26 (what `eval/metrics.py`
     assumes) or 6? If it is 6, every predicted/reference component count in the
     repo is wrong and the min-component-size sweep must be re-derived.
  2. Whether `NaiveThresholdMatching(matching_threshold=0.25)` thresholds on
     IoU or on Dice. A pair with IoU 0.176 / Dice 0.30 separates the two.
  3. Argument ORDER of `Panoptica_Evaluator.evaluate` -- the official wrapper
     calls `evaluate(prediction, ground_truth)`, so if panoptica's signature is
     `(reference, prediction)` the roles are silently swapped.
  4. That `result.rq` is the panoptic Recognition Quality
     TP / (TP + 0.5*FP + 0.5*FN) over MATCHED instances.
  5. That `result.global_bin_dsc` is plain binary Dice.
  6. The PR-AUC empty-ground-truth branch: 1.0 only for a perfectly constant
     soft map, else 0.0.

Run inside `.venv-official` with PYTHONPATH pointing at `work/isles26_upstream`.
Exits non-zero if any invariant the repo depends on is violated.
"""

from __future__ import annotations

import sys

import numpy as np

from utils.eval_utils import (  # type: ignore[import-not-found]
    compute_absolute_volume_difference,
    compute_dice_f1_instance_difference,
    compute_pr_auc,
)

failures: list[str] = []
notes: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not ok:
        failures.append(f"{name}: {detail}")


# --------------------------------------------------------------------------
# 1. Connectivity of the instance approximator.
#    Two single voxels touching only at a CORNER are one component under 26-
#    connectivity and two components under 6-connectivity.
# --------------------------------------------------------------------------
vol = np.zeros((16, 16, 16), dtype=np.uint8)
vol[4, 4, 4] = 1
vol[5, 5, 5] = 1  # corner-adjacent only
_f1, count_diff, _dice = compute_dice_f1_instance_difference(
    ground_truth=vol, prediction=np.zeros_like(vol)
)
# prediction is empty -> num_pred_instances == 0, so |diff| == num_ref_instances
n_ref_corner = count_diff
connectivity = {1: 26, 2: 6}.get(int(n_ref_corner))
print(
    f"[INFO] corner-adjacent voxel pair -> {n_ref_corner} reference instance(s) "
    f"=> connectivity looks like {connectivity}"
)
check(
    "connectivity is 26 (matches eval/metrics.py)",
    int(n_ref_corner) == 1,
    f"got {n_ref_corner} instances for a corner-adjacent pair "
    f"(1 => 26-connectivity, 2 => 6-connectivity)",
)

# Face-adjacent pair must always be a single component under either convention.
vol2 = np.zeros((16, 16, 16), dtype=np.uint8)
vol2[4, 4, 4] = 1
vol2[4, 4, 5] = 1
_f1, n_ref_face, _dice = compute_dice_f1_instance_difference(
    ground_truth=vol2, prediction=np.zeros_like(vol2)
)
check("face-adjacent pair is one component", int(n_ref_face) == 1, f"got {n_ref_face}")


# --------------------------------------------------------------------------
# 2. Is matching_threshold=0.25 on IoU or on Dice?
#    Blocks of 100 voxels each overlapping in 30 voxels:
#        IoU  = 30 / 170 = 0.176  (< 0.25 -> NOT matched if IoU)
#        Dice = 2*30/200 = 0.30   (> 0.25 -> matched if Dice)
# --------------------------------------------------------------------------
shape = (40, 40, 40)
ref = np.zeros(shape, dtype=np.uint8)
pred = np.zeros(shape, dtype=np.uint8)
# 100 voxels each: a 10x10x1 slab.
ref[10:20, 10:20, 10] = 1
# shift along axis 0 by 7 -> 3 rows of 10x10 overlap = 30 voxels
pred[17:27, 10:20, 10] = 1
assert ref.sum() == 100 and pred.sum() == 100
inter = int(np.logical_and(ref, pred).sum())
iou = inter / (ref.sum() + pred.sum() - inter)
dsc = 2 * inter / (ref.sum() + pred.sum())
f1_probe, _cd, _d = compute_dice_f1_instance_difference(ground_truth=ref, prediction=pred)
print(f"[INFO] probe pair: overlap={inter} IoU={iou:.4f} Dice={dsc:.4f} -> rq={f1_probe}")
# One ref, one pred. If matched: TP=1,FP=0,FN=0 -> rq=1.0.
# If not matched: TP=0,FP=1,FN=1 -> rq=0.0.
matched = float(f1_probe) > 0.5
check(
    "matching_threshold is on IoU (not Dice)",
    not matched,
    f"IoU={iou:.4f} < 0.25 < Dice={dsc:.4f}; rq={f1_probe} "
    f"({'MATCHED => threshold is on Dice' if matched else 'unmatched => threshold is on IoU'})",
)

# Sanity: a pair comfortably above IoU 0.25 must match.
pred_close = np.zeros(shape, dtype=np.uint8)
pred_close[12:22, 10:20, 10] = 1  # 80/120 overlap -> IoU 0.667
f1_close, _cd, _d = compute_dice_f1_instance_difference(ground_truth=ref, prediction=pred_close)
check("high-IoU pair matches", float(f1_close) == 1.0, f"rq={f1_close}")


# --------------------------------------------------------------------------
# 3. Argument order + 4. RQ formula.
#    Asymmetric phantom: 1 reference instance, 3 predicted instances, exactly
#    one of which matches. Then TP=1, FP=2, FN=0 -> rq = 1/(1+1) = 0.5, and
#    |num_ref - num_pred| = 2.
# --------------------------------------------------------------------------
ref3 = np.zeros(shape, dtype=np.uint8)
pred3 = np.zeros(shape, dtype=np.uint8)
ref3[10:20, 10:20, 10:12] = 1          # 200 voxels
pred3[10:20, 10:20, 10:12] = 1         # identical -> matched, IoU 1.0
pred3[30:33, 30:33, 30:33] = 1         # spurious component 1
pred3[35:38, 35:38, 35:38] = 1         # spurious component 2
f1_asym, cd_asym, dice_asym = compute_dice_f1_instance_difference(
    ground_truth=ref3, prediction=pred3
)
expected_rq = 1.0 / (1.0 + 0.5 * 2 + 0.5 * 0)
check(
    "rq == TP/(TP+0.5FP+0.5FN)",
    abs(float(f1_asym) - expected_rq) < 1e-9,
    f"got rq={f1_asym}, expected {expected_rq}",
)
check(
    "absolute lesion count difference counts components",
    int(cd_asym) == 2,
    f"got |diff|={cd_asym}, expected 2 (1 ref vs 3 pred)",
)
# Roles are not swapped iff |diff| is computed the way we think; |diff| is
# symmetric, so confirm order via rq asymmetry instead: swapping ref/pred here
# would give TP=1, FP=0, FN=2 -> rq = 1/(1+1) = 0.5 as well. rq and |diff| are
# both symmetric, so record the order question as informational only.
notes.append(
    "rq, |count diff| and global_bin_dsc are all symmetric in (ref, pred), so a "
    "swapped argument order in the official wrapper would be undetectable and "
    "harmless for all four mask-based ranking metrics."
)


# --------------------------------------------------------------------------
# 5. global_bin_dsc is plain binary Dice.
# --------------------------------------------------------------------------
inter3 = int(np.logical_and(ref3, pred3).sum())
plain_dice = 2 * inter3 / (int(ref3.sum()) + int(pred3.sum()))
check(
    "global_bin_dsc == plain binary Dice",
    abs(float(dice_asym) - plain_dice) < 1e-9,
    f"got {dice_asym}, expected {plain_dice}",
)

# Empty-on-both -> empty_value (default 1.0).
f1_e, cd_e, dice_e = compute_dice_f1_instance_difference(
    ground_truth=np.zeros(shape, np.uint8), prediction=np.zeros(shape, np.uint8)
)
check(
    "empty/empty -> Dice and F1 == empty_value 1.0",
    float(f1_e) == 1.0 and float(dice_e) == 1.0 and int(cd_e) == 0,
    f"f1={f1_e} dice={dice_e} count_diff={cd_e}",
)


# --------------------------------------------------------------------------
# 6. PR-AUC branches.
# --------------------------------------------------------------------------
gt = np.zeros(64, dtype=np.uint8)
gt[:8] = 1
soft = np.linspace(0, 1, 64)[::-1].copy()  # perfect ranking
pr_perfect = compute_pr_auc(gt, soft)
check("PR-AUC of a perfect ranking is 1.0", abs(pr_perfect - 1.0) < 1e-6, f"got {pr_perfect}")

# Empty GT + non-constant soft map -> 0.0 (the trap).
pr_empty_varying = compute_pr_auc(np.zeros(64, np.uint8), soft)
check(
    "empty GT + varying soft map -> 0.0",
    pr_empty_varying == 0.0,
    f"got {pr_empty_varying}",
)

# Empty GT + constant soft map -> empty_value 1.0.
pr_empty_const = compute_pr_auc(np.zeros(64, np.uint8), np.zeros(64, np.float32))
check(
    "empty GT + constant soft map -> 1.0",
    pr_empty_const == 1.0,
    f"got {pr_empty_const}",
)

# PR-AUC is invariant to any strictly monotone transform of the soft map --
# this is why temperature scaling / Platt calibration cannot move this metric.
pr_monotone = compute_pr_auc(gt, 1.0 / (1.0 + np.exp(-(soft * 7 - 3))))
check(
    "PR-AUC invariant under a strictly monotone transform",
    abs(pr_monotone - pr_perfect) < 1e-6,
    f"raw={pr_perfect} transformed={pr_monotone}",
)


# --------------------------------------------------------------------------
# Volume difference: confirm the units and that the wrapper wants voxel volume
# in mL (the official signature takes a scalar `voxel_size` already in mL).
# --------------------------------------------------------------------------
a = np.zeros((10, 10, 10), np.uint8)
a[:, :, :5] = 1  # 500 voxels
b = np.zeros((10, 10, 10), np.uint8)
b[:, :, :3] = 1  # 300 voxels
vox_ml = np.array(1.0 * 1.0 * 1.0 / 1000.0)
avd = compute_absolute_volume_difference(a, b, vox_ml)
check(
    "absolute volume difference is in mL",
    abs(float(avd) - 0.2) < 1e-9,
    f"got {avd} mL, expected 0.2 mL for a 200 x 1mm^3 voxel difference",
)


print()
for n in notes:
    print(f"[NOTE] {n}")

if failures:
    print(f"\n{len(failures)} INVARIANT(S) VIOLATED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)

print("\nAll official-metric semantics confirmed.")
