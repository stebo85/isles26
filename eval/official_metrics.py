"""The ISLES'26 OFFICIAL ranking metrics, computed by the organizers' own code.

This module is deliberately thin. It does NOT reimplement anything: it imports
`utils/eval_utils.py` from the organizers' repository (cloned to
`work/isles26_upstream/`) and calls it. Any divergence between our numbers and
the leaderboard's is then a data/geometry problem, never a metric problem.

The five ranked metrics (README of ezequieldlrosa/isles26, HEAD 2026-07-24):

    1. Dice                            -- panoptica ``result.global_bin_dsc``
    2. Absolute volume difference (mL)
    3. Absolute lesion count difference -- |num_ref_instances - num_pred_instances|
    4. Lesion-wise F1                  -- panoptica ``result.rq`` (Recognition
       Quality) under ConnectedComponentsInstanceApproximator +
       NaiveThresholdMatching(matching_threshold=0.25), i.e. ONE-TO-ONE matching
       at IoU >= 0.25
    5. PR-AUC over the soft map        -- sklearn precision_recall_curve then
       auc(recall, precision) (trapezoidal; NOT average_precision_score)

Contrast with `eval/metrics.py`, which is this repo's own richer diagnostic
harness. That one matches lesions at 1-voxel any-overlap, many-to-many -- a much
more lenient criterion that is NOT what the leaderboard uses. Keep both: this
module decides, `eval/metrics.py` explains.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

# --- import the organizers' code verbatim -----------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_DIR = _REPO_ROOT / "work" / "isles26_upstream"


def _import_official():
    if not (UPSTREAM_DIR / "utils" / "eval_utils.py").is_file():
        raise RuntimeError(
            f"official evaluation code not found at {UPSTREAM_DIR}/utils/eval_utils.py.\n"
            "Clone it first:\n"
            f"  git clone https://github.com/ezequieldlrosa/isles26.git {UPSTREAM_DIR}"
        )
    if str(UPSTREAM_DIR) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_DIR))
    from utils import eval_utils  # type: ignore[import-not-found]

    return eval_utils


eval_utils = _import_official()

OFFICIAL_SOURCE = str(UPSTREAM_DIR / "utils" / "eval_utils.py")

# Higher-is-better for Dice / lesion-F1 / PR-AUC; lower-is-better for the two
# error metrics. Used by the rank aggregation.
METRIC_DIRECTION: dict[str, str] = {
    "dice": "max",
    "abs_volume_diff_ml": "min",
    "abs_lesion_count_diff": "min",
    "lesion_f1": "max",
    "pr_auc": "max",
}
RANKED_METRICS = tuple(METRIC_DIRECTION)


# --- connected components (for the min-size postprocessing sweep) ------------

_STRUCTURES = {
    6: ndi.generate_binary_structure(3, 1),
    18: ndi.generate_binary_structure(3, 2),
    26: ndi.generate_binary_structure(3, 3),
}


def min_size_filter(mask: np.ndarray, min_voxels: int, connectivity: int = 26) -> np.ndarray:
    """Drop connected components smaller than `min_voxels`.

    `connectivity` must match whatever panoptica's
    ConnectedComponentsInstanceApproximator uses, otherwise the filter and the
    scorer disagree about what a component is. `eval/probe_panoptica_semantics.py`
    measures it; do not assume.
    """
    if min_voxels <= 0:
        return mask
    lab, n = ndi.label(mask.astype(bool), structure=_STRUCTURES[connectivity])
    if n == 0:
        return mask
    counts = np.bincount(lab.ravel(), minlength=n + 1)
    keep = counts >= min_voxels
    keep[0] = False
    return keep[lab]


def min_size_filter_mm3(
    mask: np.ndarray, min_mm3: float, voxel_volume_mm3: float, connectivity: int = 26
) -> np.ndarray:
    """Physical-size variant. Preferred over a voxel threshold: only 783/1453
    ATLAS R3.0 sessions are 1 mm isotropic, so a fixed voxel count means a
    different physical lesion size on every scanner."""
    if min_mm3 <= 0:
        return mask
    min_voxels = int(math.ceil(min_mm3 / max(voxel_volume_mm3, 1e-9)))
    return min_size_filter(mask, min_voxels, connectivity=connectivity)


# --- per-case scoring -------------------------------------------------------


@dataclass
class OfficialCaseScore:
    subject_id: str
    dice: float
    abs_volume_diff_ml: float
    abs_lesion_count_diff: int
    lesion_f1: float
    pr_auc: float
    # diagnostics, not ranked
    gt_empty: bool
    pred_empty: bool
    voxel_volume_ml: float
    threshold: float | None = None
    min_component_voxels: int | None = None
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "subject_id": self.subject_id,
            "dice": self.dice,
            "abs_volume_diff_ml": self.abs_volume_diff_ml,
            "abs_lesion_count_diff": self.abs_lesion_count_diff,
            "lesion_f1": self.lesion_f1,
            "pr_auc": self.pr_auc,
            "gt_empty": self.gt_empty,
            "pred_empty": self.pred_empty,
            "voxel_volume_ml": self.voxel_volume_ml,
            "threshold": self.threshold,
            "min_component_voxels": self.min_component_voxels,
        }
        d.update(self.extras)
        return d


def score_masks(
    gt: np.ndarray,
    pred_binary: np.ndarray,
    voxel_volume_ml: float,
) -> tuple[float, float, int, float]:
    """The four mask-based official metrics. Returns (dice, avd_ml, |dcount|, rq)."""
    f1, count_diff, dice = eval_utils.compute_dice_f1_instance_difference(
        ground_truth=gt, prediction=pred_binary
    )
    avd = eval_utils.compute_absolute_volume_difference(
        im1=gt, im2=pred_binary, voxel_size=np.array(voxel_volume_ml)
    )
    return float(dice), float(avd), int(count_diff), float(f1)


def score_soft(gt: np.ndarray, soft: np.ndarray) -> float:
    """Official PR-AUC. Note the empty-GT branch: 1.0 only for a perfectly
    constant soft map, else 0.0 -- see `flat_soft_pr_auc`."""
    return float(eval_utils.compute_pr_auc(ground_truth=gt, prediction_map=soft))


def flat_soft_pr_auc(gt: np.ndarray) -> float:
    """PR-AUC obtained when the emitted soft map is CONSTANT (all zeros).

    This is the value the 'flatten the soft map when the binary prediction is
    empty' rule buys. On a lesion-free case the official code short-circuits to
    empty_value=1.0; on a case that does have a lesion the constant map produces
    a degenerate two-point curve. We evaluate rather than derive it so the number
    always tracks whatever sklearn actually does.
    """
    return score_soft(gt, np.zeros(gt.shape, dtype=np.float32))


def score_case(
    subject_id: str,
    gt: np.ndarray,
    pred_binary: np.ndarray,
    soft: np.ndarray,
    voxel_volume_ml: float,
    threshold: float | None = None,
    min_component_voxels: int | None = None,
) -> OfficialCaseScore:
    dice, avd, count_diff, f1 = score_masks(gt, pred_binary, voxel_volume_ml)
    pr_auc = score_soft(gt, soft)
    return OfficialCaseScore(
        subject_id=subject_id,
        dice=dice,
        abs_volume_diff_ml=avd,
        abs_lesion_count_diff=count_diff,
        lesion_f1=f1,
        pr_auc=pr_auc,
        gt_empty=not bool(np.any(gt)),
        pred_empty=not bool(np.any(pred_binary)),
        voxel_volume_ml=voxel_volume_ml,
        threshold=threshold,
        min_component_voxels=min_component_voxels,
    )


# --- aggregation ------------------------------------------------------------


def aggregate_means(rows: list[dict]) -> dict:
    """Plain per-case means, which is what `example_metrics.py` reports."""
    out: dict[str, float] = {"n": len(rows)}
    for m in RANKED_METRICS:
        vals = np.asarray([r[m] for r in rows], dtype=np.float64)
        out[f"{m}_mean"] = float(np.nanmean(vals))
        out[f"{m}_median"] = float(np.nanmedian(vals))
        out[f"{m}_std"] = float(np.nanstd(vals))
    return out


def rank_aggregate(configs: dict[str, list[dict]]) -> dict[str, dict]:
    """Rank-then-aggregate across configurations, the ISLES ranking convention.

    `configs` maps a configuration name to its per-case rows (all configurations
    must cover the same subjects). For every case and every ranked metric the
    configurations are ranked (1 = best, ties averaged), then each
    configuration's ranks are averaged over cases and over metrics.

    This is the aggregate that actually decides a leaderboard, and it can
    disagree with the mean aggregate -- which is exactly why both are reported.
    """
    names = list(configs)
    if not names:
        return {}
    subject_sets = {n: {r["subject_id"] for r in configs[n]} for n in names}
    common = set.intersection(*subject_sets.values())
    if not common:
        raise ValueError("configurations share no subjects")

    by_name_subject = {
        n: {r["subject_id"]: r for r in configs[n] if r["subject_id"] in common} for n in names
    }
    subjects = sorted(common)

    # rank_sums[name][metric] accumulates ranks over cases
    rank_sums = {n: {m: 0.0 for m in RANKED_METRICS} for n in names}
    for subj in subjects:
        for metric in RANKED_METRICS:
            vals = np.asarray(
                [float(by_name_subject[n][subj][metric]) for n in names], dtype=np.float64
            )
            if METRIC_DIRECTION[metric] == "max":
                order_key = -vals
            else:
                order_key = vals
            # average ranks for ties
            sorter = np.argsort(order_key, kind="mergesort")
            ranks = np.empty(len(names), dtype=np.float64)
            sorted_keys = order_key[sorter]
            i = 0
            while i < len(names):
                j = i
                while j + 1 < len(names) and sorted_keys[j + 1] == sorted_keys[i]:
                    j += 1
                avg_rank = (i + j) / 2.0 + 1.0
                ranks[sorter[i : j + 1]] = avg_rank
                i = j + 1
            for k, n in enumerate(names):
                rank_sums[n][metric] += float(ranks[k])

    n_subj = len(subjects)
    out: dict[str, dict] = {}
    for n in names:
        per_metric = {m: rank_sums[n][m] / n_subj for m in RANKED_METRICS}
        out[n] = {
            "n_subjects": n_subj,
            "mean_rank_per_metric": per_metric,
            "mean_rank": float(np.mean(list(per_metric.values()))),
        }
    return out
