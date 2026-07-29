#!/usr/bin/env python3
"""Sweep binary thresholds over saved lesion-probability NIfTIs.

Selecting a threshold on the same out-of-fold pool that is then re-scored with
that threshold is in-sample optimisation: the reported number contains the
selection noise. On the 1450-case ATLAS OOF pool the whole sweep surface spans
~0.004 lesion-F1 and is non-monotone, so the argmax mostly tracks noise and the
three offered selection metrics land on three different thresholds. Two guards
are therefore built in here:

* nested selection (``--nested``, on by default when >= 2 folds are swept):
  the threshold is chosen on K-1 folds and the held-out fold is scored with it,
  rotating over all folds. The pooled held-out metrics are the number this tool
  reports as its headline, because they were never optimised. The in-sample
  argmax is still emitted, but explicitly labelled as in-sample/optimistic.
* a minimum-effect gate (``select_threshold``): the sweep refuses to move off
  the default threshold unless the gain over the default exceeds the paired
  case-level bootstrap standard error of the selected metric. The effect size,
  its SE and the decision are written into the summary so the choice is
  auditable rather than implicit in an argmax.

The grid also defaults to the full [0.05, 0.95] range: a narrow window can hide
the fact that a metric is still rising at the boundary, so a selected threshold
that lands on a grid edge is treated as a hard error unless --allow-grid-edge.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage as ndi


CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)

# nnU-Net's own default operating point; the sweep must beat it by more than
# noise before anything else is allowed to ship.
DEFAULT_THRESHOLD = 0.5

# Wide by design: the previous 0.35-0.55 window ended while Dice was still
# rising at the low edge, so the Dice optimum was never inside the search. Step
# 0.05 is deliberately coarser than the old 0.01 because the metric differences
# between adjacent 0.01 steps are far below the case-level noise floor.
DEFAULT_THRESHOLD_GRID = "0.05:0.95:0.05"

SCORE_KEYS = ("lesion_f1_mean", "lesion_f1_pooled", "dice_lesion_f1_hmean")


@dataclass(frozen=True)
class CaseInput:
    case_id: str
    prob_path: Path
    gt_path: Path


def case_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def format_threshold(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def parse_thresholds(spec: str) -> list[float]:
    text = spec.strip()
    if not text:
        raise argparse.ArgumentTypeError("threshold spec must not be empty")
    if "," in text:
        values = [Decimal(part.strip()) for part in text.split(",") if part.strip()]
    elif ":" in text:
        parts = [Decimal(part.strip()) for part in text.split(":")]
        if len(parts) == 2:
            start, stop = parts
            step = Decimal("0.01")
        elif len(parts) == 3:
            start, stop, step = parts
        else:
            raise argparse.ArgumentTypeError("range threshold spec must be START:STOP[:STEP]")
        if step <= 0:
            raise argparse.ArgumentTypeError("threshold step must be > 0")
        values = []
        value = start
        # Include STOP despite decimal-to-float representation.
        while value <= stop + (step / Decimal("10")):
            values.append(value)
            value += step
    else:
        values = [Decimal(text)]

    out = sorted({float(v) for v in values})
    if not out:
        raise argparse.ArgumentTypeError("threshold spec produced no values")
    bad = [v for v in out if not (0.0 <= v <= 1.0) or not math.isfinite(v)]
    if bad:
        raise argparse.ArgumentTypeError(f"thresholds must be finite and in [0, 1], got {bad[:5]}")
    return out


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_folds(value: str, n_folds: int) -> list[int]:
    text = value.strip().lower()
    if text == "all":
        return list(range(n_folds))
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            fold = int(part)
        except ValueError as exc:
            raise SystemExit(f"invalid fold {part!r}; use an int, comma-list, or 'all'") from exc
        if fold < 0 or fold >= n_folds:
            raise SystemExit(f"fold {fold} is outside available range 0-{n_folds - 1}")
        out.append(fold)
    if not out:
        raise SystemExit("--folds resolved to an empty fold list")
    return out


def load_val_case_folds(path: Path, folds_arg: str) -> list[tuple[str, int]]:
    """Return (case_id, fold) pairs; nested selection needs the fold membership."""
    payload = json.loads(path.read_text())
    folds = payload.get("folds", payload) if isinstance(payload, dict) else payload
    if not isinstance(folds, list):
        raise SystemExit(f"could not read folds from {path}")
    selected_folds = parse_folds(folds_arg, len(folds))

    val_cases: list[tuple[str, int]] = []
    seen: set[str] = set()
    for fold in selected_folds:
        try:
            fold_cases = list(folds[fold]["val"])
        except (IndexError, KeyError, TypeError) as exc:
            raise SystemExit(f"could not read fold {fold} val cases from {path}: {exc}") from exc
        for case_id in fold_cases:
            if case_id in seen:
                raise SystemExit(f"duplicate validation case across requested folds: {case_id}")
            seen.add(case_id)
            val_cases.append((case_id, fold))
    if not val_cases:
        raise SystemExit(f"requested validation case list is empty in {path}")
    return val_cases


def load_val_cases(path: Path, folds_arg: str) -> list[str]:
    return [case_id for case_id, _ in load_val_case_folds(path, folds_arg)]


def collect_prob_paths(prob_dirs: list[Path]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for prob_dir in prob_dirs:
        if not prob_dir.is_dir():
            raise SystemExit(f"probability directory does not exist: {prob_dir}")
        for path in sorted(prob_dir.glob("ATLAS_*.nii.gz")):
            case_id = case_id_from_path(path)
            if case_id in out:
                raise SystemExit(f"duplicate probability NIfTI for {case_id}: {out[case_id]} and {path}")
            out[case_id] = path
    if not out:
        raise SystemExit("no probability NIfTIs found")
    return out


def squeeze_3d(data: np.ndarray, path: Path) -> np.ndarray:
    arr = np.asanyarray(data)
    while arr.ndim > 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected 3-D or singleton-4-D image, got shape {arr.shape}")
    return arr


def load_probability(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(str(path))
    data = squeeze_3d(np.asanyarray(img.dataobj), path).astype(np.float32, copy=False)
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: probability image contains NaN/Inf")
    if float(data.min()) < -1e-4 or float(data.max()) > 1.0001:
        raise ValueError(f"{path}: probability image outside expected [0, 1] range")
    return img, data


def load_gt(path: Path) -> tuple[nib.Nifti1Image, np.ndarray, tuple[float, float, float]]:
    img = nib.load(str(path))
    data = squeeze_3d(np.asanyarray(img.dataobj), path)
    zooms = img.header.get_zooms()[:3]
    return img, (data > 0), tuple(float(v) for v in zooms)


def same_grid(prob_img: nib.Nifti1Image, prob_data: np.ndarray, gt_img: nib.Nifti1Image, gt_data: np.ndarray) -> bool:
    return prob_data.shape == gt_data.shape and np.allclose(prob_img.affine, gt_img.affine, atol=1e-3)


def resample_binary_to_gt(mask: np.ndarray, prob_img: nib.Nifti1Image, gt_img: nib.Nifti1Image, gt_shape: tuple[int, int, int]) -> np.ndarray:
    header = prob_img.header.copy()
    header.set_data_shape(mask.shape)
    header.set_data_dtype(np.uint8)
    pred_img = nib.Nifti1Image(mask.astype(np.uint8, copy=False), prob_img.affine, header)
    out = resample_from_to(pred_img, (gt_shape, gt_img.affine), order=0, mode="constant", cval=0)
    return squeeze_3d(np.asanyarray(out.dataobj), Path("<resampled>")) > 0


def overlap_summary(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    p_sum = tp + fp
    g_sum = tp + fn

    if p_sum == 0 and g_sum == 0:
        dice = 1.0
        iou = 1.0
    elif p_sum == 0 or g_sum == 0:
        dice = 0.0
        iou = 0.0
    else:
        dice = 2 * tp / (2 * tp + fp + fn)
        iou = tp / (tp + fp + fn)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else math.nan,
        "precision": float(tp / (tp + fp)) if (tp + fp) else math.nan,
        "tp_voxels": tp,
        "fp_voxels": fp,
        "fn_voxels": fn,
        "tn_voxels": tn,
    }


def lesion_summary(
    pred: np.ndarray,
    gt_label: np.ndarray,
    n_gt: int,
    gt_sizes: list[int],
    min_overlap_voxels: int,
) -> dict[str, float | int]:
    pred_label, n_pred = ndi.label(pred, structure=CONNECTIVITY_26)
    tp_pred = 0
    tp_gt = 0

    if n_pred > 0 and n_gt > 0:
        mask_both = (pred_label > 0) & (gt_label > 0)
        if mask_both.any():
            flat_pred = pred_label[mask_both].ravel()
            flat_gt = gt_label[mask_both].ravel()
            idx = flat_gt.astype(np.int64) * (n_pred + 1) + flat_pred.astype(np.int64)
            overlap_flat = np.bincount(idx, minlength=(n_gt + 1) * (n_pred + 1))
            overlap = overlap_flat.reshape(n_gt + 1, n_pred + 1)
            gt_max_overlap = overlap[1:, 1:].max(axis=1)
            pred_max_overlap = overlap[1:, 1:].max(axis=0)
            tp_gt = int((gt_max_overlap >= min_overlap_voxels).sum())
            tp_pred = int((pred_max_overlap >= min_overlap_voxels).sum())

    fn = n_gt - tp_gt
    fp = n_pred - tp_pred
    if n_gt == 0 and n_pred == 0:
        precision = recall = f1 = 1.0
    elif n_gt == 0:
        precision = 0.0
        recall = 1.0
        f1 = 0.0
    elif n_pred == 0:
        precision = 1.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = tp_pred / n_pred
        recall = tp_gt / n_gt
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return {
        "gt_lesion_count": int(n_gt),
        "pred_lesion_count": int(n_pred),
        "detected_gt_lesions": int(tp_gt),
        "tp_pred_lesions": int(tp_pred),
        "fp_lesions": int(fp),
        "fn_lesions": int(fn),
        "lesion_precision": float(precision),
        "lesion_recall": float(recall),
        "lesion_f1": float(f1),
        "abs_lesion_count_diff": int(abs(n_pred - n_gt)),
        "gt_lesion_voxels": int(sum(gt_sizes)),
    }


def case_metrics_for_threshold(
    threshold: float,
    pred: np.ndarray,
    gt: np.ndarray,
    gt_label: np.ndarray,
    n_gt: int,
    gt_sizes: list[int],
    voxel_size_mm: tuple[float, float, float],
    min_overlap_voxels: int,
) -> dict[str, float | int]:
    overlap = overlap_summary(pred, gt)
    lesion = lesion_summary(pred, gt_label, n_gt, gt_sizes, min_overlap_voxels)
    voxel_volume_ml = float(np.prod(voxel_size_mm)) / 1000.0
    pred_voxels = int(pred.sum())
    gt_voxels = int(gt.sum())
    pred_vol_ml = pred_voxels * voxel_volume_ml
    gt_vol_ml = gt_voxels * voxel_volume_ml
    return {
        "threshold": threshold,
        **overlap,
        **lesion,
        "pred_voxels": pred_voxels,
        "gt_voxels": gt_voxels,
        "pred_vol_ml": float(pred_vol_ml),
        "gt_vol_ml": float(gt_vol_ml),
        "abs_volume_diff_ml": float(abs(pred_vol_ml - gt_vol_ml)),
    }


def evaluate_case(case: CaseInput, thresholds: list[float], min_overlap_voxels: int) -> tuple[str, list[dict[str, float | int]]]:
    prob_img, prob = load_probability(case.prob_path)
    gt_img, gt, voxel_size_mm = load_gt(case.gt_path)
    gt_label, n_gt = ndi.label(gt, structure=CONNECTIVITY_26)
    gt_sizes = np.bincount(gt_label.ravel(), minlength=n_gt + 1).astype(np.int64).tolist()[1:]
    grids_match = same_grid(prob_img, prob, gt_img, gt)

    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        pred = prob >= threshold
        if not grids_match:
            pred = resample_binary_to_gt(pred, prob_img, gt_img, tuple(int(v) for v in gt.shape))
        rows.append(
            case_metrics_for_threshold(
                threshold,
                pred,
                gt,
                gt_label,
                n_gt,
                gt_sizes,
                voxel_size_mm,
                min_overlap_voxels,
            )
        )
    return case.case_id, rows


def safe_mean(values: list[float]) -> float:
    arr = np.asarray([v for v in values if not (isinstance(v, float) and math.isnan(v))], dtype=np.float64)
    return float(arr.mean()) if arr.size else math.nan


def safe_median(values: list[float]) -> float:
    arr = np.asarray([v for v in values if not (isinstance(v, float) and math.isnan(v))], dtype=np.float64)
    return float(np.median(arr)) if arr.size else math.nan


def harmonic(a: float, b: float) -> float:
    if not math.isfinite(a) or not math.isfinite(b) or (a + b) <= 0:
        return 0.0
    return float(2 * a * b / (a + b))


def summarise_rows(rows: Sequence[dict[str, float | int]]) -> dict[str, float | int]:
    """Aggregate per-case rows into one summary line.

    Split out of ``aggregate_by_threshold`` because nested selection pools rows
    that were thresholded differently per fold, so the aggregation cannot be
    keyed on a single threshold.
    """
    dice = [float(r["dice"]) for r in rows]
    lesion_f1 = [float(r["lesion_f1"]) for r in rows]
    lesion_precision = [float(r["lesion_precision"]) for r in rows]
    lesion_recall = [float(r["lesion_recall"]) for r in rows]
    voxel_precision = [float(r["precision"]) for r in rows]
    voxel_recall = [float(r["sensitivity"]) for r in rows]

    pred_lesions = int(sum(int(r["pred_lesion_count"]) for r in rows))
    gt_lesions = int(sum(int(r["gt_lesion_count"]) for r in rows))
    detected_gt = int(sum(int(r["detected_gt_lesions"]) for r in rows))
    tp_pred = int(sum(int(r["tp_pred_lesions"]) for r in rows))
    pooled_precision = (tp_pred / pred_lesions) if pred_lesions else (1.0 if gt_lesions == 0 else 0.0)
    pooled_recall = (detected_gt / gt_lesions) if gt_lesions else 1.0
    pooled_f1 = 0.0 if (pooled_precision + pooled_recall) == 0 else 2 * pooled_precision * pooled_recall / (pooled_precision + pooled_recall)

    dice_mean = safe_mean(dice)
    lesion_f1_mean = safe_mean(lesion_f1)
    return {
        "n_cases": len(rows),
        "dice_mean": dice_mean,
        "dice_median": safe_median(dice),
        "voxel_precision_mean": safe_mean(voxel_precision),
        "voxel_recall_mean": safe_mean(voxel_recall),
        "lesion_f1_mean": lesion_f1_mean,
        "lesion_f1_median": safe_median(lesion_f1),
        "lesion_precision_mean": safe_mean(lesion_precision),
        "lesion_recall_mean": safe_mean(lesion_recall),
        "lesion_f1_pooled": float(pooled_f1),
        "lesion_precision_pooled": float(pooled_precision),
        "lesion_recall_pooled": float(pooled_recall),
        "dice_lesion_f1_hmean": harmonic(dice_mean, lesion_f1_mean),
        "pred_lesion_count_sum": pred_lesions,
        "gt_lesion_count_sum": gt_lesions,
        "detected_gt_lesions_sum": detected_gt,
        "tp_pred_lesions_sum": tp_pred,
        "fp_lesions_sum": int(sum(int(r["fp_lesions"]) for r in rows)),
        "fn_lesions_sum": int(sum(int(r["fn_lesions"]) for r in rows)),
        "abs_lesion_count_diff_mean": safe_mean([float(r["abs_lesion_count_diff"]) for r in rows]),
        "pred_vol_ml_mean": safe_mean([float(r["pred_vol_ml"]) for r in rows]),
        "gt_vol_ml_mean": safe_mean([float(r["gt_vol_ml"]) for r in rows]),
        "abs_volume_diff_ml_mean": safe_mean([float(r["abs_volume_diff_ml"]) for r in rows]),
    }


def aggregate_by_threshold(per_case_by_threshold: Mapping[float, Sequence[dict[str, float | int]]]) -> list[dict[str, float | int]]:
    summary: list[dict[str, float | int]] = []
    for threshold in sorted(per_case_by_threshold):
        summary.append({"threshold": threshold, **summarise_rows(per_case_by_threshold[threshold])})
    return summary


def rows_by_threshold(per_case: Mapping[float, Mapping[str, dict[str, float | int]]]) -> dict[float, list[dict[str, float | int]]]:
    """Drop the case-id keying used for fold subsetting back to plain row lists."""
    return {threshold: list(cases.values()) for threshold, cases in per_case.items()}


def best_row(summary: list[dict[str, float | int]], score_key: str) -> dict[str, float | int]:
    """Unrestricted argmax over the sweep. In-sample and ungated by construction."""
    if score_key == "dice_lesion_f1_hmean":
        return max(summary, key=lambda row: (float(row[score_key]), float(row["lesion_f1_mean"]), float(row["dice_mean"])))
    if score_key == "lesion_f1_pooled":
        return max(summary, key=lambda row: (float(row[score_key]), float(row["lesion_f1_mean"]), float(row["dice_mean"])))
    return max(summary, key=lambda row: (float(row[score_key]), float(row["dice_mean"]), float(row["lesion_recall_mean"])))


@dataclass(frozen=True)
class CaseVectors:
    """Per-case quantities needed to recompute any sweep score under resampling."""

    dice: np.ndarray
    lesion_f1: np.ndarray
    tp_pred_lesions: np.ndarray
    pred_lesion_count: np.ndarray
    detected_gt_lesions: np.ndarray
    gt_lesion_count: np.ndarray


def case_vectors(rows: Sequence[dict[str, float | int]]) -> CaseVectors:
    return CaseVectors(
        dice=np.asarray([float(r["dice"]) for r in rows], dtype=np.float64),
        lesion_f1=np.asarray([float(r["lesion_f1"]) for r in rows], dtype=np.float64),
        tp_pred_lesions=np.asarray([float(r["tp_pred_lesions"]) for r in rows], dtype=np.float64),
        pred_lesion_count=np.asarray([float(r["pred_lesion_count"]) for r in rows], dtype=np.float64),
        detected_gt_lesions=np.asarray([float(r["detected_gt_lesions"]) for r in rows], dtype=np.float64),
        gt_lesion_count=np.asarray([float(r["gt_lesion_count"]) for r in rows], dtype=np.float64),
    )


def score_from_vectors(vectors: CaseVectors, idx: np.ndarray, score_key: str) -> np.ndarray:
    """Score each resample in ``idx`` (shape (n_resamples, n_cases)).

    Mirrors ``summarise_rows`` exactly for the three selectable score keys; kept
    vectorised so the bootstrap costs milliseconds instead of minutes.
    """
    if score_key == "lesion_f1_mean":
        return np.nanmean(vectors.lesion_f1[idx], axis=1)
    if score_key == "dice_mean":
        return np.nanmean(vectors.dice[idx], axis=1)
    if score_key == "dice_lesion_f1_hmean":
        dice_mean = np.nanmean(vectors.dice[idx], axis=1)
        f1_mean = np.nanmean(vectors.lesion_f1[idx], axis=1)
        total = dice_mean + f1_mean
        return np.where(total > 0, 2 * dice_mean * f1_mean / np.where(total > 0, total, 1.0), 0.0)
    if score_key == "lesion_f1_pooled":
        tp_pred = vectors.tp_pred_lesions[idx].sum(axis=1)
        pred = vectors.pred_lesion_count[idx].sum(axis=1)
        detected = vectors.detected_gt_lesions[idx].sum(axis=1)
        gt = vectors.gt_lesion_count[idx].sum(axis=1)
        precision = np.where(pred > 0, tp_pred / np.where(pred > 0, pred, 1.0), np.where(gt > 0, 0.0, 1.0))
        recall = np.where(gt > 0, detected / np.where(gt > 0, gt, 1.0), 1.0)
        total = precision + recall
        return np.where(total > 0, 2 * precision * recall / np.where(total > 0, total, 1.0), 0.0)
    raise ValueError(f"unsupported score key for bootstrap: {score_key}")


def bootstrap_paired_difference(
    candidate: CaseVectors,
    reference: CaseVectors,
    score_key: str,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """Paired case-level bootstrap of score(candidate) - score(reference).

    Paired because both thresholds are applied to the same cases: the shared
    case-difficulty variance cancels, which is exactly the variance an unpaired
    SE would hide.
    """
    n_cases = int(candidate.dice.size)
    if n_cases != int(reference.dice.size):
        raise ValueError("candidate and reference must cover the same cases")
    identity = np.arange(n_cases)[None, :]
    effect = float(
        score_from_vectors(candidate, identity, score_key)[0] - score_from_vectors(reference, identity, score_key)[0]
    )
    if n_cases < 2 or n_boot < 2:
        return {"effect": effect, "se": math.nan, "ci_lo": math.nan, "ci_hi": math.nan, "n_boot": 0}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_cases, size=(n_boot, n_cases))
    diffs = score_from_vectors(candidate, idx, score_key) - score_from_vectors(reference, idx, score_key)
    return {
        "effect": effect,
        "se": float(np.std(diffs, ddof=1)),
        "ci_lo": float(np.percentile(diffs, 2.5)),
        "ci_hi": float(np.percentile(diffs, 97.5)),
        "n_boot": int(n_boot),
    }


def resolve_threshold_key(thresholds: Iterable[float], threshold: float) -> float | None:
    """Match a requested threshold against sweep keys without float-equality traps."""
    return next((t for t in sorted(thresholds) if math.isclose(t, threshold, abs_tol=1e-9)), None)


@dataclass(frozen=True)
class SelectionDecision:
    """Auditable record of one threshold choice."""

    score_key: str
    default_threshold: float
    argmax_threshold: float
    selected_threshold: float
    default_score: float
    argmax_score: float
    effect_size: float
    bootstrap_se: float
    bootstrap_ci95_lo: float
    bootstrap_ci95_hi: float
    effect_z: float
    # Tuning one metric usually pays for it in another; the collateral Dice
    # change is recorded so the trade is visible rather than implicit.
    dice_effect_size: float
    dice_bootstrap_se: float
    dice_effect_z: float
    gate_se_multiplier: float
    n_boot: int
    n_cases: int
    gate_enabled: bool
    deviated_from_default: bool
    argmax_on_grid_edge: bool
    selected_on_grid_edge: bool
    reason: str


def select_threshold(
    summary: list[dict[str, float | int]],
    per_case: Mapping[float, Mapping[str, dict[str, float | int]]],
    score_key: str,
    *,
    default_threshold: float = DEFAULT_THRESHOLD,
    gate_se_multiplier: float = 1.0,
    n_boot: int = 2000,
    seed: int = 0,
    gate_enabled: bool = True,
) -> SelectionDecision:
    """Argmax the sweep, then refuse to move unless the gain beats its own noise.

    The sweep surface on this data is flat to within a few 1e-3 while the
    case-level SE of the same statistic is an order of magnitude larger, so a
    bare argmax ships noise. The gate makes the default threshold the null
    hypothesis and requires the paired bootstrap effect to clear
    ``gate_se_multiplier`` standard errors before deviating from it.
    """
    thresholds = sorted(per_case)
    if not thresholds:
        raise ValueError("empty sweep")
    default_key = resolve_threshold_key(thresholds, default_threshold)
    if default_key is None:
        raise SystemExit(
            f"default threshold {format_threshold(default_threshold)} is not in the sweep grid "
            f"{[format_threshold(t) for t in thresholds]}; every honest comparison in this tool is against that "
            "null hypothesis, so add it to --thresholds or point --default-threshold at a grid value"
        )

    argmax = best_row(summary, score_key)
    argmax_threshold = float(argmax["threshold"])
    argmax_score = float(argmax[score_key])
    default_row = next(row for row in summary if math.isclose(float(row["threshold"]), default_key, abs_tol=1e-9))
    default_score = float(default_row[score_key])

    case_order = sorted(per_case[default_key])
    candidate = case_vectors([per_case[argmax_threshold][case_id] for case_id in case_order])
    reference = case_vectors([per_case[default_key][case_id] for case_id in case_order])
    boot = bootstrap_paired_difference(candidate, reference, score_key, n_boot, seed)
    dice_boot = bootstrap_paired_difference(candidate, reference, "dice_mean", n_boot, seed)
    effect = float(boot["effect"])
    se = float(boot["se"])
    z = effect / se if math.isfinite(se) and se > 0 else (math.inf if effect > 0 else 0.0)
    dice_se = float(dice_boot["se"])
    dice_effect_z = (
        float(dice_boot["effect"]) / dice_se
        if math.isfinite(dice_se) and dice_se > 0
        else (math.copysign(math.inf, dice_boot["effect"]) if dice_boot["effect"] else 0.0)
    )

    # A degenerate SE (single case, or every case identical under both
    # thresholds) leaves nothing to gate against, so fall back to "any strict
    # improvement passes".
    threshold_on_effect = gate_se_multiplier * se if (math.isfinite(se) and se > 0) else 0.0
    if not gate_enabled:
        deviated = not math.isclose(argmax_threshold, default_key, abs_tol=1e-9)
        reason = "minimum-effect gate disabled (--no-min-effect-gate): bare in-sample argmax"
    elif math.isclose(argmax_threshold, default_key, abs_tol=1e-9):
        deviated = False
        reason = "argmax is the default threshold; nothing to gate"
    elif effect > threshold_on_effect:
        deviated = True
        reason = (
            f"effect {effect:+.5f} exceeds {gate_se_multiplier:g}x bootstrap SE {se:.5f} (z={z:.2f}); "
            "deviation from the default threshold accepted"
        )
    else:
        deviated = False
        reason = (
            f"effect {effect:+.5f} does not exceed {gate_se_multiplier:g}x bootstrap SE {se:.5f} (z={z:.2f}); "
            f"keeping default threshold {format_threshold(default_key)}"
        )

    selected_threshold = argmax_threshold if deviated else default_key
    on_edge = len(thresholds) > 1 and (
        math.isclose(selected_threshold, thresholds[0], abs_tol=1e-9)
        or math.isclose(selected_threshold, thresholds[-1], abs_tol=1e-9)
    )
    argmax_on_edge = len(thresholds) > 1 and (
        math.isclose(argmax_threshold, thresholds[0], abs_tol=1e-9)
        or math.isclose(argmax_threshold, thresholds[-1], abs_tol=1e-9)
    )

    return SelectionDecision(
        score_key=score_key,
        default_threshold=float(default_key),
        argmax_threshold=argmax_threshold,
        selected_threshold=float(selected_threshold),
        default_score=default_score,
        argmax_score=argmax_score,
        effect_size=effect,
        bootstrap_se=se,
        bootstrap_ci95_lo=float(boot["ci_lo"]),
        bootstrap_ci95_hi=float(boot["ci_hi"]),
        effect_z=float(z),
        dice_effect_size=float(dice_boot["effect"]),
        dice_bootstrap_se=float(dice_boot["se"]),
        dice_effect_z=dice_effect_z,
        gate_se_multiplier=float(gate_se_multiplier),
        n_boot=int(boot["n_boot"]),
        n_cases=len(case_order),
        gate_enabled=bool(gate_enabled),
        deviated_from_default=bool(deviated),
        argmax_on_grid_edge=bool(argmax_on_edge),
        selected_on_grid_edge=bool(on_edge),
        reason=reason,
    )


def nested_selection(
    per_case: Mapping[float, Mapping[str, dict[str, float | int]]],
    case_folds: Mapping[str, int],
    score_key: str,
    *,
    default_threshold: float = DEFAULT_THRESHOLD,
    gate_se_multiplier: float = 1.0,
    n_boot: int = 2000,
    seed: int = 0,
    gate_enabled: bool = True,
) -> dict[str, object]:
    """Select on K-1 folds, score the held-out fold, rotate over all folds.

    This is the number worth reporting: no case contributes to the choice of the
    threshold it is scored under, so the selection noise that inflates the
    in-sample argmax is not counted as performance.
    """
    folds = sorted({int(f) for f in case_folds.values()})
    if len(folds) < 2:
        raise ValueError("nested selection needs at least 2 folds")

    thresholds = sorted(per_case)
    held_out_rows: list[dict[str, float | int]] = []
    per_fold: list[dict[str, object]] = []
    for held_out in folds:
        inner_cases = [case_id for case_id, fold in case_folds.items() if int(fold) != held_out]
        outer_cases = [case_id for case_id, fold in case_folds.items() if int(fold) == held_out]
        inner_per_case = {
            threshold: {case_id: per_case[threshold][case_id] for case_id in inner_cases} for threshold in thresholds
        }
        inner_summary = aggregate_by_threshold(rows_by_threshold(inner_per_case))
        decision = select_threshold(
            inner_summary,
            inner_per_case,
            score_key,
            default_threshold=default_threshold,
            gate_se_multiplier=gate_se_multiplier,
            n_boot=n_boot,
            # Vary the stream per fold so the folds are not all resampled identically.
            seed=seed + held_out + 1,
            gate_enabled=gate_enabled,
        )
        fold_rows = [per_case[decision.selected_threshold][case_id] for case_id in outer_cases]
        held_out_rows.extend(fold_rows)
        per_fold.append(
            {
                "fold": held_out,
                "n_train_cases": len(inner_cases),
                "n_held_out_cases": len(outer_cases),
                "decision": asdict(decision),
                "held_out_metrics": summarise_rows(fold_rows),
            }
        )

    nested_metrics = summarise_rows(held_out_rows)
    default_key = resolve_threshold_key(thresholds, default_threshold)
    if default_key is None:
        raise SystemExit(f"default threshold {format_threshold(default_threshold)} is not in the sweep grid")
    default_metrics = summarise_rows([per_case[default_key][case_id] for case_id in case_folds])
    selected = sorted({float(entry["decision"]["selected_threshold"]) for entry in per_fold})  # type: ignore[index]
    return {
        "score_key": score_key,
        "n_folds": len(folds),
        "n_cases": len(held_out_rows),
        "selected_thresholds_per_fold": {
            int(entry["fold"]): float(entry["decision"]["selected_threshold"]) for entry in per_fold  # type: ignore[index]
        },
        "distinct_selected_thresholds": [format_threshold(t) for t in selected],
        "held_out_metrics": nested_metrics,
        "fixed_default_metrics": default_metrics,
        "gain_over_default": float(nested_metrics[score_key]) - float(default_metrics[score_key]),
        "per_fold": per_fold,
        "note": (
            "Honest estimate: the threshold applied to each case was chosen on the other folds only. "
            "Compare against fixed_default_metrics, not against the in-sample argmax."
        ),
    }


def json_safe(value: object) -> object:
    """Map non-finite floats to null so the summary stays strict JSON.

    A degenerate bootstrap (0 resamples, 1 case) legitimately yields NaN/inf and
    must not silently abort the whole sweep at dump time.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    fieldnames = [
        "threshold",
        "n_cases",
        "dice_mean",
        "dice_median",
        "voxel_precision_mean",
        "voxel_recall_mean",
        "lesion_f1_mean",
        "lesion_f1_median",
        "lesion_precision_mean",
        "lesion_recall_mean",
        "lesion_f1_pooled",
        "lesion_precision_pooled",
        "lesion_recall_pooled",
        "dice_lesion_f1_hmean",
        "pred_lesion_count_sum",
        "gt_lesion_count_sum",
        "detected_gt_lesions_sum",
        "tp_pred_lesions_sum",
        "fp_lesions_sum",
        "fn_lesions_sum",
        "abs_lesion_count_diff_mean",
        "pred_vol_ml_mean",
        "gt_vol_ml_mean",
        "abs_volume_diff_ml_mean",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["threshold"] = format_threshold(float(out["threshold"]))
            writer.writerow(out)


def write_markdown(
    path: Path,
    summary: list[dict[str, float | int]],
    best_rows: dict[str, dict[str, float | int]],
    primary_score: str,
    decision: SelectionDecision,
    nested: dict[str, object] | None,
) -> None:
    lines = [
        "# Probability Threshold Sweep",
        "",
        f"Primary selection metric: `{primary_score}`.",
        "",
        "## Headline (honest) numbers",
        "",
    ]
    if nested is None:
        lines.extend(
            [
                "**No nested estimate available** (fewer than 2 folds swept, or --no-nested). "
                "Every number below was measured on the same cases the threshold was chosen on "
                "and is therefore optimistic.",
                "",
            ]
        )
    else:
        held = nested["held_out_metrics"]  # type: ignore[index]
        fixed = nested["fixed_default_metrics"]  # type: ignore[index]
        lines.extend(
            [
                f"Nested {nested['n_folds']}-fold selection: the threshold applied to each case was chosen on the "  # type: ignore[index]
                "other folds only, so these numbers were never optimised.",
                "",
                "| estimate | threshold(s) | Dice mean | Lesion F1 mean | pooled lesion F1 | AVD mL mean |",
                "|---|---|---:|---:|---:|---:|",
                "| nested (REPORT THIS) | {thr} | {dice:.4f} | {f1:.4f} | {pf1:.4f} | {avd:.4f} |".format(
                    thr=", ".join(nested["distinct_selected_thresholds"]),  # type: ignore[index]
                    dice=float(held["dice_mean"]),  # type: ignore[index]
                    f1=float(held["lesion_f1_mean"]),  # type: ignore[index]
                    pf1=float(held["lesion_f1_pooled"]),  # type: ignore[index]
                    avd=float(held["abs_volume_diff_ml_mean"]),  # type: ignore[index]
                ),
                "| fixed default threshold | {thr} | {dice:.4f} | {f1:.4f} | {pf1:.4f} | {avd:.4f} |".format(
                    thr=format_threshold(decision.default_threshold),
                    dice=float(fixed["dice_mean"]),  # type: ignore[index]
                    f1=float(fixed["lesion_f1_mean"]),  # type: ignore[index]
                    pf1=float(fixed["lesion_f1_pooled"]),  # type: ignore[index]
                    avd=float(fixed["abs_volume_diff_ml_mean"]),  # type: ignore[index]
                ),
                "",
                f"Honest gain of tuning over the fixed default on `{primary_score}`: "
                f"{float(nested['gain_over_default']):+.5f}.",  # type: ignore[index]
                "",
            ]
        )

    lines.extend(
        [
            "## Selection decision (minimum-effect gate)",
            "",
            f"- default threshold: `{format_threshold(decision.default_threshold)}` "
            f"({primary_score} = {decision.default_score:.5f})",
            f"- in-sample argmax: `{format_threshold(decision.argmax_threshold)}` "
            f"({primary_score} = {decision.argmax_score:.5f})",
            f"- paired effect vs default: {decision.effect_size:+.5f} "
            f"(bootstrap SE {decision.bootstrap_se:.5f}, 95% CI "
            f"[{decision.bootstrap_ci95_lo:+.5f}, {decision.bootstrap_ci95_hi:+.5f}], z = {decision.effect_z:.2f}, "
            f"{decision.n_boot} resamples over {decision.n_cases} cases)",
            f"- collateral Dice effect of that argmax: {decision.dice_effect_size:+.5f} "
            f"(bootstrap SE {decision.dice_bootstrap_se:.5f}, z = {decision.dice_effect_z:.2f})",
            f"- gate: {'enabled' if decision.gate_enabled else 'DISABLED'} at "
            f"{decision.gate_se_multiplier:g}x SE",
            f"- decision: **threshold {format_threshold(decision.selected_threshold)}** — {decision.reason}",
            "",
            "## In-sample optima (selection set == report set; optimistic, not a result)",
            "",
            "| score | threshold | Dice mean | Lesion F1 mean | Lesion recall mean | Lesion precision mean | Dice/F1 hmean | pooled lesion F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in best_rows.items():
        lines.append(
            "| {name} | {thr} | {dice:.4f} | {f1:.4f} | {rec:.4f} | {prec:.4f} | {hm:.4f} | {pf1:.4f} |".format(
                name=name,
                thr=format_threshold(float(row["threshold"])),
                dice=float(row["dice_mean"]),
                f1=float(row["lesion_f1_mean"]),
                rec=float(row["lesion_recall_mean"]),
                prec=float(row["lesion_precision_mean"]),
                hm=float(row["dice_lesion_f1_hmean"]),
                pf1=float(row["lesion_f1_pooled"]),
            )
        )

    if nested is not None:
        lines.extend(
            [
                "",
                "## Nested selection detail",
                "",
                "| held-out fold | n held out | threshold chosen on the other folds | held-out Dice mean | held-out Lesion F1 mean |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for entry in nested["per_fold"]:  # type: ignore[index]
            held = entry["held_out_metrics"]  # type: ignore[index]
            lines.append(
                "| {fold} | {n} | {thr} | {dice:.4f} | {f1:.4f} |".format(
                    fold=int(entry["fold"]),  # type: ignore[index]
                    n=int(entry["n_held_out_cases"]),  # type: ignore[index]
                    thr=format_threshold(float(entry["decision"]["selected_threshold"])),  # type: ignore[index]
                    dice=float(held["dice_mean"]),  # type: ignore[index]
                    f1=float(held["lesion_f1_mean"]),  # type: ignore[index]
                )
            )

    lines.extend(
        [
            "",
            "## Sweep",
            "",
            "| threshold | Dice mean | Lesion F1 mean | Lesion recall mean | Lesion precision mean | pooled lesion F1 | pred lesions |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            "| {thr} | {dice:.4f} | {f1:.4f} | {rec:.4f} | {prec:.4f} | {pf1:.4f} | {pred} |".format(
                thr=format_threshold(float(row["threshold"])),
                dice=float(row["dice_mean"]),
                f1=float(row["lesion_f1_mean"]),
                rec=float(row["lesion_recall_mean"]),
                prec=float(row["lesion_precision_mean"]),
                pf1=float(row["lesion_f1_pooled"]),
                pred=int(row["pred_lesion_count_sum"]),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def write_best_masks(cases: list[CaseInput], threshold: float, mask_dir: Path, progress_every: int) -> None:
    if mask_dir.exists():
        for path in mask_dir.glob("ATLAS_*.nii.gz"):
            path.unlink()
    mask_dir.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(cases, start=1):
        prob_img, prob = load_probability(case.prob_path)
        mask = (prob >= threshold).astype(np.uint8, copy=False)
        header = prob_img.header.copy()
        header.set_data_shape(mask.shape)
        header.set_data_dtype(np.uint8)
        out_img = nib.Nifti1Image(mask, prob_img.affine, header)
        out_img.set_data_dtype(np.uint8)
        nib.save(out_img, str(mask_dir / f"{case.case_id}.nii.gz"))
        if progress_every > 0 and (index == 1 or index == len(cases) or index % progress_every == 0):
            print(f"[info] wrote best-threshold mask {index}/{len(cases)}: {case.case_id}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--folds", default="all")
    parser.add_argument("--prob-dir", action="append", required=True, type=Path, help="Directory containing saved lesion-probability ATLAS_*.nii.gz files; repeat for folds.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--thresholds", type=parse_thresholds, default=parse_thresholds(DEFAULT_THRESHOLD_GRID))
    parser.add_argument(
        "--primary-score",
        choices=SCORE_KEYS,
        default="lesion_f1_mean",
    )
    parser.add_argument("--min-overlap-voxels", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--write-best-masks", action="store_true")
    parser.add_argument("--best-mask-dir", type=Path)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--default-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Null-hypothesis operating point the sweep must beat by more than noise.",
    )
    parser.add_argument(
        "--gate-se-multiplier",
        type=float,
        default=1.0,
        help="Deviate from --default-threshold only if the paired effect exceeds this many bootstrap SEs.",
    )
    parser.add_argument("--gate-bootstrap", type=int, default=2000, help="Paired bootstrap resamples for the gate.")
    parser.add_argument("--gate-seed", type=int, default=0)
    parser.add_argument(
        "--no-min-effect-gate",
        action="store_true",
        help="Select by bare in-sample argmax (the pre-audit behaviour); the emitted summary says so.",
    )
    parser.add_argument(
        "--no-nested",
        action="store_true",
        help="Skip the nested (select-on-K-1-folds, score-held-out-fold) honest estimate.",
    )
    parser.add_argument(
        "--allow-grid-edge",
        action="store_true",
        help="Downgrade the 'selected threshold sits on a grid edge' failure to a warning.",
    )
    args = parser.parse_args(argv)

    if args.min_overlap_voxels < 1:
        parser.error("--min-overlap-voxels must be >= 1")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.gate_se_multiplier < 0:
        parser.error("--gate-se-multiplier must be >= 0")
    if args.gate_bootstrap < 0:
        parser.error("--gate-bootstrap must be >= 0")

    manifest = {row["nnunet_id"]: row for row in load_csv(args.manifest)}
    case_folds = {case_id: fold for case_id, fold in load_val_case_folds(args.splits, args.folds)}
    val_cases = list(case_folds)
    prob_paths = collect_prob_paths(args.prob_dir)

    missing_prob = [case_id for case_id in val_cases if case_id not in prob_paths]
    if missing_prob:
        raise SystemExit(f"missing probability NIfTIs for {len(missing_prob)} cases; first={missing_prob[:5]}")
    missing_manifest = [case_id for case_id in val_cases if case_id not in manifest]
    if missing_manifest:
        raise SystemExit(f"missing manifest rows for {len(missing_manifest)} cases; first={missing_manifest[:5]}")
    extra_prob = sorted(set(prob_paths) - set(val_cases))
    if extra_prob:
        print(f"[warn] ignoring {len(extra_prob)} probability NIfTIs outside requested folds; first={extra_prob[:5]}", file=sys.stderr)

    cases = [
        CaseInput(
            case_id=case_id,
            prob_path=prob_paths[case_id],
            gt_path=Path(manifest[case_id].get("lesion_source_path") or manifest[case_id].get("lesion_path") or manifest[case_id].get("label") or ""),
        )
        for case_id in val_cases
    ]
    for case in cases:
        if not case.gt_path.exists():
            raise SystemExit(f"missing GT for {case.case_id}: {case.gt_path}")

    thresholds = args.thresholds
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Keyed by case id so nested selection can hold whole folds out.
    per_case: dict[float, dict[str, dict[str, float | int]]] = {threshold: {} for threshold in thresholds}
    if args.workers == 1:
        for index, case in enumerate(cases, start=1):
            case_id, rows = evaluate_case(case, thresholds, args.min_overlap_voxels)
            for row in rows:
                per_case[float(row["threshold"])][case_id] = {"case_id": case_id, **row}
            if args.progress_every > 0 and (index == 1 or index == len(cases) or index % args.progress_every == 0):
                print(f"[info] swept {index}/{len(cases)} cases; latest={case_id}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(evaluate_case, case, thresholds, args.min_overlap_voxels) for case in cases]
            completed = 0
            for future in as_completed(futures):
                case_id, rows = future.result()
                for row in rows:
                    per_case[float(row["threshold"])][case_id] = {"case_id": case_id, **row}
                completed += 1
                if args.progress_every > 0 and (completed == 1 or completed == len(cases) or completed % args.progress_every == 0):
                    print(f"[info] swept {completed}/{len(cases)} cases; latest={case_id}", flush=True)

    summary = aggregate_by_threshold(rows_by_threshold(per_case))
    best_rows = {score_key: best_row(summary, score_key) for score_key in SCORE_KEYS}

    decision = select_threshold(
        summary,
        per_case,
        args.primary_score,
        default_threshold=args.default_threshold,
        gate_se_multiplier=args.gate_se_multiplier,
        n_boot=args.gate_bootstrap,
        seed=args.gate_seed,
        gate_enabled=not args.no_min_effect_gate,
    )
    selected_threshold = decision.selected_threshold

    n_folds = len({int(f) for f in case_folds.values()})
    nested: dict[str, object] | None = None
    if args.no_nested:
        print("[warn] --no-nested: the reported numbers are in-sample and optimistic", file=sys.stderr)
    elif n_folds < 2:
        print(
            f"[warn] only {n_folds} fold requested; nested selection needs >= 2, so no honest estimate is available "
            "and every number in this sweep is in-sample",
            file=sys.stderr,
        )
    else:
        nested = nested_selection(
            per_case,
            case_folds,
            args.primary_score,
            default_threshold=args.default_threshold,
            gate_se_multiplier=args.gate_se_multiplier,
            n_boot=args.gate_bootstrap,
            seed=args.gate_seed,
            gate_enabled=not args.no_min_effect_gate,
        )
        edge_folds = [
            int(entry["fold"]) for entry in nested["per_fold"] if entry["decision"]["selected_on_grid_edge"]  # type: ignore[index]
        ]
        if edge_folds:
            print(
                f"[warn] nested selection landed on a grid edge for fold(s) {edge_folds}; the grid may be too narrow",
                file=sys.stderr,
            )

    write_csv(args.out_dir / "sweep_summary.csv", summary)
    payload = {
        "n_cases": len(cases),
        "n_folds": n_folds,
        "thresholds": [format_threshold(v) for v in thresholds],
        "primary_score": args.primary_score,
        "selected_threshold": selected_threshold,
        "selection": asdict(decision),
        # Kept under the historical key for consumers, but these are argmax rows
        # scored on the very cases that chose them.
        "best": best_rows,
        "best_note": "IN-SAMPLE argmax rows: selection set == report set, so these overstate performance.",
        "nested": nested,
        "headline": "nested.held_out_metrics" if nested is not None else "none (in-sample only)",
        "summary": summary,
    }
    (args.out_dir / "sweep_summary.json").write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n")
    write_markdown(args.out_dir / "sweep_summary.md", summary, best_rows, args.primary_score, decision, nested)
    (args.out_dir / "best_threshold.txt").write_text(format_threshold(selected_threshold) + "\n")

    if args.write_best_masks:
        mask_dir = args.best_mask_dir or (args.out_dir / "best_masks")
        print(f"[info] writing masks for selected threshold {format_threshold(selected_threshold)} to {mask_dir}", flush=True)
        write_best_masks(cases, selected_threshold, mask_dir, args.progress_every)

    print("[done] threshold sweep")
    if nested is not None:
        held = nested["held_out_metrics"]  # type: ignore[index]
        fixed = nested["fixed_default_metrics"]  # type: ignore[index]
        print(
            "[honest] nested {k}-fold selection: {score}={val:.5f} dice={dice:.5f} "
            "(fixed default {thr}: {score}={dval:.5f} dice={ddice:.5f}; honest gain {gain:+.5f})".format(
                k=nested["n_folds"],  # type: ignore[index]
                score=args.primary_score,
                val=float(held[args.primary_score]),  # type: ignore[index]
                dice=float(held["dice_mean"]),  # type: ignore[index]
                thr=format_threshold(decision.default_threshold),
                dval=float(fixed[args.primary_score]),  # type: ignore[index]
                ddice=float(fixed["dice_mean"]),  # type: ignore[index]
                gain=float(nested["gain_over_default"]),  # type: ignore[index]
            )
        )
    print(f"[decision] selected threshold ({args.primary_score}): {format_threshold(selected_threshold)} — {decision.reason}")
    print("[in-sample] argmax rows below were scored on the cases that selected them; do not report them as results")
    for name, row in best_rows.items():
        print(
            "{name}: threshold={thr} dice={dice:.4f} lesion_f1={f1:.4f} recall={rec:.4f} precision={prec:.4f}".format(
                name=name,
                thr=format_threshold(float(row["threshold"])),
                dice=float(row["dice_mean"]),
                f1=float(row["lesion_f1_mean"]),
                rec=float(row["lesion_recall_mean"]),
                prec=float(row["lesion_precision_mean"]),
            )
        )

    if decision.argmax_on_grid_edge and not decision.selected_on_grid_edge:
        print(
            f"[warn] in-sample argmax {format_threshold(decision.argmax_threshold)} sits on a grid edge; the metric may "
            "still be improving outside the searched window",
            file=sys.stderr,
        )
    if decision.selected_on_grid_edge:
        # An edge optimum means the search window, not the data, decided the
        # answer: the true optimum may lie outside the grid entirely.
        message = (
            f"selected threshold {format_threshold(selected_threshold)} sits on the edge of the searched grid "
            f"[{format_threshold(thresholds[0])}, {format_threshold(thresholds[-1])}]; widen --thresholds so the "
            "optimum is interior (or pass --allow-grid-edge to accept it)"
        )
        if args.allow_grid_edge:
            print(f"[warn] {message}", file=sys.stderr)
        else:
            print(f"[error] {message}", file=sys.stderr)
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
