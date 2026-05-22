#!/usr/bin/env python3
"""Sweep (prob-threshold x min-cc-voxels) over pre-saved ATLAS-val lesion
probability maps and select a post-processing operating point.

IMPORTANT: This script tunes exclusively on the ATLAS fold-0 validation set
(never on the soop/test set) to avoid leaderboard overfitting. It reuses
eval/metrics.py for all metric computations to ensure parity with the
official evaluation pipeline.

Operating-point selection guard: the chosen (thr, cc) must retain small-lesion
(0–1 mL) pooled recall >= --small-recall-tolerance × the fixed reference recall
computed at threshold=0.5 / cc=0 (the deployment default, no post-processing).
This prevents high thresholds that already destroy small-lesion recall from
winning on lesion-F1 alone.

Data validation: all fold-0 val cases enumerated from --splits must have a
matching prob map and GT mask with equal shape AND matching affine before any
metric is computed. Missing or mismatched cases raise SystemExit unless
--allow-missing is set, in which case they are skipped and recorded in the
output JSON under "skipped_cases".

Outputs written to --out-dir:
  postproc_operating_point.json  -- chosen (thr, cc) + metrics + skipped_cases
  sweep.csv                      -- full grid results
  sweep.md                       -- human-readable table

Usage example:
  python tune_postproc.py \\
    --prob-dir work/synthstroke/pred/atlas_val_prob/our_isles26_fs_synth_mixreal \\
    --splits work/nnunet/nnUNet_preprocessed/Dataset501_ATLASR21/splits_final.json \\
    --labels-dir work/nnunet/nnUNet_raw/Dataset501_ATLASR21/labelsTr \\
    --out-dir baselines/reports/synthstroke/our_isles26_fs_synth_mixreal/postproc_tuning
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import nibabel as nib
import scipy.ndimage

# Reuse the repo's metric functions for full parity with the evaluation pipeline.
from eval.metrics import (
    connected_components,
    evaluate_case,
    lesion_wise_metrics,
    size_binned_recall_ml,
)

_CC26 = np.ones((3, 3, 3), dtype=np.uint8)  # 26-connectivity, matches eval/metrics.py


def load_prob_nii(path: Path) -> nib.Nifti1Image:
    """Load a float32 probability NIfTI and return the image object (affine preserved)."""
    return nib.load(str(path))


def load_gt_nii(path: Path) -> nib.Nifti1Image:
    """Load a GT mask NIfTI and return the image object (affine preserved)."""
    return nib.load(str(path))


def prob_data(img: nib.Nifti1Image) -> np.ndarray:
    """Extract float32 array from a prob NIfTI image."""
    return np.squeeze(np.asanyarray(img.dataobj).astype(np.float32))


def gt_data(img: nib.Nifti1Image) -> np.ndarray:
    """Extract binarised GT array (MANDATORY: some ATLAS masks are float ~0.9999)."""
    return (np.squeeze(np.asanyarray(img.dataobj)) > 0.5).astype(np.uint8)


def get_voxel_size(img: nib.Nifti1Image) -> tuple[float, float, float]:
    zooms = img.header.get_zooms()[:3]
    return (float(zooms[0]), float(zooms[1]), float(zooms[2]))


def apply_cc_filter(mask: np.ndarray, min_cc_voxels: int) -> np.ndarray:
    """Remove connected components < min_cc_voxels using 26-connectivity."""
    if min_cc_voxels <= 0 or not mask.any():
        return mask
    labeled, n_comp = scipy.ndimage.label(mask, structure=_CC26)
    if n_comp == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    remove = np.where(sizes < min_cc_voxels)[0]
    remove = remove[remove > 0]
    if remove.size == 0:
        return mask
    out = mask.copy()
    out[np.isin(labeled, remove)] = 0
    return out


def compute_case_metrics(prob: np.ndarray, gt: np.ndarray,
                         voxel_size: tuple[float, float, float],
                         thr: float, cc: int) -> dict:
    """Binarise prob, apply CC, compute metrics. Returns dict with dice, lesion_f1,
    and small-lesion (0-1 mL) recall."""
    mask = (prob > thr).astype(np.uint8)
    mask = apply_cc_filter(mask, cc)

    lw = lesion_wise_metrics(mask, gt)
    # Dice
    p_sum = int(mask.sum())
    g_sum = int(gt.sum())
    if p_sum == 0 and g_sum == 0:
        dice = 1.0
    elif p_sum == 0 or g_sum == 0:
        dice = 0.0
    else:
        tp = int((mask.astype(bool) & gt.astype(bool)).sum())
        dice = 2 * tp / (2 * tp + int((mask & ~gt.astype(bool)).sum()) + int((~mask.astype(bool) & gt).sum()))

    # Small-lesion (0-1 mL) recall
    bins_ml = size_binned_recall_ml(lw, voxel_size)
    small_n = 0
    small_det = 0
    for name, info in bins_ml.items():
        # bins matching 0-1 mL: "vsmall_<0.1mL" and "small_0.1-1mL"
        if "<0.1mL" in name or "0.1-1mL" in name:
            n = info["n_gt"]
            det = info["n_detected"]
            small_n += n
            small_det += det
    small_recall = (small_det / small_n) if small_n > 0 else float("nan")

    return {
        "dice": float(dice),
        "lesion_f1": float(lw.lesion_f1),
        "small_recall": float(small_recall),
        "small_n_gt": small_n,
        "small_n_det": small_det,
    }


def _pooled_small_recall(loaded: list[tuple[str, np.ndarray, np.ndarray, tuple]],
                         thr: float, cc: int) -> float:
    """Compute pooled (aggregate) small-lesion recall across all cases."""
    total_n = 0
    total_det = 0
    for case_id, prob, gt, voxel_size in loaded:
        m = compute_case_metrics(prob, gt, voxel_size, thr, cc)
        total_n += m["small_n_gt"]
        total_det += m["small_n_det"]
    return (total_det / total_n) if total_n > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--prob-dir", required=True, type=Path,
                    help="directory of per-case lesion prob NIfTIs named <case>.nii.gz")
    ap.add_argument("--splits", required=True, type=Path,
                    help="nnUNet splits_final.json; fold 0 'val' is used")
    ap.add_argument("--labels-dir",
                    default="work/nnunet/nnUNet_raw/Dataset501_ATLASR21/labelsTr",
                    type=Path,
                    help="directory with GT masks named <case>.nii.gz")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--thresholds", default="0.3,0.4,0.5,0.6",
                    help="comma-separated probability thresholds to sweep")
    ap.add_argument("--cc-voxels", default="0,5,10,20,50",
                    help="comma-separated min-cc-voxels values to sweep")
    ap.add_argument("--small-recall-tolerance", type=float, default=0.95,
                    help="min fraction of the FIXED reference small-lesion recall "
                         "(thr=0.5, cc=0) that the selected (thr, cc) must retain "
                         "(default 0.95)")
    ap.add_argument("--allow-missing", action="store_true", default=False,
                    help="if set, skip cases that fail validation checks (missing "
                         "files, load errors, shape/affine mismatches) instead of "
                         "raising SystemExit; skipped cases are recorded in "
                         "postproc_operating_point.json under 'skipped_cases'")
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",")]
    cc_values = [int(x) for x in args.cc_voxels.split(",")]

    # ------------------------------------------------------------------
    # Load fold-0 val cases — fail closed unless --allow-missing is set
    # ------------------------------------------------------------------
    splits_data = json.loads(args.splits.read_text())
    folds = splits_data if isinstance(splits_data, list) else splits_data["folds"]
    val_cases: list[str] = list(folds[0]["val"])

    # FIX 1 & 2: validate each case strictly; collect failures
    failed: dict[str, str] = {}   # case_id -> reason
    loaded: list[tuple[str, np.ndarray, np.ndarray, tuple]] = []

    for case_id in val_cases:
        prob_path = args.prob_dir / f"{case_id}.nii.gz"
        gt_path = args.labels_dir / f"{case_id}.nii.gz"

        # (a) prob map exists
        if not prob_path.exists():
            failed[case_id] = f"prob map not found: {prob_path}"
            continue

        # (b) GT mask exists
        if not gt_path.exists():
            failed[case_id] = f"GT mask not found: {gt_path}"
            continue

        # (c) both load successfully
        try:
            prob_img = load_prob_nii(prob_path)
        except Exception as exc:
            failed[case_id] = f"prob map load error: {exc}"
            continue
        try:
            gt_img = load_gt_nii(gt_path)
        except Exception as exc:
            failed[case_id] = f"GT mask load error: {exc}"
            continue

        prob_arr = prob_data(prob_img)
        gt_arr = gt_data(gt_img)

        # (d) shapes match
        if prob_arr.shape != gt_arr.shape:
            failed[case_id] = (
                f"shape mismatch: prob={prob_arr.shape} gt={gt_arr.shape}"
            )
            continue

        # (e) affines match (FIX 2)
        if not np.allclose(prob_img.affine, gt_img.affine, atol=1e-3):
            failed[case_id] = (
                f"affine mismatch: prob affine differs from GT affine "
                f"(max_diff={np.abs(prob_img.affine - gt_img.affine).max():.6f})"
            )
            continue

        voxel_size = get_voxel_size(gt_img)
        loaded.append((case_id, prob_arr, gt_arr, voxel_size))

    # Handle failures according to --allow-missing
    if failed:
        if not args.allow_missing:
            lines = [
                "[error] validation failed for the following cases "
                f"({len(failed)}/{len(val_cases)}):"
            ]
            for cid, reason in failed.items():
                lines.append(f"  {cid}: {reason}")
            lines.append(
                "Re-run with --allow-missing to skip these cases and tune on "
                "the available subset."
            )
            raise SystemExit("\n".join(lines))
        else:
            print(
                f"\n[WARNING] --allow-missing: skipping {len(failed)} of "
                f"{len(val_cases)} fold-0 val cases due to validation failures:",
                file=sys.stderr,
            )
            for cid, reason in failed.items():
                print(f"  {cid}: {reason}", file=sys.stderr)
            print("[WARNING] Results are based on an INCOMPLETE subset.\n",
                  file=sys.stderr)

    skipped_cases: list[dict[str, str]] = [
        {"case_id": cid, "reason": reason} for cid, reason in failed.items()
    ]

    if not loaded:
        raise SystemExit("[error] no cases survived validation; check --prob-dir and --labels-dir")

    print(f"[tune_postproc] {len(loaded)}/{len(val_cases)} fold-0 val cases passed validation")

    # ------------------------------------------------------------------
    # FIX 4: compute FIXED reference small-lesion recall at thr=0.5, cc=0
    # ------------------------------------------------------------------
    DEPLOY_THR = 0.5
    DEPLOY_CC = 0
    fixed_ref_small_recall = _pooled_small_recall(loaded, thr=DEPLOY_THR, cc=DEPLOY_CC)
    print(f"[tune_postproc] fixed reference small-lesion recall "
          f"(thr={DEPLOY_THR}, cc={DEPLOY_CC}): {fixed_ref_small_recall:.4f}")

    # ------------------------------------------------------------------
    # Grid sweep
    # ------------------------------------------------------------------
    rows = []
    for thr in thresholds:
        for cc in cc_values:
            case_results = []
            for case_id, prob, gt, voxel_size in loaded:
                m = compute_case_metrics(prob, gt, voxel_size, thr, cc)
                m["case_id"] = case_id
                case_results.append(m)

            mean_dice = float(np.nanmean([r["dice"] for r in case_results]))
            mean_f1 = float(np.nanmean([r["lesion_f1"] for r in case_results]))
            # Pooled (aggregate) small-lesion recall across all cases
            total_small_n = sum(r["small_n_gt"] for r in case_results)
            total_small_det = sum(r["small_n_det"] for r in case_results)
            pooled_small_recall = (total_small_det / total_small_n) if total_small_n > 0 else float("nan")

            rows.append({
                "thr": thr,
                "cc": cc,
                "mean_dice": mean_dice,
                "mean_lesion_f1": mean_f1,
                "pooled_small_recall_0_1ml": pooled_small_recall,
                "n_cases": len(case_results),
            })
            print(f"  thr={thr:.2f} cc={cc:3d} | dice={mean_dice:.4f} f1={mean_f1:.4f} "
                  f"small_recall={pooled_small_recall:.4f}")

    # ------------------------------------------------------------------
    # FIX 4: Select operating point using FIXED reference
    # ------------------------------------------------------------------
    tol = args.small_recall_tolerance
    min_recall = (tol * fixed_ref_small_recall
                  if not math.isnan(fixed_ref_small_recall) else float("nan"))

    eligible = []
    for row in rows:
        row_recall = row["pooled_small_recall_0_1ml"]
        passes = (
            not math.isnan(min_recall)
            and not math.isnan(row_recall)
            and row_recall >= min_recall
        )
        row["recall_passes_filter"] = passes
        if passes:
            eligible.append(row)

    if not eligible:
        print("[warn] no row passes small-recall filter; relaxing to all rows", file=sys.stderr)
        eligible = rows

    best = max(eligible, key=lambda r: r["mean_lesion_f1"])
    print(f"\n[tune_postproc] BEST: thr={best['thr']:.2f} cc={best['cc']} "
          f"f1={best['mean_lesion_f1']:.4f} dice={best['mean_dice']:.4f} "
          f"small_recall={best['pooled_small_recall_0_1ml']:.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Write operating point JSON
    op: dict[str, Any] = {
        "prob_threshold": best["thr"],
        "min_cc_voxels": best["cc"],
        "mean_lesion_f1": best["mean_lesion_f1"],
        "mean_dice": best["mean_dice"],
        "pooled_small_recall_0_1ml": best["pooled_small_recall_0_1ml"],
        "n_cases": best["n_cases"],
        "small_recall_tolerance": tol,
        "fixed_ref_small_recall": fixed_ref_small_recall,
        "selected_small_recall": best["pooled_small_recall_0_1ml"],
        "note": (
            "Tuned on ATLAS fold-0 val only (never soop) to avoid leaderboard overfit. "
            "fixed_ref_small_recall is the pooled small-lesion recall at thr=0.5, cc=0 "
            "(deployment default); selected (thr, cc) must retain >= "
            f"{tol} x fixed_ref_small_recall."
        ),
    }
    if skipped_cases:
        op["skipped_cases"] = skipped_cases
    op_path = args.out_dir / "postproc_operating_point.json"
    op_path.write_text(json.dumps(op, indent=2) + "\n")
    print(f"[tune_postproc] operating point -> {op_path}")

    # Write sweep CSV
    fieldnames = ["thr", "cc", "mean_dice", "mean_lesion_f1",
                  "pooled_small_recall_0_1ml",
                  "recall_passes_filter", "n_cases"]
    csv_path = args.out_dir / "sweep.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[tune_postproc] sweep table -> {csv_path}")

    # Write sweep Markdown (FIX 3: precompute display strings to avoid invalid
    # conditional-inside-format-spec f-string patterns)
    ref_s = f"{fixed_ref_small_recall:.4f}" if not math.isnan(fixed_ref_small_recall) else "nan"
    md_lines = [
        "# Post-processing sweep results",
        "",
        "> Tuned on ATLAS fold-0 val only (never soop) to avoid leaderboard overfit.",
        "> Reuses eval/metrics.py for metric parity with official evaluation.",
        f"> Fixed reference small-lesion recall (thr=0.5, cc=0): {ref_s}",
        "",
        f"**Selected operating point:** prob_threshold={best['thr']:.2f}  "
        f"min_cc_voxels={best['cc']}  "
        f"lesion_F1={best['mean_lesion_f1']:.4f}  "
        f"Dice={best['mean_dice']:.4f}  "
        f"small_recall={best['pooled_small_recall_0_1ml']:.4f}",
        "",
        "| thr | cc | mean_dice | mean_lesion_f1 | small_recall_0-1mL | passes |",
        "|----:|---:|----------:|---------------:|-------------------:|:------:|",
    ]
    for row in rows:
        check = "Y" if row.get("recall_passes_filter") else "N"
        # FIX 3: precompute display string; never embed conditional inside format spec
        sr_s = (f"{row['pooled_small_recall_0_1ml']:.4f}"
                if not math.isnan(row["pooled_small_recall_0_1ml"]) else "nan")
        md_lines.append(
            f"| {row['thr']:.2f} | {row['cc']:3d} | "
            f"{row['mean_dice']:.4f} | {row['mean_lesion_f1']:.4f} | "
            f"{sr_s} | {check} |"
        )
    md_path = args.out_dir / "sweep.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"[tune_postproc] sweep markdown -> {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
