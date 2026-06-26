#!/usr/bin/env python3
"""Sweep binary thresholds over saved lesion-probability NIfTIs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage as ndi


CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


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


def load_val_cases(path: Path, folds_arg: str) -> list[str]:
    payload = json.loads(path.read_text())
    folds = payload.get("folds", payload) if isinstance(payload, dict) else payload
    if not isinstance(folds, list):
        raise SystemExit(f"could not read folds from {path}")
    selected_folds = parse_folds(folds_arg, len(folds))

    val_cases: list[str] = []
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
            val_cases.append(case_id)
    if not val_cases:
        raise SystemExit(f"requested validation case list is empty in {path}")
    return val_cases


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


def aggregate_by_threshold(per_case_by_threshold: dict[float, list[dict[str, float | int]]]) -> list[dict[str, float | int]]:
    summary: list[dict[str, float | int]] = []
    for threshold in sorted(per_case_by_threshold):
        rows = per_case_by_threshold[threshold]
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
        summary.append(
            {
                "threshold": threshold,
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
        )
    return summary


def best_row(summary: list[dict[str, float | int]], score_key: str) -> dict[str, float | int]:
    if score_key == "dice_lesion_f1_hmean":
        return max(summary, key=lambda row: (float(row[score_key]), float(row["lesion_f1_mean"]), float(row["dice_mean"])))
    if score_key == "lesion_f1_pooled":
        return max(summary, key=lambda row: (float(row[score_key]), float(row["lesion_f1_mean"]), float(row["dice_mean"])))
    return max(summary, key=lambda row: (float(row[score_key]), float(row["dice_mean"]), float(row["lesion_recall_mean"])))


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


def write_markdown(path: Path, summary: list[dict[str, float | int]], best_rows: dict[str, dict[str, float | int]], primary_score: str) -> None:
    lines = [
        "# Probability Threshold Sweep",
        "",
        f"Primary selection metric: `{primary_score}`.",
        "",
        "## Best Operating Points",
        "",
        "| score | threshold | Dice mean | Lesion F1 mean | Lesion recall mean | Lesion precision mean | Dice/F1 hmean | pooled lesion F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
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
    parser.add_argument("--thresholds", type=parse_thresholds, default=parse_thresholds("0.35:0.55:0.01"))
    parser.add_argument(
        "--primary-score",
        choices=("lesion_f1_mean", "lesion_f1_pooled", "dice_lesion_f1_hmean"),
        default="lesion_f1_mean",
    )
    parser.add_argument("--min-overlap-voxels", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--write-best-masks", action="store_true")
    parser.add_argument("--best-mask-dir", type=Path)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args(argv)

    if args.min_overlap_voxels < 1:
        parser.error("--min-overlap-voxels must be >= 1")
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    manifest = {row["nnunet_id"]: row for row in load_csv(args.manifest)}
    val_cases = load_val_cases(args.splits, args.folds)
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

    per_case_by_threshold: dict[float, list[dict[str, float | int]]] = {threshold: [] for threshold in thresholds}
    if args.workers == 1:
        for index, case in enumerate(cases, start=1):
            case_id, rows = evaluate_case(case, thresholds, args.min_overlap_voxels)
            for row in rows:
                per_case_by_threshold[float(row["threshold"])].append(row)
            if args.progress_every > 0 and (index == 1 or index == len(cases) or index % args.progress_every == 0):
                print(f"[info] swept {index}/{len(cases)} cases; latest={case_id}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(evaluate_case, case, thresholds, args.min_overlap_voxels) for case in cases]
            completed = 0
            for future in as_completed(futures):
                case_id, rows = future.result()
                for row in rows:
                    per_case_by_threshold[float(row["threshold"])].append(row)
                completed += 1
                if args.progress_every > 0 and (completed == 1 or completed == len(cases) or completed % args.progress_every == 0):
                    print(f"[info] swept {completed}/{len(cases)} cases; latest={case_id}", flush=True)

    summary = aggregate_by_threshold(per_case_by_threshold)
    best_rows = {
        "lesion_f1_mean": best_row(summary, "lesion_f1_mean"),
        "dice_lesion_f1_hmean": best_row(summary, "dice_lesion_f1_hmean"),
        "lesion_f1_pooled": best_row(summary, "lesion_f1_pooled"),
    }
    selected = best_rows[args.primary_score]
    selected_threshold = float(selected["threshold"])

    write_csv(args.out_dir / "sweep_summary.csv", summary)
    payload = {
        "n_cases": len(cases),
        "thresholds": [format_threshold(v) for v in thresholds],
        "primary_score": args.primary_score,
        "selected_threshold": selected_threshold,
        "best": best_rows,
        "summary": summary,
    }
    (args.out_dir / "sweep_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    write_markdown(args.out_dir / "sweep_summary.md", summary, best_rows, args.primary_score)
    (args.out_dir / "best_threshold.txt").write_text(format_threshold(selected_threshold) + "\n")

    if args.write_best_masks:
        mask_dir = args.best_mask_dir or (args.out_dir / "best_masks")
        print(f"[info] writing masks for selected threshold {format_threshold(selected_threshold)} to {mask_dir}", flush=True)
        write_best_masks(cases, selected_threshold, mask_dir, args.progress_every)

    print("[done] threshold sweep")
    print(f"selected threshold ({args.primary_score}): {format_threshold(selected_threshold)}")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
