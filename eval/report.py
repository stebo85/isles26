"""Aggregate per-case metrics into stratified reports.

The output is both a JSON object (for re-aggregation later) and a Markdown
report that highlights:

- overall mean/median Dice, IoU, lesion-wise F1, HD95, ASSD, AVD
- per-chronicity breakdown (acute vs chronic vs mixed)
- per-center breakdown
- size-binned recall (small-lesion sensitivity)
- per-case rank table (sorted by Dice ascending so worst cases are visible)
- worst-case audit (5 lowest Dice and 5 lowest lesion F1)

Aggregation rules
- We aggregate with `nanmean` and `nanmedian` to skip cases where a metric is
  undefined (e.g. HD95 on empty masks). Counts of valid cases are reported
  alongside each aggregate.
- For per-bin recall, we sum n_gt and n_detected across cases, then divide.
  This avoids the "average of ratios" pitfall when bin counts are small per
  case (one case with 1/1 detected would otherwise have the same weight as a
  case with 50/100).
"""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np


def _safe_mean(xs: Iterable[float]) -> float:
    arr = np.asarray([x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def _safe_median(xs: Iterable[float]) -> float:
    arr = np.asarray([x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=np.float64)
    return float(np.median(arr)) if arr.size else float("nan")


def _summary_metric(name: str, values: list[float]) -> dict:
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=np.float64)
    if arr.size == 0:
        return {"metric": name, "n_valid": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "metric": name,
        "n_valid": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def aggregate(cases: list[dict]) -> dict:
    """Build the aggregate report from a list of CaseMetrics.to_dict()."""

    def collect(path: list[str]) -> list[float]:
        out = []
        for c in cases:
            v = c
            for k in path:
                v = v.get(k, None) if isinstance(v, dict) else None
                if v is None:
                    break
            if isinstance(v, (int, float)):
                out.append(float(v))
            else:
                out.append(float("nan"))
        return out

    metric_paths = [
        ("dice", ["overlap", "dice"]),
        ("iou", ["overlap", "iou"]),
        ("sensitivity", ["overlap", "sensitivity"]),
        ("precision", ["overlap", "precision"]),
        ("specificity", ["overlap", "specificity"]),
        ("hd95_mm", ["boundary", "hd95_mm"]),
        ("assd_mm", ["boundary", "assd_mm"]),
        ("surface_dice_3mm", ["boundary", "surface_dice_3mm"]),
        ("abs_volume_diff_ml", ["volume", "abs_volume_diff_ml"]),
        ("rel_abs_volume_diff", ["volume", "relative_abs_volume_diff"]),
        ("lesion_f1", ["lesionwise", "lesion_f1"]),
        ("lesion_precision", ["lesionwise", "lesion_precision"]),
        ("lesion_recall", ["lesionwise", "lesion_recall"]),
        ("abs_lesion_count_diff", ["lesionwise", "abs_lesion_count_diff"]),
    ]
    overall = [_summary_metric(name, collect(path)) for name, path in metric_paths]

    # Size-binned recall: pooled across cases (sum n_gt, sum n_detected).
    bins_pooled_vox = defaultdict(lambda: {"n_gt": 0, "n_detected": 0})
    bins_pooled_ml = defaultdict(lambda: {"n_gt": 0, "n_detected": 0})
    for c in cases:
        for bin_name, info in c.get("size_bins_voxels", {}).items():
            bins_pooled_vox[bin_name]["n_gt"] += info["n_gt"]
            bins_pooled_vox[bin_name]["n_detected"] += info["n_detected"]
        for bin_name, info in c.get("size_bins_ml", {}).items():
            bins_pooled_ml[bin_name]["n_gt"] += info["n_gt"]
            bins_pooled_ml[bin_name]["n_detected"] += info["n_detected"]

    def _finalize_bins(d):
        return {
            k: {
                "n_gt": v["n_gt"],
                "n_detected": v["n_detected"],
                "recall": (v["n_detected"] / v["n_gt"]) if v["n_gt"] > 0 else float("nan"),
            }
            for k, v in d.items()
        }

    # Stratified by chronicity and center.
    def _group_by(key: str) -> dict[str, dict]:
        groups: dict[str, list[dict]] = defaultdict(list)
        for c in cases:
            groups[c.get(key) or "unknown"].append(c)
        return {g: aggregate_minimal(v) for g, v in groups.items()}

    return {
        "n_cases": len(cases),
        "overall": overall,
        "size_bins_voxels_pooled": _finalize_bins(bins_pooled_vox),
        "size_bins_ml_pooled": _finalize_bins(bins_pooled_ml),
        "by_chronicity": _group_by("chronicity"),
        "by_center": _group_by("center"),
        "per_case": [
            {
                "subject_id": c["subject_id"],
                "chronicity": c.get("chronicity"),
                "center": c.get("center"),
                "dice": c["overlap"]["dice"],
                "iou": c["overlap"]["iou"],
                "sensitivity": c["overlap"]["sensitivity"],
                "precision": c["overlap"]["precision"],
                "lesion_f1": c["lesionwise"]["lesion_f1"],
                "lesion_recall": c["lesionwise"]["lesion_recall"],
                "lesion_precision": c["lesionwise"]["lesion_precision"],
                "hd95_mm": c["boundary"]["hd95_mm"],
                "assd_mm": c["boundary"]["assd_mm"],
                "surface_dice_3mm": c["boundary"]["surface_dice_3mm"],
                "abs_volume_diff_ml": c["volume"]["abs_volume_diff_ml"],
                "pred_vol_ml": c["volume"]["pred_vol_ml"],
                "gt_vol_ml": c["volume"]["gt_vol_ml"],
                "gt_lesion_count": c["lesionwise"]["gt_lesion_count"],
                "pred_lesion_count": c["lesionwise"]["pred_lesion_count"],
            }
            for c in cases
        ],
    }


def aggregate_minimal(cases: list[dict]) -> dict:
    """Smaller aggregate used inside stratified reports."""
    if not cases:
        return {"n": 0}
    dice = [c["overlap"]["dice"] for c in cases]
    f1 = [c["lesionwise"]["lesion_f1"] for c in cases]
    recall = [c["lesionwise"]["lesion_recall"] for c in cases]
    hd95 = [c["boundary"]["hd95_mm"] for c in cases]
    avd = [c["volume"]["abs_volume_diff_ml"] for c in cases]
    return {
        "n": len(cases),
        "dice_mean": _safe_mean(dice),
        "dice_median": _safe_median(dice),
        "lesion_f1_mean": _safe_mean(f1),
        "lesion_recall_mean": _safe_mean(recall),
        "hd95_mm_mean": _safe_mean(hd95),
        "abs_volume_diff_ml_mean": _safe_mean(avd),
    }


def write_json(report: dict, out_path: Path) -> None:
    out_path.write_text(json.dumps(report, indent=2, default=str))


def write_markdown(report: dict, out_path: Path, title: str = "Benchmark report") -> None:
    lines = [f"# {title}", "", f"Subjects evaluated: **{report['n_cases']}**", ""]

    lines.append("## Overall metrics")
    lines.append("")
    lines.append("| metric | n_valid | mean | median | std | min | max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for m in report["overall"]:
        lines.append("| {metric} | {n_valid} | {mean:.4f} | {median:.4f} | {std:.4f} | {min:.4f} | {max:.4f} |".format(**m))
    lines.append("")

    lines.append("## Size-binned lesion recall (pooled across cases, voxels)")
    lines.append("")
    lines.append("| bin | n_GT lesions | n_detected | recall |")
    lines.append("|---|---:|---:|---:|")
    for name, info in report["size_bins_voxels_pooled"].items():
        recall_str = f"{info['recall']:.3f}" if not math.isnan(info["recall"]) else "n/a"
        lines.append(f"| {name} | {info['n_gt']} | {info['n_detected']} | {recall_str} |")
    lines.append("")

    lines.append("## Size-binned lesion recall (pooled across cases, mL)")
    lines.append("")
    lines.append("| bin | n_GT lesions | n_detected | recall |")
    lines.append("|---|---:|---:|---:|")
    for name, info in report["size_bins_ml_pooled"].items():
        recall_str = f"{info['recall']:.3f}" if not math.isnan(info["recall"]) else "n/a"
        lines.append(f"| {name} | {info['n_gt']} | {info['n_detected']} | {recall_str} |")
    lines.append("")

    lines.append("## By chronicity")
    lines.append("")
    lines.append("| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for grp, info in report["by_chronicity"].items():
        lines.append("| {grp} | {n} | {d_m:.4f} | {d_med:.4f} | {f1:.4f} | {rec:.4f} | {hd95:.3f} | {avd:.3f} |".format(
            grp=grp, n=info["n"], d_m=info["dice_mean"], d_med=info["dice_median"],
            f1=info["lesion_f1_mean"], rec=info["lesion_recall_mean"],
            hd95=info["hd95_mm_mean"], avd=info["abs_volume_diff_ml_mean"]))
    lines.append("")

    lines.append("## By center (Manufacturer | Model)")
    lines.append("")
    lines.append("| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for grp, info in report["by_center"].items():
        lines.append("| {grp} | {n} | {d:.4f} | {f1:.4f} | {hd95:.3f} | {avd:.3f} |".format(
            grp=grp, n=info["n"], d=info["dice_mean"],
            f1=info["lesion_f1_mean"], hd95=info["hd95_mm_mean"],
            avd=info["abs_volume_diff_ml_mean"]))
    lines.append("")

    lines.append("## Per-case table (sorted by Dice, worst first)")
    lines.append("")
    lines.append("| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    rows = sorted(report["per_case"], key=lambda r: (math.isnan(r["dice"]), r["dice"]))
    for r in rows:
        lines.append(
            "| {sid} | {chr} | {ctr} | {d:.4f} | {f1:.4f} | {rec:.4f} | {hd:.3f} | {pv:.3f} | {gv:.3f} | {gc} | {pc} |".format(
                sid=r["subject_id"],
                chr=r["chronicity"] or "?",
                ctr=(r["center"] or "?"),
                d=r["dice"], f1=r["lesion_f1"], rec=r["lesion_recall"],
                hd=r["hd95_mm"], pv=r["pred_vol_ml"], gv=r["gt_vol_ml"],
                gc=r["gt_lesion_count"], pc=r["pred_lesion_count"],
            )
        )
    out_path.write_text("\n".join(lines))
