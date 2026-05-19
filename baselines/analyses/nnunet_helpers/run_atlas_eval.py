#!/usr/bin/env python3
"""Evaluate nnU-Net predictions on ATLAS R2.1 fold-0 validation cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib

from eval.loader import load_mask, resample_to_reference
from eval.metrics import DEFAULT_SIZE_BINS_ML, evaluate_case
from eval.qc_viz import make_overlay
from eval.report import aggregate, aggregate_minimal, write_json, write_markdown


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "na", "n/a"}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def parse_bool(value: object) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1", "1.0"}:
        return True
    if text in {"false", "f", "no", "n", "0", "0.0"}:
        return False
    return None


def chronicity_bin(row: dict[str, str]) -> str:
    days = parse_float(row.get("days_post_stroke"))
    if days is None or days <= 0:
        return "unknown"
    lt180 = parse_bool(row.get("chronicity_lt180d"))
    if lt180 is None:
        lt180 = days < 180
    return "lt180d" if lt180 else "ge180d"


def size_bin_ml(volume_ml: float | None) -> str:
    if volume_ml is None or volume_ml < 0:
        return "unknown"
    for name, lo, hi in DEFAULT_SIZE_BINS_ML:
        if lo <= volume_ml < hi:
            return name
    return DEFAULT_SIZE_BINS_ML[-1][0]


def load_fold0_val(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    folds = data.get("folds", data) if isinstance(data, dict) else data
    try:
        val_cases = list(folds[0]["val"])
    except (IndexError, KeyError, TypeError) as exc:
        raise SystemExit(f"could not read fold 0 val cases from {path}: {exc}") from exc
    if not val_cases:
        raise SystemExit(f"fold 0 val list is empty in {path}")
    return val_cases


def path_from_manifest(row: dict[str, str], primary: str) -> Path:
    value = row.get(primary)
    if not value:
        raise SystemExit(
            f"manifest row for {row.get('nnunet_id')} is missing {primary}; "
            f"rerun analysis_nnunet_01 so the conversion manifest has source-path columns"
        )
    return Path(value)


def atlas_characterization(
    manifest_row: dict[str, str],
    per_session_index: dict[tuple[str, str, str], dict[str, str]],
) -> tuple[str, str, float | None, str]:
    key = (
        manifest_row.get("site", ""),
        manifest_row.get("subject", ""),
        manifest_row.get("session", ""),
    )
    session_row = per_session_index.get(key)
    if session_row is None:
        print(f"[warn] missing per-session characterization for {key}; falling back to manifest", file=sys.stderr)
        session_row = manifest_row

    site = session_row.get("site") or manifest_row.get("site") or "unknown"
    chronicity = chronicity_bin(session_row)
    lesion_volume_ml = parse_float(session_row.get("lesion_volume_ml"))
    lesion_size_bin = size_bin_ml(lesion_volume_ml)
    return site, chronicity, lesion_volume_ml, lesion_size_bin


def group_by_case_key(cases: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        groups[case.get(key) or "unknown"].append(case)
    return {name: aggregate_minimal(rows) for name, rows in sorted(groups.items())}


def fmt_float(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{value:.{digits}f}"


def append_atlas_sections(markdown_path: Path, report: dict) -> None:
    lines: list[str] = [
        "",
        "## ATLAS stratification notes",
        "",
        "In this report, the `center` field used by the shared evaluator is the ATLAS site.",
        "",
        "## By lesion size bin",
        "",
        "| bin | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, info in report["by_lesion_size_bin"].items():
        lines.append(
            "| {grp} | {n} | {dice_mean} | {dice_median} | {f1} | {recall} | {hd95} | {avd} |".format(
                grp=group,
                n=info["n"],
                dice_mean=fmt_float(info["dice_mean"]),
                dice_median=fmt_float(info["dice_median"]),
                f1=fmt_float(info["lesion_f1_mean"]),
                recall=fmt_float(info["lesion_recall_mean"]),
                hd95=fmt_float(info["hd95_mm_mean"], 3),
                avd=fmt_float(info["abs_volume_diff_ml_mean"], 3),
            )
        )
    markdown_path.write_text(markdown_path.read_text().rstrip() + "\n" + "\n".join(lines) + "\n")


def representative_qc_cases(per_case: list[dict]) -> list[dict]:
    def dice_key(case: dict) -> float:
        value = case["overlap"]["dice"]
        return -1.0 if isinstance(value, float) and math.isnan(value) else float(value)

    ordered = sorted(per_case, key=dice_key)
    mid = len(ordered) // 2
    candidates = ordered[:3] + ordered[max(0, mid - 1):mid + 1] + ordered[-2:]

    selected = []
    seen = set()
    for case in candidates:
        case_id = case["subject_id"]
        if case_id in seen:
            continue
        selected.append(case)
        seen.add(case_id)
    return selected


def write_qc(selected: list[dict], out_dir: Path) -> None:
    qc_dir = out_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    for case in selected:
        gt_path = Path(case["gt_path"])
        t1_path = Path(case["t1_path"])
        pred_path = Path(case["prediction_path"])
        gt_img = nib.load(str(gt_path))
        gt_data, _, _ = load_mask(gt_path)
        pred_data = resample_to_reference(pred_path, gt_img)
        make_overlay(
            t1_path,
            gt_data,
            pred_data,
            qc_dir / f"{case['subject_id']}.png",
            case["subject_id"],
            case["overlap"]["dice"],
            case["lesionwise"]["lesion_f1"],
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--pred-dir", required=True, type=Path)
    parser.add_argument("--per-session-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--min-overlap-voxels", type=int, default=1)
    args = parser.parse_args(argv)

    if args.min_overlap_voxels < 1:
        parser.error("--min-overlap-voxels must be >= 1")

    manifest = {row["nnunet_id"]: row for row in load_csv(args.manifest)}
    per_session_index = {
        (row.get("site", ""), row.get("subject", ""), row.get("session", "")): row
        for row in load_csv(args.per_session_csv)
    }
    val_cases = load_fold0_val(args.splits)

    per_case: list[dict] = []
    for case_id in val_cases:
        if case_id not in manifest:
            raise SystemExit(f"{case_id} is in fold 0 val split but missing from {args.manifest}")
        manifest_row = manifest[case_id]
        gt_path = path_from_manifest(manifest_row, "lesion_source_path")
        t1_path = path_from_manifest(manifest_row, "t1_source_path")
        pred_path = args.pred_dir / f"{case_id}.nii.gz"
        if not pred_path.exists():
            raise SystemExit(f"missing prediction for {case_id}: {pred_path}")
        if not gt_path.exists():
            raise SystemExit(f"missing GT lesion source for {case_id}: {gt_path}")
        if not t1_path.exists():
            raise SystemExit(f"missing T1w source for {case_id}: {t1_path}")

        site, chronicity, lesion_volume_ml, lesion_size_bin = atlas_characterization(
            manifest_row,
            per_session_index,
        )

        gt_img = nib.load(str(gt_path))
        gt_data, _, voxel_size = load_mask(gt_path)
        pred_data = resample_to_reference(pred_path, gt_img)
        metrics = evaluate_case(
            pred_data,
            gt_data,
            voxel_size,
            subject_id=case_id,
            chronicity=chronicity,
            center=site,
            min_overlap_voxels=args.min_overlap_voxels,
        )
        case_dict = metrics.to_dict()
        case_dict["prediction_path"] = str(pred_path)
        case_dict["gt_path"] = str(gt_path)
        case_dict["t1_path"] = str(t1_path)
        case_dict["site"] = site
        case_dict["subject"] = manifest_row.get("subject", "")
        case_dict["session"] = manifest_row.get("session", "")
        case_dict["chronicity_bin"] = chronicity
        case_dict["lesion_volume_ml_characterization"] = lesion_volume_ml
        case_dict["lesion_size_bin"] = lesion_size_bin
        per_case.append(case_dict)
        print(
            f"{case_id}: site={site} chronicity={chronicity} size_bin={lesion_size_bin} "
            f"Dice={metrics.overlap.dice:.4f} F1={metrics.lesionwise.lesion_f1:.4f} "
            f"GTml={metrics.volume.gt_vol_ml:.2f} PredML={metrics.volume.pred_vol_ml:.2f}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "per_case.json").write_text(json.dumps(per_case, indent=2, default=str) + "\n")

    report = aggregate(per_case)
    report["atlas_val_cases"] = val_cases
    report["by_site"] = report["by_center"]
    report["by_chronicity_bin"] = report["by_chronicity"]
    report["by_lesion_size_bin"] = group_by_case_key(per_case, "lesion_size_bin")
    report["n_missing_predictions"] = 0
    write_json(report, args.out_dir / "report.json")
    write_markdown(report, args.out_dir / "report.md", title=args.title)
    append_atlas_sections(args.out_dir / "report.md", report)
    write_qc(representative_qc_cases(per_case), args.out_dir)

    print(f"\nWrote {args.out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
