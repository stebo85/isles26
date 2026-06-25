#!/usr/bin/env python3
"""Report out-of-fold nnU-Net performance by center and chronicity."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        return {row["nnunet_id"]: row for row in csv.DictReader(f)}


def case_id_from_path(text: str) -> str:
    match = re.search(r"(ATLAS_\d{4})", text)
    return match.group(1) if match else Path(text).name.replace(".nii.gz", "")


def finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def summarize(values: list[float]) -> dict[str, object]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
    }


def load_case_rows(summary: Path, manifest: dict[str, dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    payload = json.loads(summary.read_text())
    rows = []
    invalid_rows = []
    for item in payload.get("metric_per_case", []):
        lesion = item.get("metrics", {}).get("1", {})
        case_id = case_id_from_path(str(item.get("reference_file", item.get("prediction_file", ""))))
        meta = manifest.get(case_id)
        dice = finite(lesion.get("Dice"))
        if meta is None:
            continue
        if dice is None:
            invalid_rows.append({
                "case_id": case_id,
                "site": meta.get("site") or "unknown",
                "chronicity_bin": meta.get("chronicity_bin") or "unknown",
                "size_bin": meta.get("size_bin") or "unknown",
                "reason": "non_finite_dice",
                "n_pred": lesion.get("n_pred"),
                "n_ref": lesion.get("n_ref"),
            })
            continue
        tp = float(lesion.get("TP", 0.0))
        fp = float(lesion.get("FP", 0.0))
        fn = float(lesion.get("FN", 0.0))
        rows.append({
            "case_id": case_id,
            "site": meta.get("site") or "unknown",
            "chronicity_bin": meta.get("chronicity_bin") or "unknown",
            "size_bin": meta.get("size_bin") or "unknown",
            "lesion_volume_ml": meta.get("lesion_volume_ml") or "",
            "dice": dice,
            "iou": finite(lesion.get("IoU")),
            "sensitivity": None if tp + fn <= 0 else tp / (tp + fn),
            "precision": None if tp + fp <= 0 else tp / (tp + fp),
        })
    return rows, invalid_rows


def group_rows(rows: list[dict[str, object]], field: str, min_n: int = 1) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    out = []
    for group, items in grouped.items():
        if len(items) < min_n:
            continue
        out.append({
            "group": group,
            "n": len(items),
            "dice": summarize([float(r["dice"]) for r in items]),
            "sensitivity": summarize([float(r["sensitivity"]) for r in items if r["sensitivity"] is not None]),
            "precision": summarize([float(r["precision"]) for r in items if r["precision"] is not None]),
        })
    out.sort(key=lambda r: (-int(r["n"]), str(r["group"])))
    return out


def format_float(value: object) -> str:
    number = finite(value)
    return "n/a" if number is None else f"{number:.4f}"


def add_group_table(lines: list[str], title: str, rows: list[dict[str, object]]) -> None:
    lines += [
        "",
        f"## {title}",
        "",
        "| group | n | Dice mean | Dice median | sensitivity mean | precision mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['n']} | {format_float(row['dice']['mean'])} | "
            f"{format_float(row['dice']['median'])} | {format_float(row['sensitivity']['mean'])} | "
            f"{format_float(row['precision']['mean'])} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--min-center-cases", default=5, type=int)
    args = parser.parse_args()

    manifest = read_manifest(args.manifest)
    case_rows, invalid_rows = load_case_rows(args.summary, manifest)
    if not case_rows:
        raise SystemExit("no joined case rows")

    center = group_rows(case_rows, "site", min_n=args.min_center_cases)
    chronicity = group_rows(case_rows, "chronicity_bin")
    size = group_rows(case_rows, "size_bin")
    overall = summarize([float(row["dice"]) for row in case_rows])

    report = {
        "n_cases": len(case_rows) + len(invalid_rows),
        "n_valid_dice": len(case_rows),
        "invalid_cases": invalid_rows,
        "overall_dice": overall,
        "by_center": center,
        "by_chronicity": chronicity,
        "by_size_bin": size,
        "case_rows": case_rows,
        "source_summary": str(args.summary),
        "caveat": "This is out-of-fold 5-fold CV grouped by hidden-generalization axes, not leave-center-out retraining.",
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    lines = [
        "# TopK10 hidden-generalization triage",
        "",
        f"Cases: {len(case_rows) + len(invalid_rows)} total; {len(case_rows)} with finite Dice. "
        f"Overall out-of-fold Dice mean {format_float(overall['mean'])}.",
        "",
        "Caveat: these are TopK10 5-fold out-of-fold predictions grouped by center/chronicity.",
        "Use the generated leave-group-out splits for true center-held-out and chronicity-held-out retraining.",
    ]
    if invalid_rows:
        ids = ", ".join(row["case_id"] for row in invalid_rows)
        lines += ["", f"Non-finite Dice rows excluded from aggregates: {ids}."]
    add_group_table(lines, "Center-Held-Out Candidates", center)
    add_group_table(lines, "Chronicity-Held-Out Candidates", chronicity)
    add_group_table(lines, "Lesion-Size Strata", size)
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"[done] wrote {args.out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
