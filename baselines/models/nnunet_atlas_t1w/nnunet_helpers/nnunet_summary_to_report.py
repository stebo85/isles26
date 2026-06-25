#!/usr/bin/env python3
"""Convert an nnU-Net cross-validation summary into this repo's report schema."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path


def finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def summarize(metric: str, values: list[float]) -> dict[str, object]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return {"metric": metric, "n_valid": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "metric": metric,
        "n_valid": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def case_id_from_path(text: str) -> str:
    match = re.search(r"(ATLAS_\d{4})", text)
    return match.group(1) if match else Path(text).name.replace(".nii.gz", "")


def ratio(num: float, den: float) -> float | None:
    return None if den <= 0 else num / den


def load_cases(summary: dict[str, object]) -> list[dict[str, object]]:
    cases = []
    for row in summary.get("metric_per_case", []):
        if not isinstance(row, dict):
            continue
        metrics = row.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        lesion = metrics.get("1", {})
        if not isinstance(lesion, dict):
            continue
        tp = float(lesion.get("TP", 0.0))
        fp = float(lesion.get("FP", 0.0))
        fn = float(lesion.get("FN", 0.0))
        tn = float(lesion.get("TN", 0.0))
        pred = row.get("prediction_file", "")
        case = {
            "case_id": case_id_from_path(str(pred or row.get("reference_file", ""))),
            "dice": finite(lesion.get("Dice")),
            "iou": finite(lesion.get("IoU")),
            "sensitivity": ratio(tp, tp + fn),
            "precision": ratio(tp, tp + fp),
            "specificity": ratio(tn, tn + fp),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "prediction_file": pred,
            "reference_file": row.get("reference_file", ""),
        }
        cases.append(case)
    return cases


def metric_values(cases: list[dict[str, object]], metric: str) -> list[float]:
    vals = []
    for case in cases:
        value = finite(case.get(metric))
        if value is not None:
            vals.append(value)
    return vals


def markdown(title: str, report: dict[str, object], notes: str) -> str:
    lines = [
        f"# {title}",
        "",
        f"Cases: {report['n_cases']}",
        "",
        "| metric | n_valid | mean | median | std | min | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["overall"]:
        def fmt(value: object) -> str:
            number = finite(value)
            return "n/a" if number is None else f"{number:.4f}"
        lines.append(
            f"| {row['metric']} | {row['n_valid']} | {fmt(row['mean'])} | "
            f"{fmt(row['median'])} | {fmt(row['std'])} | {fmt(row['min'])} | {fmt(row['max'])} |"
        )
    if notes:
        lines += ["", "## Notes", "", notes]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    cases = load_cases(summary)
    if not cases:
        raise SystemExit(f"no metric_per_case rows found in {args.summary}")

    report = {
        "n_cases": len(cases),
        "overall": [
            summarize("dice", metric_values(cases, "dice")),
            summarize("iou", metric_values(cases, "iou")),
            summarize("sensitivity", metric_values(cases, "sensitivity")),
            summarize("precision", metric_values(cases, "precision")),
            summarize("specificity", metric_values(cases, "specificity")),
        ],
        "source_summary": str(args.summary),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.out_dir / "per_case.json").write_text(json.dumps(cases, indent=2, allow_nan=False) + "\n")
    (args.out_dir / "report.md").write_text(markdown(args.title, report, args.notes))
    print(f"[done] wrote {args.out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
