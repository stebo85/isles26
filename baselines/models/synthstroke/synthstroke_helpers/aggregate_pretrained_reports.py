#!/usr/bin/env python3
"""Aggregate pretrained SynthStroke benchmark reports into one Markdown table."""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


VARIANTS = [
    "synthstroke-baseline",
    "synthstroke-synth",
    "synthstroke-synth-pseudo",
    "synthstroke-synth-plus",
    "synthstroke-qatlas",
    "synthstroke-qsynth",
]

EVAL_SETS = [
    "atlas_val",
    "soop_trace",
    "soop_t1w",
]

REFERENCE_ROWS = [
    {
        "model": "nnU-Net baseline",
        "eval_set": "atlas_val",
        "n_cases": None,
        "dice_mean": 0.640,
        "dice_median": None,
        "lesion_f1_mean": 0.654,
        "lesion_f1_median": None,
        "source": "given reference",
    },
    {
        "model": "nnU-Net baseline",
        "eval_set": "soop_t1w",
        "n_cases": None,
        "dice_mean": 0.188,
        "dice_median": None,
        "lesion_f1_mean": 0.201,
        "lesion_f1_median": None,
        "source": "given reference",
    },
    {
        "model": "DeepISLES reference",
        "eval_set": "soop_trace",
        "n_cases": None,
        "dice_mean": 0.540,
        "dice_median": None,
        "lesion_f1_mean": 0.582,
        "lesion_f1_median": None,
        "source": "given reference",
    },
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_reports_root = root / "baselines" / "reports" / "synthstroke"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=default_reports_root,
        help="Root containing <variant>/<eval_set>/report.json directories.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_reports_root / "pretrained_report.md",
        help="Markdown comparison report to write.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        print(f"[warn] could not parse {path}: {exc}")
        return None


def overall_stats(report: Dict[str, Any], metric: str) -> Tuple[Optional[float], Optional[float]]:
    for row in report.get("overall", []):
        if row.get("metric") == metric:
            return as_float(row.get("mean")), as_float(row.get("median"))
    return None, None


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def fmt_metric(value: Any) -> str:
    out = as_float(value)
    return "n/a" if out is None else f"{out:.3f}"


def fmt_count(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "n/a"


def markdown_link(target: Path, base_dir: Path) -> str:
    rel = os.path.relpath(target, start=base_dir).replace(os.sep, "/")
    return f"[report]({rel})"


def row_from_report(
    reports_root: Path,
    out_dir: Path,
    variant: str,
    eval_set: str,
) -> Dict[str, Any]:
    report_path = reports_root / variant / eval_set / "report.json"
    report = read_json(report_path)
    if report is None:
        return {
            "model": variant,
            "eval_set": eval_set,
            "n_cases": None,
            "dice_mean": None,
            "dice_median": None,
            "lesion_f1_mean": None,
            "lesion_f1_median": None,
            "source": "missing",
        }

    dice_mean, dice_median = overall_stats(report, "dice")
    f1_mean, f1_median = overall_stats(report, "lesion_f1")
    report_md = report_path.with_name("report.md")
    source = markdown_link(report_md, out_dir) if report_md.exists() else markdown_link(report_path, out_dir)
    return {
        "model": variant,
        "eval_set": eval_set,
        "n_cases": report.get("n_cases"),
        "dice_mean": dice_mean,
        "dice_median": dice_median,
        "lesion_f1_mean": f1_mean,
        "lesion_f1_median": f1_median,
        "source": source,
    }


def render_table(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# SynthStroke Pretrained Benchmark",
        "",
        "Missing report.json files are shown as n/a.",
        "Reference rows use the fixed headline values specified for this benchmark.",
        "",
        "| model | eval set | n | Dice mean | Dice median | lesion-F1 mean | lesion-F1 median | source |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {eval_set} | {n_cases} | {dice_mean} | {dice_median} | "
            "{lesion_f1_mean} | {lesion_f1_median} | {source} |".format(
                model=row["model"],
                eval_set=row["eval_set"],
                n_cases=fmt_count(row["n_cases"]),
                dice_mean=fmt_metric(row["dice_mean"]),
                dice_median=fmt_metric(row["dice_median"]),
                lesion_f1_mean=fmt_metric(row["lesion_f1_mean"]),
                lesion_f1_median=fmt_metric(row["lesion_f1_median"]),
                source=row["source"],
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- nnU-Net soop is listed under soop_t1w because that baseline uses T1w inputs warped into TRACE space.",
            "- DeepISLES soop is listed under soop_trace because its predictions are evaluated in native soop_bench/TRACE space.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    reports_root = args.reports_root.resolve()
    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(REFERENCE_ROWS)
    for variant in VARIANTS:
        for eval_set in EVAL_SETS:
            rows.append(row_from_report(reports_root, out_path.parent, variant, eval_set))

    out_path.write_text(render_table(rows))
    print(f"[done] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
