#!/usr/bin/env python3
"""Convert an OOD blend sweep summary into this repo's report schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    best = summary["best_overall"]
    report = {
        "n_cases": int(summary["n_cases"]),
        "overall": [
            {
                "metric": "dice",
                "n_valid": int(summary["n_cases"]),
                "mean": float(best["mean_dice"]),
                "median": float(best["median_dice"]),
                "std": None,
                "min": None,
                "max": None,
            }
        ],
        "selected_blend": best,
        "source_summary": str(args.summary),
        "selection_note": "Blend weight and threshold were selected on this same soop_bench sweep.",
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = [
        f"# {args.title}",
        "",
        f"Cases: {summary['n_cases']}",
        "",
        "| selected model | w_nn | threshold | mean Dice | median Dice |",
        "|---|---:|---:|---:|---:|",
        f"| {best['label']} | {best['weight_nn']:.1f} | {best['threshold']:.2f} | "
        f"{best['mean_dice']:.4f} | {best['median_dice']:.4f} |",
        "",
        "Caveat: blend weight and threshold were selected on this same soop_bench sweep.",
    ]
    if args.notes:
        lines += ["", args.notes]
    (args.out_dir / "report_adapter.md").write_text("\n".join(lines) + "\n")
    print(f"[done] wrote {args.out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
