#!/usr/bin/env python3
"""Convert an OOD blend sweep summary into this repo's report schema.

The upstream sweep argmaxes a 7x7 (blend weight x threshold) grid on twelve
soop_bench subjects. Writing that single maximum out as the report's only
metric turns a 49-point in-sample search on n=12 into something that reads like
a validated result: reproduced during the audit, the in-sample optimum 0.3006
falls to 0.2827 under honest leave-one-out selection (39% of the headline gain
is selection optimism) and the blend's paired advantage over the single-model
baseline is +0.0140 with SE 0.0179 (t=0.78), carried almost entirely by one
subject out of twelve.

This reporter therefore refuses to emit the in-sample maximum on its own. The
leaderboard metric (`overall` -> `dice`) is the leave-one-out estimate; the
in-sample optimum is kept beside it under an explicit
`dice_in_sample_grid_optimum` label, together with the paired difference
against the single-model baseline, its standard error and the full per-case
difference vector.

Per-case Dice is required for all of that. It is taken, in order, from
--per-case-dice, from `per_case_dice` inside the summary's grid entries, or
recomputed from the saved probability maps with --prob-root (same blending and
Dice as the sweep itself). It must average back to the summary's own per-cell
mean_dice or the run aborts: the honest estimate is computed entirely from that
table, so a table describing a different subject set would quietly replace the
headline with a number for a sweep that was never run.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def grid_entries(summary: dict) -> dict[str, dict]:
    grid = summary.get("grid")
    if not isinstance(grid, dict) or not grid:
        raise SystemExit("summary has no 'grid'; cannot audit an in-sample optimum without the searched grid")
    return grid


def per_case_from_sidecar(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text())
    table = payload.get("grid", payload) if isinstance(payload, dict) else payload
    if not isinstance(table, dict):
        raise SystemExit(f"{path}: expected a mapping of grid key -> {{subject: dice}}")
    return {key: {str(sid): float(d) for sid, d in cases.items()} for key, cases in table.items()}


def per_case_from_summary(grid: dict[str, dict]) -> dict[str, dict[str, float]] | None:
    if not all(isinstance(entry.get("per_case_dice"), dict) for entry in grid.values()):
        return None
    return {key: {str(sid): float(d) for sid, d in entry["per_case_dice"].items()} for key, entry in grid.items()}


def per_case_from_probability_maps(grid: dict[str, dict], prob_root: Path, bench_dir: Path) -> dict[str, dict[str, float]]:
    """Recompute per-case Dice for every grid cell from the saved probability maps.

    Mirrors analysis_nnunet_17: both maps already live in TRACE space, the GT is
    canonicalised the same way the eval framework does, and Dice comes from the
    eval framework so the recomputed grid reproduces the committed one.
    """
    sys.path.insert(0, str(REPO_ROOT))
    import nibabel as nib
    import numpy as np

    from eval.loader import discover_soop_bench
    from eval.metrics import overlap_metrics

    nn_dir = prob_root / "nn_prob_trace"
    sp_dir = prob_root / "sp_prob_trace"
    for directory in (nn_dir, sp_dir):
        if not directory.is_dir():
            raise SystemExit(f"missing probability directory: {directory}")

    def load(path: Path, canonical: bool = False) -> "np.ndarray":
        img = nib.load(str(path))
        if canonical:
            img = nib.as_closest_canonical(img)
        data = np.asanyarray(img.dataobj).astype(np.float32)
        while data.ndim > 3 and data.shape[-1] == 1:
            data = data[..., 0]
        return data

    cases: list[tuple[str, "np.ndarray", "np.ndarray", "np.ndarray"]] = []
    for subject in discover_soop_bench(bench_dir):
        sid = subject.subject_id
        nn_path = nn_dir / f"{sid}.nii.gz"
        sp_path = sp_dir / f"{sid}.nii.gz"
        if not (nn_path.is_file() and sp_path.is_file()):
            continue
        gt = load(subject.lesion_mask, canonical=True) > 0.5
        nn = load(nn_path)
        sp = load(sp_path)
        if nn.shape != gt.shape or sp.shape != gt.shape:
            continue
        cases.append((sid, nn, sp, gt))
    if not cases:
        raise SystemExit(f"no usable subjects found under {bench_dir} / {prob_root}")

    table: dict[str, dict[str, float]] = {}
    for key, entry in grid.items():
        weight = float(entry["weight_nn"])
        threshold = float(entry["threshold"])
        table[key] = {
            sid: float(overlap_metrics((weight * nn + (1.0 - weight) * sp) >= threshold, gt).dice)
            for sid, nn, sp, gt in cases
        }
    return table


def check_reproduces(grid: dict[str, dict], table: dict[str, dict[str, float]]) -> list[str]:
    """Warn if per-case Dice does not average back to the summary's own cells."""
    problems = []
    for key, entry in grid.items():
        cases = table.get(key)
        if not cases:
            problems.append(f"{key}: no per-case Dice")
            continue
        recomputed = statistics.fmean(cases.values())
        if not math.isclose(recomputed, float(entry["mean_dice"]), abs_tol=1e-6):
            problems.append(f"{key}: per-case mean {recomputed:.6f} != summary {float(entry['mean_dice']):.6f}")
    return problems


def leave_one_out(grid: dict[str, dict], table: dict[str, dict[str, float]], subjects: list[str]) -> dict:
    """Select the grid cell on n-1 subjects, score the held-out one, rotate.

    This is what the in-sample argmax would be worth on a thirteenth subject:
    no subject helps choose the configuration it is then scored under.
    """
    held_out: dict[str, float] = {}
    chosen: dict[str, str] = {}
    for held in subjects:
        others = [sid for sid in subjects if sid != held]
        best_key = max(
            sorted(grid),
            key=lambda key: statistics.fmean(table[key][sid] for sid in others),
        )
        chosen[held] = best_key
        held_out[held] = float(table[best_key][held])
    values = [held_out[sid] for sid in subjects]
    return {
        "mean_dice": statistics.fmean(values),
        "median_dice": statistics.median(values),
        "per_case_dice": held_out,
        "selected_cell_per_held_out_case": chosen,
        "distinct_selected_cells": sorted(set(chosen.values())),
        "n_cases": len(values),
        "note": "Leave-one-out honest estimate: the configuration scoring each subject was chosen without it.",
    }


def paired_difference(table: dict[str, dict[str, float]], subjects: list[str], candidate_key: str, baseline_key: str) -> dict:
    """Paired per-case difference candidate - baseline, with its own SE.

    The mean difference alone hides that the blend changes nothing on most
    subjects; the vector and the drop-the-most-influential-case check make that
    visible in the report itself.
    """
    diffs = {sid: float(table[candidate_key][sid] - table[baseline_key][sid]) for sid in subjects}
    values = [diffs[sid] for sid in subjects]
    n = len(values)
    mean_diff = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else math.nan
    se = sd / math.sqrt(n) if n > 1 else math.nan
    t_stat = mean_diff / se if n > 1 and se > 0 else math.nan
    most_influential = max(subjects, key=lambda sid: abs(diffs[sid]))
    remaining = [diffs[sid] for sid in subjects if sid != most_influential]
    return {
        "candidate": candidate_key,
        "baseline": baseline_key,
        "n_cases": n,
        "mean_difference": mean_diff,
        "sd": sd,
        "se": se,
        "t": t_stat,
        "n_better": sum(1 for v in values if v > 0),
        "n_worse": sum(1 for v in values if v < 0),
        "n_unchanged": sum(1 for v in values if v == 0),
        "per_case_difference": diffs,
        "most_influential_case": most_influential,
        "mean_difference_excluding_most_influential": statistics.fmean(remaining) if remaining else math.nan,
        "note": "Paired over the same subjects; positive means the selected cell beats the single-model baseline.",
    }


def json_safe(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--per-case-dice",
        type=Path,
        help="JSON mapping grid key -> {subject_id: dice}; needed for the honest statistics.",
    )
    parser.add_argument(
        "--prob-root",
        type=Path,
        help="Directory holding nn_prob_trace/ and sp_prob_trace/; per-case Dice is recomputed from these if no table is supplied.",
    )
    parser.add_argument("--bench-dir", type=Path, default=REPO_ROOT / "data/soop_bench")
    parser.add_argument(
        "--baseline-key",
        help="Grid key used as the single-model baseline; defaults to the best weight_nn=1.0 cell.",
    )
    parser.add_argument(
        "--allow-missing-per-case",
        action="store_true",
        help="Emit a report without the honest statistics; it is then labelled UNVALIDATED throughout.",
    )
    parser.add_argument(
        "--allow-per-case-mismatch",
        action="store_true",
        help="Downgrade 'per-case Dice does not average back to the summary' from a hard error to a warning.",
    )
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    grid = grid_entries(summary)
    best = summary["best_overall"]
    best_key = next(
        (
            key
            for key, entry in grid.items()
            if math.isclose(float(entry["weight_nn"]), float(best["weight_nn"]), abs_tol=1e-9)
            and math.isclose(float(entry["threshold"]), float(best["threshold"]), abs_tol=1e-9)
        ),
        None,
    )
    if best_key is None:
        raise SystemExit("best_overall does not correspond to any grid cell; refusing to report an untraceable optimum")

    baseline_key = args.baseline_key
    if baseline_key is None:
        single_model = [key for key, entry in grid.items() if float(entry["weight_nn"]) == 1.0]
        if not single_model:
            raise SystemExit("no weight_nn=1.0 cell to use as the single-model baseline; pass --baseline-key")
        baseline_key = max(single_model, key=lambda key: float(grid[key]["mean_dice"]))
    if baseline_key not in grid:
        raise SystemExit(f"--baseline-key {baseline_key} is not a grid cell")

    table: dict[str, dict[str, float]] | None = None
    per_case_source = "unavailable"
    if args.per_case_dice is not None:
        table = per_case_from_sidecar(args.per_case_dice)
        per_case_source = str(args.per_case_dice)
    else:
        table = per_case_from_summary(grid)
        if table is not None:
            per_case_source = f"{args.summary} (grid[*].per_case_dice)"
        elif args.prob_root is not None:
            table = per_case_from_probability_maps(grid, args.prob_root, args.bench_dir)
            per_case_source = f"recomputed from {args.prob_root}"

    if table is None and not args.allow_missing_per_case:
        raise SystemExit(
            "per-case Dice is required to report anything but the in-sample grid maximum. Supply "
            "--per-case-dice, add per_case_dice to the summary's grid entries, or point --prob-root at the "
            "saved nn_prob_trace/sp_prob_trace maps. Use --allow-missing-per-case only if the report may say "
            "UNVALIDATED."
        )

    n_cases = int(summary["n_cases"])
    warnings: list[str] = []
    loo: dict | None = None
    paired: dict | None = None
    if table is not None:
        subjects = sorted(table[best_key])
        missing = [key for key in grid if set(table.get(key, {})) != set(subjects)]
        if missing:
            raise SystemExit(f"per-case Dice does not cover every grid cell for the same subjects; first={missing[:3]}")
        problems = check_reproduces(grid, table)
        # Every honest number below is computed from `table`, so a table that
        # does not average back to the summary's own cells describes a different
        # sweep and would silently produce a wrong headline. The usual cause is a
        # table covering fewer subjects than the summary (a probability map went
        # missing, or a sidecar was built from a partial run).
        if problems and not args.allow_per_case_mismatch:
            raise SystemExit(
                f"per-case Dice does not reproduce the summary's own mean_dice for {len(problems)} of "
                f"{len(grid)} grid cells, so the leave-one-out estimate would not describe this sweep "
                f"(table covers {len(subjects)} subjects, summary claims {n_cases}; first={problems[:3]}). "
                "Fix the table, or pass --allow-per-case-mismatch if the discrepancy is understood."
            )
        warnings.extend(problems)
        if len(subjects) != n_cases:
            warnings.append(
                f"per-case Dice covers {len(subjects)} subjects but the summary reports {n_cases} cases; "
                "the honest estimate is over the smaller set"
            )
        loo = leave_one_out(grid, table, subjects)
        paired = paired_difference(table, subjects, best_key, baseline_key)
    else:
        warnings.append("no per-case Dice: the leave-one-out estimate and paired test could not be computed")

    in_sample = {
        "grid_key": best_key,
        "n_grid_points_searched": len(grid),
        "n_cases": n_cases,
        **{key: best[key] for key in best},
        "note": (
            f"IN-SAMPLE MAXIMUM of a {len(grid)}-point grid search over the same {n_cases} cases. "
            "Not a validated result; see loo_honest_estimate."
        ),
    }

    overall = [
        {
            "metric": "dice",
            # The count the honest mean was actually taken over, not the count
            # the summary claims: compare_models.py turns this into the
            # leaderboard's "n=" cell, so the two must not drift apart.
            "n_valid": int(loo["n_cases"]) if loo is not None else None,
            "mean": (loo or {}).get("mean_dice"),
            "median": (loo or {}).get("median_dice"),
            "std": None,
            "min": None,
            "max": None,
            "note": (
                "Leave-one-out honest estimate of the grid-search procedure (blend weight and threshold "
                "re-selected without the scored case)."
                if loo is not None
                else "UNVALIDATED: per-case Dice unavailable, no honest estimate could be computed."
            ),
        },
        {
            "metric": "dice_in_sample_grid_optimum",
            "n_valid": n_cases,
            "mean": float(best["mean_dice"]),
            "median": float(best["median_dice"]),
            "std": None,
            "min": None,
            "max": None,
            "note": in_sample["note"],
        },
    ]

    report = {
        "n_cases": n_cases,
        "overall": overall,
        "headline_metric": "dice = leave-one-out honest estimate" if loo else "UNVALIDATED in-sample maximum",
        "selected_blend": best,
        "in_sample_optimum": in_sample,
        "loo_honest_estimate": loo,
        "paired_vs_single_model_baseline": paired,
        "per_case_dice_source": per_case_source,
        "source_summary": str(args.summary),
        "warnings": warnings,
        "selection_note": (
            f"Blend weight and threshold were argmaxed over {len(grid)} grid points on these same {n_cases} cases. "
            "The leaderboard metric is the leave-one-out estimate; the in-sample maximum is reported separately "
            "and must not be quoted as performance."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")

    lines = [
        f"# {args.title}",
        "",
        f"Cases: {n_cases}. Blend weight and threshold were argmaxed over {len(grid)} grid points on these same "
        f"{n_cases} cases, so the grid maximum is a selection statistic, not a measurement.",
        "",
        "| estimate | w_nn | threshold | mean Dice | median Dice |",
        "|---|---:|---:|---:|---:|",
    ]
    if loo is not None:
        lines.append(
            f"| **leave-one-out honest (report this)** | varies | varies | {fmt(loo['mean_dice'])} | "
            f"{fmt(loo['median_dice'])} |"
        )
    else:
        lines.append("| leave-one-out honest (report this) | — | — | n/a | n/a |")
    lines.append(
        f"| in-sample grid maximum (NOT a result) | {float(best['weight_nn']):.1f} | {float(best['threshold']):.2f} | "
        f"{fmt(float(best['mean_dice']))} | {fmt(float(best['median_dice']))} |"
    )
    lines.append(
        f"| single-model baseline ({baseline_key}) | {float(grid[baseline_key]['weight_nn']):.1f} | "
        f"{float(grid[baseline_key]['threshold']):.2f} | {fmt(float(grid[baseline_key]['mean_dice']))} | "
        f"{fmt(float(grid[baseline_key]['median_dice']))} |"
    )
    lines.append("")

    if loo is not None:
        optimism = float(best["mean_dice"]) - float(loo["mean_dice"])
        lines += [
            f"Selection optimism (in-sample maximum minus honest estimate): **{optimism:+.4f}**. "
            f"Cells chosen under leave-one-out: {', '.join(loo['distinct_selected_cells'])}.",
            "",
        ]
    if paired is not None:
        lines += [
            f"## Paired difference vs the single-model baseline ({baseline_key})",
            "",
            f"- mean difference: **{paired['mean_difference']:+.4f}** (SD {fmt(paired['sd'])}, "
            f"SE {fmt(paired['se'])}, t = {fmt(paired['t'], 2)}, n = {paired['n_cases']})",
            f"- subjects better / worse / unchanged: {paired['n_better']} / {paired['n_worse']} / "
            f"{paired['n_unchanged']}",
            f"- dropping the most influential subject ({paired['most_influential_case']}, "
            f"{paired['per_case_difference'][paired['most_influential_case']]:+.4f}) leaves "
            f"{fmt(paired['mean_difference_excluding_most_influential'])}",
            "",
            "| subject | baseline Dice | selected Dice | difference |",
            "|---|---:|---:|---:|",
        ]
        for sid in sorted(paired["per_case_difference"]):
            lines.append(
                f"| {sid} | {fmt(table[baseline_key][sid])} | {fmt(table[best_key][sid])} | "
                f"{paired['per_case_difference'][sid]:+.4f} |"
            )
        lines.append("")
    else:
        lines += [
            "**UNVALIDATED**: no per-case Dice was available, so neither the honest leave-one-out estimate nor the "
            "paired test against the single-model baseline could be computed. The only number here is the maximum "
            "of an in-sample grid search.",
            "",
        ]

    if warnings:
        lines += ["Warnings:", ""] + [f"- {w}" for w in warnings] + [""]
    lines.append(
        "Caveat: blend weight and threshold were selected on this same sweep; the in-sample maximum above is a "
        "selection statistic and must not be quoted as model performance."
    )
    if args.notes:
        lines += ["", args.notes]
    (args.out_dir / "report_adapter.md").write_text("\n".join(lines) + "\n")
    if table is not None:
        (args.out_dir / "per_case_dice.json").write_text(json.dumps(json_safe(table), indent=2, allow_nan=False) + "\n")
    print(f"[done] wrote {args.out_dir / 'report.json'}")
    if loo is not None:
        print(
            f"[honest] LOO mean Dice {loo['mean_dice']:.4f} vs in-sample maximum {float(best['mean_dice']):.4f} "
            f"(optimism {float(best['mean_dice']) - float(loo['mean_dice']):+.4f})"
        )
    if paired is not None:
        print(
            f"[paired] vs {baseline_key}: {paired['mean_difference']:+.4f} SE {fmt(paired['se'])} "
            f"t {fmt(paired['t'], 2)} ({paired['n_better']} better / {paired['n_worse']} worse / "
            f"{paired['n_unchanged']} unchanged)"
        )
    for warning in warnings:
        print(f"[warn] {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
