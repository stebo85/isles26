#!/usr/bin/env python3
"""Measure complementarity across already-trained lesion-segmentation models.

The script combines per-case native-space lesion probability maps from multiple
models, tunes every candidate over the same threshold/min-CC grid, and reports
whether the all-model mean beats the best single model. It also estimates the
per-case oracle ceiling from perfect routing among the single models.
"""

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

import numpy as np

from baselines.models.synthstroke.synthstroke_helpers.tune_postproc import (
    apply_cc_filter,
    compute_case_metrics,
    get_voxel_size,
    gt_data,
    load_gt_nii,
    load_prob_nii,
    prob_data,
)


class ModelSpec(NamedTuple):
    name: str
    path: Path


class Candidate(NamedTuple):
    name: str
    members: Tuple[str, ...]
    kind: str


class LoadedCase(NamedTuple):
    case_id: str
    probs: Dict[str, np.ndarray]
    gt: np.ndarray
    voxel_size: Tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prob-dirs",
        nargs="+",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="one or more model probability directories; may be repeated",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("work/nnunet/nnUNet_raw/Dataset501_ATLASR21/labelsTr"),
        help="directory with GT masks named <case>.nii.gz",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        help="optional newline-delimited case-id list",
    )
    parser.add_argument(
        "--thresholds",
        default="0.3,0.4,0.5,0.6",
        help="comma-separated probability thresholds to sweep",
    )
    parser.add_argument(
        "--min-cc",
        default="0,10",
        help="comma-separated min-cc-voxel values to sweep",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--pairs",
        action="store_true",
        help="also evaluate every unordered 2-model mean ensemble",
    )
    return parser.parse_args()


def parse_prob_dirs(groups: List[List[str]]) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    seen: Set[str] = set()
    for item in itertools.chain.from_iterable(groups):
        if "=" not in item:
            raise SystemExit(f"[error] --prob-dirs item must be NAME=DIR: {item}")
        name, raw_path = item.split("=", 1)
        name = name.strip()
        if not name or not raw_path:
            raise SystemExit(f"[error] --prob-dirs item must be NAME=DIR: {item}")
        if name in seen:
            raise SystemExit(f"[error] duplicate model name in --prob-dirs: {name}")
        path = Path(raw_path)
        if not path.is_dir():
            raise SystemExit(f"[error] probability directory not found for {name}: {path}")
        specs.append(ModelSpec(name=name, path=path))
        seen.add(name)
    if len(specs) < 2:
        raise SystemExit("[error] at least two model probability directories are required")
    return specs


def parse_float_list(raw: str, label: str) -> List[float]:
    try:
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise SystemExit(f"[error] could not parse {label}: {raw}") from exc
    if not values or any(not math.isfinite(value) for value in values):
        raise SystemExit(f"[error] {label} must contain at least one finite value")
    return values


def parse_int_list(raw: str, label: str) -> List[int]:
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise SystemExit(f"[error] could not parse {label}: {raw}") from exc
    if not values or any(value < 0 for value in values):
        raise SystemExit(f"[error] {label} must contain non-negative integers")
    return values


def case_id_from_nii(path: Path) -> Optional[str]:
    return path.name[:-7] if path.name.endswith(".nii.gz") else None


def scan_case_ids(directory: Path) -> Set[str]:
    if not directory.is_dir():
        return set()
    case_ids: Set[str] = set()
    for path in directory.glob("*.nii.gz"):
        case_id = case_id_from_nii(path)
        if case_id is not None:
            case_ids.add(case_id)
    return case_ids


def read_case_file(path: Path) -> List[str]:
    seen: Set[str] = set()
    case_ids: List[str] = []
    for raw in path.read_text().splitlines():
        case_id = raw.strip()
        if not case_id or case_id.startswith("#"):
            continue
        if case_id.endswith(".nii.gz"):
            case_id = case_id[:-7]
        if case_id not in seen:
            case_ids.append(case_id)
            seen.add(case_id)
    return case_ids


def default_case_ids(models: List[ModelSpec], labels_dir: Path) -> List[str]:
    common = scan_case_ids(labels_dir)
    for model in models:
        common &= scan_case_ids(model.path)
    return sorted(common)


def missing_inputs(case_id: str, models: List[ModelSpec], labels_dir: Path) -> List[str]:
    missing: List[str] = []
    gt_path = labels_dir / f"{case_id}.nii.gz"
    if not gt_path.exists():
        missing.append(f"GT mask not found: {gt_path}")
    for model in models:
        prob_path = model.path / f"{case_id}.nii.gz"
        if not prob_path.exists():
            missing.append(f"prob map not found for {model.name}: {prob_path}")
    return missing


def load_case(
    case_id: str,
    models: List[ModelSpec],
    labels_dir: Path,
) -> Tuple[Optional[LoadedCase], Optional[str]]:
    missing = missing_inputs(case_id, models, labels_dir)
    if missing:
        return None, "; ".join(missing)

    gt_path = labels_dir / f"{case_id}.nii.gz"
    try:
        gt_img = load_gt_nii(gt_path)
        gt = gt_data(gt_img)
    except Exception as exc:  # noqa: BLE001 - report and skip bad cases.
        return None, f"GT mask load/data error: {exc}"

    probs: Dict[str, np.ndarray] = {}
    for model in models:
        prob_path = model.path / f"{case_id}.nii.gz"
        try:
            prob_img = load_prob_nii(prob_path)
            prob = prob_data(prob_img)
        except Exception as exc:  # noqa: BLE001 - report and skip bad cases.
            return None, f"prob map load/data error for {model.name}: {exc}"

        if prob.shape != gt.shape:
            return None, (
                f"shape mismatch for {model.name}: "
                f"prob={prob.shape} gt={gt.shape}"
            )
        if not np.allclose(prob_img.affine, gt_img.affine):
            max_diff = float(np.abs(prob_img.affine - gt_img.affine).max())
            return None, (
                f"affine mismatch for {model.name}: "
                f"max_abs_diff={max_diff:.6g}"
            )
        probs[model.name] = prob

    return LoadedCase(case_id, probs, gt, get_voxel_size(gt_img)), None


def load_cases(
    case_ids: Iterable[str],
    models: List[ModelSpec],
    labels_dir: Path,
) -> Tuple[List[LoadedCase], Dict[str, str]]:
    loaded: List[LoadedCase] = []
    skipped: Dict[str, str] = {}
    for case_id in case_ids:
        loaded_case, reason = load_case(case_id, models, labels_dir)
        if loaded_case is None:
            skipped[case_id] = reason or "unknown validation failure"
            continue
        loaded.append(loaded_case)
    return loaded, skipped


def build_candidates(model_names: List[str], include_pairs: bool) -> List[Candidate]:
    candidates = [
        Candidate(name=name, members=(name,), kind="single")
        for name in model_names
    ]
    candidates.append(
        Candidate(name="ens_mean_all", members=tuple(model_names), kind="mean_all")
    )
    if include_pairs:
        for a_name, b_name in itertools.combinations(model_names, 2):
            candidates.append(
                Candidate(
                    name=f"ens_mean_{a_name}+{b_name}",
                    members=(a_name, b_name),
                    kind="pair_mean",
                )
            )

    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise SystemExit("[error] candidate name collision; choose different model names")
    return candidates


def mean_prob(arrays: List[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(arrays[0], dtype=np.float32)
    for array in arrays:
        out += array.astype(np.float32, copy=False)
    out /= np.float32(len(arrays))
    return out


def candidate_prob(case: LoadedCase, candidate: Candidate) -> np.ndarray:
    if len(candidate.members) == 1:
        return case.probs[candidate.members[0]]
    return mean_prob([case.probs[name] for name in candidate.members])


def mean_finite(values: List[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def better_mean_dice(row: Dict[str, Any], best: Optional[Dict[str, Any]]) -> bool:
    if best is None:
        return True
    row_score = row["mean_dice"]
    best_score = best["mean_dice"]
    if math.isnan(row_score):
        return False
    if math.isnan(best_score):
        return True
    return bool(row_score > best_score)


def evaluate_candidate(
    candidate: Candidate,
    loaded: List[LoadedCase],
    thresholds: List[float],
    cc_values: List[int],
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    probs_by_case = {
        loaded_case.case_id: candidate_prob(loaded_case, candidate)
        for loaded_case in loaded
    }
    best_row: Optional[Dict[str, Any]] = None
    best_per_case: Dict[str, float] = {}

    for thr in thresholds:
        for cc in cc_values:
            dice_values: List[float] = []
            f1_values: List[float] = []
            per_case_dice: Dict[str, float] = {}
            for loaded_case in loaded:
                metrics = compute_case_metrics(
                    probs_by_case[loaded_case.case_id],
                    loaded_case.gt,
                    loaded_case.voxel_size,
                    thr,
                    cc,
                )
                dice = float(metrics["dice"])
                dice_values.append(dice)
                f1_values.append(float(metrics["lesion_f1"]))
                per_case_dice[loaded_case.case_id] = dice

            row = {
                "name": candidate.name,
                "kind": candidate.kind,
                "members": list(candidate.members),
                "mean_dice": mean_finite(dice_values),
                "lesion_f1": mean_finite(f1_values),
                "best_threshold": float(thr),
                "best_min_cc": int(cc),
                "n_cases": len(loaded),
            }
            if better_mean_dice(row, best_row):
                best_row = row
                best_per_case = per_case_dice

    if best_row is None:
        raise RuntimeError(f"no sweep rows computed for candidate {candidate.name}")
    return best_row, best_per_case


def pearsonr_or_nan(x_values: List[float], y_values: List[float]) -> float:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def pairwise_complementarity(
    model_names: List[str],
    per_case: Dict[str, Dict[str, float]],
    case_ids: List[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for a_name, b_name in itertools.combinations(model_names, 2):
        a = [per_case[a_name][case_id] for case_id in case_ids]
        b = [per_case[b_name][case_id] for case_id in case_ids]
        a_arr = np.asarray(a, dtype=np.float64)
        b_arr = np.asarray(b, dtype=np.float64)
        valid = np.isfinite(a_arr) & np.isfinite(b_arr)
        if valid.any():
            frac_a_gt_b = float(np.mean(a_arr[valid] > b_arr[valid]))
            mean_abs_diff = float(np.mean(np.abs(a_arr[valid] - b_arr[valid])))
        else:
            frac_a_gt_b = float("nan")
            mean_abs_diff = float("nan")
        rows.append(
            {
                "model_a": a_name,
                "model_b": b_name,
                "fraction_a_strictly_exceeds_b": frac_a_gt_b,
                "mean_abs_dice_diff": mean_abs_diff,
                "pearson_r": pearsonr_or_nan(a, b),
                "n_cases": int(valid.sum()),
            }
        )
    return rows


def finite_sort_score(row: Dict[str, Any]) -> float:
    value = float(row["mean_dice"])
    return value if math.isfinite(value) else float("-inf")


def fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "nan"
    value = float(value)
    return f"{value:.{digits}f}" if math.isfinite(value) else "nan"


def md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def oracle_summary(
    single_results: List[Dict[str, Any]],
    ens_result: Dict[str, Any],
    per_case: Dict[str, Dict[str, float]],
    case_ids: List[str],
) -> Dict[str, Any]:
    best_single = max(single_results, key=finite_sort_score)
    oracle_values = [
        max(per_case[result["name"]][case_id] for result in single_results)
        for case_id in case_ids
    ]
    oracle_mean = mean_finite([float(value) for value in oracle_values])
    return {
        "oracle_best_per_case_dice": oracle_mean,
        "best_single_name": best_single["name"],
        "best_single_mean_dice": best_single["mean_dice"],
        "ens_mean_all_mean_dice": ens_result["mean_dice"],
        "delta_vs_best_single": oracle_mean - float(best_single["mean_dice"]),
        "delta_vs_ens_mean_all": oracle_mean - float(ens_result["mean_dice"]),
        "ens_mean_all_delta_vs_best_single": (
            float(ens_result["mean_dice"]) - float(best_single["mean_dice"])
        ),
    }


def render_report(
    candidate_rows: List[Dict[str, Any]],
    oracle: Dict[str, Any],
    pairwise_rows: List[Dict[str, Any]],
    n_cases: int,
    skipped_cases: Dict[str, str],
) -> str:
    sorted_candidates = sorted(candidate_rows, key=finite_sort_score, reverse=True)
    lines = [
        "# Ensemble Complementarity",
        "",
        (
            "All candidates are tuned on the same validation cases over the "
            "requested threshold/min-CC grid. Treat these numbers as a relative "
            "comparison, not as an unbiased deployment estimate."
        ),
        (
            "Probabilities are the raw saved maps and are evaluated without "
            "extra brain masking; this is internally consistent across candidates."
        ),
        "",
        f"Usable cases: {n_cases}",
        f"Skipped cases: {len(skipped_cases)}",
        "",
        "| candidate | mean_dice | lesion_f1 | best(thr,cc) |",
        "|---|---:|---:|---|",
    ]
    for row in sorted_candidates:
        lines.append(
            "| {name} | {dice} | {f1} | ({thr},{cc}) |".format(
                name=md_cell(row["name"]),
                dice=fmt_float(row["mean_dice"]),
                f1=fmt_float(row["lesion_f1"]),
                thr=fmt_float(row["best_threshold"], digits=2),
                cc=int(row["best_min_cc"]),
            )
        )

    lines.extend(
        [
            "",
            "## Oracle",
            "",
            (
                "Oracle best-per-case Dice: {oracle_dice} "
                "(oracle - best single {best_name}: {delta_best}; "
                "oracle - ens_mean_all: {delta_ens}; "
                "ens_mean_all - best single: {delta_ens_best})."
            ).format(
                oracle_dice=fmt_float(oracle["oracle_best_per_case_dice"]),
                best_name=md_cell(oracle["best_single_name"]),
                delta_best=fmt_float(oracle["delta_vs_best_single"]),
                delta_ens=fmt_float(oracle["delta_vs_ens_mean_all"]),
                delta_ens_best=fmt_float(oracle["ens_mean_all_delta_vs_best_single"]),
            ),
            "",
            "## Pairwise Complementarity",
            "",
            "| model A | model B | frac A dice > B | mean abs dice diff | Pearson r |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in pairwise_rows:
        lines.append(
            "| {a} | {b} | {frac} | {diff} | {corr} |".format(
                a=md_cell(row["model_a"]),
                b=md_cell(row["model_b"]),
                frac=fmt_float(row["fraction_a_strictly_exceeds_b"]),
                diff=fmt_float(row["mean_abs_dice_diff"]),
                corr=fmt_float(row["pearson_r"]),
            )
        )
    if skipped_cases:
        lines.extend(
            [
                "",
                "Skipped case reasons are recorded in `summary.json`.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    models = parse_prob_dirs(args.prob_dirs)
    thresholds = parse_float_list(args.thresholds, "--thresholds")
    cc_values = parse_int_list(args.min_cc, "--min-cc")
    case_ids = read_case_file(args.cases) if args.cases else default_case_ids(models, args.labels_dir)

    if not case_ids:
        raise SystemExit("[error] no candidate cases found")

    loaded, skipped_cases = load_cases(case_ids, models, args.labels_dir)
    if not loaded:
        raise SystemExit("[error] no cases survived validation")

    model_names = [model.name for model in models]
    candidates = build_candidates(model_names, include_pairs=args.pairs)
    print(
        f"[ensemble_complementarity] evaluating {len(candidates)} candidates "
        f"on {len(loaded)} cases ({len(skipped_cases)} skipped)"
    )

    candidate_rows: List[Dict[str, Any]] = []
    selected_per_case_by_candidate: Dict[str, Dict[str, float]] = {}
    for candidate in candidates:
        row, selected_per_case = evaluate_candidate(
            candidate,
            loaded,
            thresholds,
            cc_values,
        )
        candidate_rows.append(row)
        selected_per_case_by_candidate[candidate.name] = selected_per_case
        print(
            f"  {candidate.name}: dice={row['mean_dice']:.4f} "
            f"f1={row['lesion_f1']:.4f} "
            f"best=({row['best_threshold']:.2f},{row['best_min_cc']})"
        )

    loaded_case_ids = [loaded_case.case_id for loaded_case in loaded]
    per_case_json = {
        case_id: {
            name: selected_per_case_by_candidate[name][case_id]
            for name in [*model_names, "ens_mean_all"]
        }
        for case_id in loaded_case_ids
    }
    single_rows = [row for row in candidate_rows if row["kind"] == "single"]
    ens_row = next(row for row in candidate_rows if row["name"] == "ens_mean_all")
    oracle = oracle_summary(
        single_rows,
        ens_row,
        selected_per_case_by_candidate,
        loaded_case_ids,
    )
    pairwise_rows = pairwise_complementarity(
        model_names,
        selected_per_case_by_candidate,
        loaded_case_ids,
    )

    sorted_candidate_rows = sorted(candidate_rows, key=finite_sort_score, reverse=True)
    summary = {
        "candidates": sorted_candidate_rows,
        "oracle": oracle,
        "pairwise": pairwise_rows,
        "n_cases": len(loaded),
        "skipped_cases": skipped_cases,
        "config": {
            "prob_dirs": [{"name": model.name, "dir": model.path} for model in models],
            "labels_dir": args.labels_dir,
            "cases_file": args.cases,
            "case_source": "file" if args.cases else "intersection_prob_dirs_and_gt",
            "n_requested_cases": len(case_ids),
            "thresholds": thresholds,
            "min_cc": cc_values,
            "pairs": bool(args.pairs),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.md").write_text(
        render_report(
            sorted_candidate_rows,
            oracle,
            pairwise_rows,
            len(loaded),
            skipped_cases,
        )
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(json_ready(summary), indent=2) + "\n"
    )
    (args.out_dir / "per_case.json").write_text(
        json.dumps(json_ready(per_case_json), indent=2) + "\n"
    )
    print(f"[ensemble_complementarity] wrote outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
