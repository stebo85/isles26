#!/usr/bin/env python3
"""Score predictions with the ISLES'26 OFFICIAL metrics, and sweep the two
knobs we can still turn without retraining: the probability threshold and
min-connected-component pruning.

Two modes:

  score   one configuration (or a sweep grid) over a shard of cases; writes
          `per_case_shard_<i>.json`. Designed to run as a SLURM array.
  reduce  merge the shards, aggregate by configuration (means AND
          rank-then-aggregate), and write report.json / report.md / per_case.csv.

Why a sweep and not a single score: `docs/models/status.md` recorded
"connected-component postprocessing: none chosen" and threshold 0.54, but both
decisions were made against Dice and against this repo's LENIENT 1-voxel
lesion-F1 respectively. Two of the five official metrics (absolute lesion count
difference, and lesion-F1 under one-to-one IoU>=0.25 matching) are directly
sensitive to spurious small components, so both decisions have to be re-taken
against the official scorer.

Example (in-distribution surface, one shard of an array):

    python eval/run_official_eval.py score \
        --gt-dir work/nnunet/nnUNet_raw/Dataset502_ATLASR30/labelsTr \
        --prob-dir 'work/nnunet_mapping3_oof_ensemble/folds/fold_*/probabilities' \
        --out-dir work/official_eval/d502_topk10 \
        --thresholds 0.20:0.60:0.05 --min-cc 0,5,10,20,35,50 \
        --shard-index $SLURM_ARRAY_TASK_ID --shard-count 30
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.official_metrics import (  # noqa: E402
    OFFICIAL_SOURCE,
    RANKED_METRICS,
    aggregate_means,
    flat_soft_pr_auc,
    min_size_filter,
    rank_aggregate,
    score_masks,
    score_soft,
)


# Structuring elements matching the official scorer's component definition.
# `eval/probe_panoptica_semantics.py` measures that panoptica's
# ConnectedComponentsInstanceApproximator treats a corner-adjacent voxel pair as
# ONE instance, i.e. 26-connectivity. Pruning with a different connectivity than
# the scorer counts with would make the filter and the metric disagree.
_CONNECTIVITY = {
    6: ndi.generate_binary_structure(3, 1),
    18: ndi.generate_binary_structure(3, 2),
    26: ndi.generate_binary_structure(3, 3),
}


# --- config grid ------------------------------------------------------------


def parse_float_list(spec: str) -> list[float]:
    """Accept either `a,b,c` or `start:stop:step` (stop inclusive)."""
    spec = spec.strip()
    if ":" in spec:
        start, stop, step = (float(x) for x in spec.split(":"))
        n = int(round((stop - start) / step)) + 1
        return [round(start + i * step, 6) for i in range(n)]
    return [float(x) for x in spec.split(",") if x.strip()]


def parse_int_list(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def config_name(threshold: float, min_cc: int) -> str:
    return f"thr{threshold:.2f}_k{min_cc}"


# --- IO ---------------------------------------------------------------------


def discover(prob_patterns: list[str]) -> dict[str, Path]:
    """Map subject id -> probability NIfTI. Later patterns must not collide with
    earlier ones: out-of-fold volumes are partitioned by construction, and a
    duplicate means two folds both predicted the same case, which would silently
    pick one at random."""
    found: dict[str, Path] = {}
    for pattern in prob_patterns:
        for d in sorted(glob.glob(pattern)):
            for p in sorted(Path(d).glob("*.nii.gz")):
                sid = p.name[: -len(".nii.gz")]
                if sid in found:
                    raise ValueError(
                        f"duplicate probability volume for {sid}: {found[sid]} and {p}. "
                        "Out-of-fold predictions must partition the cases."
                    )
                found[sid] = p
    return found


def load_binarised_gt(path: Path) -> tuple[np.ndarray, float]:
    """Load a ground-truth mask and its voxel volume in mL.

    Re-binarising is mandatory, not defensive: ATLAS site R039 stores masks as
    float64 0.9999... (documented in docs/data/atlas-r21-characterization.md).
    """
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    mask = (np.asarray(data) > 0.5).astype(np.uint8)
    zooms = img.header.get_zooms()[:3]
    voxel_ml = float(np.prod([float(z) for z in zooms])) / 1000.0
    return mask, voxel_ml


# --- per-case work ----------------------------------------------------------


def score_one_case(
    subject_id: str,
    gt_path: str,
    prob_path: str,
    thresholds: list[float],
    min_ccs: list[int],
    connectivity: int,
    reference_threshold: float,
) -> list[dict]:
    gt, voxel_ml = load_binarised_gt(Path(gt_path))
    prob_img = nib.load(prob_path)
    prob = np.asarray(prob_img.dataobj, dtype=np.float32)

    if prob.shape != gt.shape:
        raise ValueError(
            f"{subject_id}: probability shape {prob.shape} != ground truth {gt.shape}. "
            "The official compute_pr_auc raises on mismatch; do not resample silently."
        )

    # Threshold-independent PR-AUC terms, computed once.
    pr_auc_raw = score_soft(gt, prob)
    pr_auc_flat = flat_soft_pr_auc(gt)  # what emitting a constant map would score

    # Acceptance check for the harness itself: PR-AUC when the *binary mask* is
    # submitted as the soft map. Computed only at the reference configuration.
    ref_pred = (prob >= reference_threshold).astype(np.uint8)
    pr_auc_binary_as_soft = score_soft(gt, ref_pred.astype(np.float32))

    del ref_pred

    gt_empty = not bool(np.any(gt))
    rows: list[dict] = []
    for thr in thresholds:
        # Label ONCE per threshold and derive every min-component-size mask from
        # the same labelling. Re-labelling per K was both wasteful and the main
        # driver of the peak memory that OOM-killed the first full array: the
        # official scorer already casts both masks to int64 internally, so a
        # 0.5 mm case at ~80 M voxels costs ~1.3 GB inside panoptica alone.
        pred_thr = prob >= thr
        lab, n_comp = ndi.label(pred_thr, structure=_CONNECTIVITY[connectivity])
        counts = np.bincount(lab.ravel(), minlength=n_comp + 1) if n_comp else np.zeros(1, np.int64)

        for k in min_ccs:
            if k <= 0 or n_comp == 0:
                pred = pred_thr.astype(np.uint8)
            else:
                keep = counts >= k
                keep[0] = False
                pred = keep[lab].astype(np.uint8)
            dice, avd, count_diff, f1 = score_masks(gt, pred, voxel_ml)
            pred_empty = not bool(np.any(pred))
            del pred
            rows.append(
                {
                    "subject_id": subject_id,
                    "config": config_name(thr, k),
                    "threshold": thr,
                    "min_component_voxels": k,
                    "dice": dice,
                    "abs_volume_diff_ml": avd,
                    "abs_lesion_count_diff": count_diff,
                    "lesion_f1": f1,
                    # Ranked PR-AUC as we would ship it today: the raw softmax.
                    "pr_auc": pr_auc_raw,
                    # ... and with the "emit a constant map when we predict
                    # nothing" rule, which is the only intervention that can move
                    # PR-AUC (it is invariant to monotone rescaling).
                    "pr_auc_flatten_if_empty": pr_auc_flat if pred_empty else pr_auc_raw,
                    "pr_auc_binary_as_soft": pr_auc_binary_as_soft,
                    "gt_empty": gt_empty,
                    "pred_empty": pred_empty,
                    "voxel_volume_ml": voxel_ml,
                }
            )
        del pred_thr, lab, counts
    del prob, gt
    return rows


def _worker(args: tuple) -> tuple[str, list[dict] | None, str | None]:
    subject_id = args[0]
    try:
        return subject_id, score_one_case(*args), None
    except Exception:  # noqa: BLE001 - one bad case must not kill the shard
        return subject_id, None, traceback.format_exc()


# --- commands ---------------------------------------------------------------


def cmd_score(args: argparse.Namespace) -> int:
    gt_dir = Path(args.gt_dir)
    probs = discover(args.prob_dir)
    subjects = sorted(probs)
    if args.cases_file:
        wanted = {
            line.strip() for line in Path(args.cases_file).read_text().splitlines() if line.strip()
        }
        subjects = [s for s in subjects if s in wanted]

    missing_gt = [s for s in subjects if not (gt_dir / f"{s}.nii.gz").is_file()]
    if missing_gt:
        raise SystemExit(
            f"[error] {len(missing_gt)} subjects have a prediction but no ground truth "
            f"(first few: {missing_gt[:5]})"
        )

    shard = subjects[args.shard_index :: args.shard_count]
    thresholds = parse_float_list(args.thresholds)
    min_ccs = parse_int_list(args.min_cc)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"per_case_shard_{args.shard_index:04d}.json"
    if out_path.is_file() and not args.overwrite:
        # Resuming is only safe if the existing shard used the SAME partition.
        # A leftover shard from a run with a different --shard-count covers a
        # different set of cases, so silently keeping it both drops cases and
        # duplicates others. This actually happened once, after an OOM-killed
        # 30-way array left three shards behind for a 60-way rerun.
        existing = json.loads(out_path.read_text())
        if existing.get("shard_count") != args.shard_count:
            raise SystemExit(
                f"[error] {out_path} was written with shard_count="
                f"{existing.get('shard_count')} but this run uses {args.shard_count}. "
                "Delete the stale shards before resuming."
            )
        print(f"[skip] {out_path} exists (use --overwrite to recompute)")
        return 0

    print(
        f"[info] shard {args.shard_index}/{args.shard_count}: {len(shard)} of {len(subjects)} "
        f"cases x {len(thresholds)} thresholds x {len(min_ccs)} min-CC = "
        f"{len(shard) * len(thresholds) * len(min_ccs)} scorings",
        flush=True,
    )

    tasks = [
        (
            s,
            str(gt_dir / f"{s}.nii.gz"),
            str(probs[s]),
            thresholds,
            min_ccs,
            args.connectivity,
            args.reference_threshold,
        )
        for s in shard
    ]

    rows: list[dict] = []
    errors: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_worker, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            sid, res, err = fut.result()
            if err is not None:
                errors[sid] = err
                print(f"[error] {sid}\n{err}", file=sys.stderr, flush=True)
            else:
                rows.extend(res)
            if i % 10 == 0:
                print(f"[progress] {i}/{len(tasks)}", flush=True)

    payload = {
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "official_source": OFFICIAL_SOURCE,
        "connectivity": args.connectivity,
        "thresholds": thresholds,
        "min_component_voxels": min_ccs,
        "n_cases": len(shard),
        "n_failed": len(errors),
        "errors": errors,
        "rows": rows,
    }
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, out_path)
    print(f"[done] wrote {out_path} ({len(rows)} rows, {len(errors)} failures)")
    return 1 if errors and args.strict else 0


def cmd_reduce(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    shards = sorted(out_dir.glob("per_case_shard_*.json"))
    if not shards:
        raise SystemExit(f"[error] no shards found in {out_dir}")

    rows: list[dict] = []
    errors: dict[str, str] = {}
    meta: dict = {}
    for s in shards:
        payload = json.loads(s.read_text())
        rows.extend(payload["rows"])
        errors.update(payload.get("errors", {}))
        meta = {
            k: payload[k]
            for k in ("official_source", "connectivity", "thresholds", "min_component_voxels")
        }

    if errors:
        print(f"[warn] {len(errors)} cases failed: {sorted(errors)[:10]}", file=sys.stderr)
        if args.strict:
            raise SystemExit("[error] failures present and --strict set")

    by_config: dict[str, list[dict]] = {}
    for r in rows:
        by_config.setdefault(r["config"], []).append(r)

    counts = {c: len(v) for c, v in by_config.items()}
    if len(set(counts.values())) != 1:
        print(f"[warn] configurations cover different case counts: {counts}", file=sys.stderr)

    summaries = {}
    for cfg, cfg_rows in by_config.items():
        s = aggregate_means(cfg_rows)
        # the flatten-if-empty PR-AUC variant, reported alongside
        flat = np.asarray([r["pr_auc_flatten_if_empty"] for r in cfg_rows], dtype=np.float64)
        s["pr_auc_flatten_if_empty_mean"] = float(np.nanmean(flat))
        s["n_pred_empty"] = int(sum(1 for r in cfg_rows if r["pred_empty"]))
        s["n_gt_empty"] = int(sum(1 for r in cfg_rows if r["gt_empty"]))
        s["threshold"] = cfg_rows[0]["threshold"]
        s["min_component_voxels"] = cfg_rows[0]["min_component_voxels"]
        summaries[cfg] = s

    ranks = rank_aggregate(by_config)

    # Acceptance check: PR-AUC when the binary mask is used as the soft map.
    ref_cfg = config_name(args.reference_threshold, 0)
    binary_as_soft = None
    if ref_cfg in by_config:
        binary_as_soft = float(
            np.nanmean([r["pr_auc_binary_as_soft"] for r in by_config[ref_cfg]])
        )

    # Best configuration under each aggregate.
    best_by_rank = min(ranks, key=lambda c: ranks[c]["mean_rank"]) if ranks else None
    best_by_dice = max(summaries, key=lambda c: summaries[c]["dice_mean"])
    best_by_f1 = max(summaries, key=lambda c: summaries[c]["lesion_f1_mean"])

    report = {
        "title": args.title,
        "official_source": meta.get("official_source", OFFICIAL_SOURCE),
        "connectivity": meta.get("connectivity"),
        "n_configs": len(summaries),
        "n_cases": counts.get(ref_cfg, next(iter(counts.values()))),
        "n_failed": len(errors),
        "reference_config": ref_cfg,
        "pr_auc_binary_as_soft_mean": binary_as_soft,
        "best_config_by_mean_rank": best_by_rank,
        "best_config_by_dice": best_by_dice,
        "best_config_by_lesion_f1": best_by_f1,
        "summaries": summaries,
        "mean_ranks": ranks,
        "errors": sorted(errors),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    # per-case CSV for downstream analysis
    csv_path = out_dir / "per_case.csv"
    fields = [
        "subject_id",
        "config",
        "threshold",
        "min_component_voxels",
        *RANKED_METRICS,
        "pr_auc_flatten_if_empty",
        "gt_empty",
        "pred_empty",
        "voxel_volume_ml",
    ]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["subject_id"], r["config"])):
            w.writerow(r)

    # Markdown
    lines = [
        f"# {args.title}",
        "",
        f"Scored with the ISLES'26 organizers' own code (`{report['official_source']}`).",
        f"n = {report['n_cases']} cases, {report['n_configs']} configurations, "
        f"{report['n_failed']} failures. Component connectivity = {report['connectivity']}.",
        "",
        "Lower is better for absolute volume difference and absolute lesion count "
        "difference; higher is better for Dice, lesion-F1 and PR-AUC.",
        "",
        "| config | thr | minCC | Dice | AVD mL | abs count diff | lesion-F1 (RQ) | PR-AUC | "
        "PR-AUC flat-if-empty | mean rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cfg in sorted(
        summaries, key=lambda c: ranks[c]["mean_rank"] if ranks else summaries[c]["dice_mean"]
    ):
        s = summaries[cfg]
        mr = ranks[cfg]["mean_rank"] if ranks else float("nan")
        lines.append(
            f"| {cfg} | {s['threshold']:.2f} | {s['min_component_voxels']} | "
            f"{s['dice_mean']:.4f} | {s['abs_volume_diff_ml_mean']:.4f} | "
            f"{s['abs_lesion_count_diff_mean']:.4f} | {s['lesion_f1_mean']:.4f} | "
            f"{s['pr_auc_mean']:.4f} | {s['pr_auc_flatten_if_empty_mean']:.4f} | {mr:.3f} |"
        )
    lines += [
        "",
        f"- best by mean rank across all five metrics: **{best_by_rank}**",
        f"- best by Dice alone: {best_by_dice}",
        f"- best by lesion-F1 alone: {best_by_f1}",
    ]
    if binary_as_soft is not None:
        lines.append(
            f"- harness acceptance check: submitting the binary mask as the soft map at "
            f"{ref_cfg} scores PR-AUC {binary_as_soft:.4f}"
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="score a shard of cases over the sweep grid")
    s.add_argument("--gt-dir", required=True)
    s.add_argument(
        "--prob-dir",
        required=True,
        nargs="+",
        help="one or more directories (globs allowed) of float probability NIfTIs",
    )
    s.add_argument("--out-dir", required=True)
    s.add_argument("--thresholds", default="0.20:0.60:0.05")
    s.add_argument("--min-cc", default="0,5,10,20,35,50", help="min component size in VOXELS")
    s.add_argument("--connectivity", type=int, default=26, choices=(6, 18, 26))
    s.add_argument("--reference-threshold", type=float, default=0.50)
    s.add_argument("--shard-index", type=int, default=0)
    s.add_argument("--shard-count", type=int, default=1)
    s.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    s.add_argument("--cases-file", default=None)
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--strict", action="store_true")
    s.set_defaults(func=cmd_score)

    r = sub.add_parser("reduce", help="merge shards and aggregate")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--title", default="ISLES'26 official-metric evaluation")
    r.add_argument("--reference-threshold", type=float, default=0.50)
    r.add_argument("--strict", action="store_true")
    r.set_defaults(func=cmd_reduce)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
