#!/usr/bin/env python3
"""Does splitting merged predicted blobs help the two instance-based official
metrics?

Motivation. ATLAS traces lesions in a fragmented style (median 2-3 connected
components per case, max 84), and our model tends to predict one confluent blob
where the ground truth is several instances. Measured on the center-held-out
surface: min-component pruning to k=20 makes the absolute lesion count difference
WORSE on 169 cases -- those are cases where we were already UNDER-counting.

Under the official one-to-one IoU>=0.25 matching this is a genuine two-sided bet,
which is why it has to be measured rather than argued:
  - splitting a blob that really is several lesions turns 1 TP + (n-1) FN into
    n TPs, which helps RQ a lot and helps the count metric;
  - splitting a blob that really is one lesion turns 1 TP into 1 TP + (n-1) FPs
    AND can push every fragment's IoU below 0.25, losing the match entirely.
    That loses on both metrics at once.

Method: h-maxima seeds on the probability map inside each predicted component,
then a watershed constrained to that component. `h` controls how deep a valley
between two probability peaks must be before they are considered separate
lesions; h=0 disables splitting and reproduces the baseline exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.official_metrics import score_masks, score_soft  # noqa: E402

_C26 = np.ones((3, 3, 3), dtype=np.uint8)


def split_components(mask: np.ndarray, prob: np.ndarray, h: float, min_frag_voxels: int) -> np.ndarray:
    """Split each connected component of `mask` at probability valleys deeper
    than `h`. Returns a mask (the voxels are unchanged; only the component
    STRUCTURE changes, via a 1-voxel watershed line)."""
    if h <= 0:
        return mask
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed

    lab, n = ndi.label(mask, structure=_C26)
    if n == 0:
        return mask
    out = mask.copy()
    objs = ndi.find_objects(lab)
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        sub_mask = lab[sl] == i
        if int(sub_mask.sum()) < 2 * min_frag_voxels:
            continue  # too small to split into two viable instances
        sub_prob = prob[sl] * sub_mask
        seeds = h_maxima(sub_prob, h) * sub_mask
        seed_lab, n_seeds = ndi.label(seeds, structure=_C26)
        if n_seeds < 2:
            continue
        ws = watershed(-sub_prob, markers=seed_lab, mask=sub_mask, watershed_line=True)
        # Fragments that end up below the viability floor are re-merged by simply
        # not cutting them out; only the watershed LINE is removed, which is what
        # separates the instances for the connected-component counter.
        sizes = np.bincount(ws.ravel(), minlength=n_seeds + 1)[1:]
        if (sizes >= min_frag_voxels).sum() < 2:
            continue
        cut = sub_mask & (ws == 0)
        out[sl][cut] = False
    return out


def score_one(subject_id: str, gt_path: str, prob_path: str, threshold: float,
              min_cc: int, h_values: list[float], min_frag: int) -> list[dict]:
    gt_img = nib.load(gt_path)
    gt = (np.asanyarray(gt_img.dataobj) > 0.5).astype(np.uint8)
    zooms = [float(z) for z in gt_img.header.get_zooms()[:3]]
    vox_ml = float(np.prod(zooms)) / 1000.0

    prob = np.asarray(nib.load(prob_path).dataobj, dtype=np.float32)
    if prob.shape != gt.shape:
        raise ValueError(f"{subject_id}: shape mismatch {prob.shape} vs {gt.shape}")

    base = prob >= threshold
    if min_cc > 0:
        lab, n = ndi.label(base, structure=_C26)
        if n:
            counts = np.bincount(lab.ravel(), minlength=n + 1)
            keep = counts >= min_cc
            keep[0] = False
            base = keep[lab]
    pr_auc = score_soft(gt, prob)

    rows = []
    for h in h_values:
        pred = split_components(base, prob, h, min_frag).astype(np.uint8)
        dice, avd, dcount, rq = score_masks(gt, pred, vox_ml)
        rows.append({"subject_id": subject_id, "h": h, "dice": dice,
                     "abs_volume_diff_ml": avd, "abs_lesion_count_diff": dcount,
                     "lesion_f1": rq, "pr_auc": pr_auc})
    return rows


def _worker(a):
    try:
        return score_one(*a), None
    except Exception:
        return None, f"{a[0]}: {traceback.format_exc()}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prob-dir", nargs="+", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--min-cc", type=int, default=20)
    ap.add_argument("--h-values", default="0,0.10,0.20,0.30,0.40")
    ap.add_argument("--min-frag-voxels", type=int, default=20)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    args = ap.parse_args(argv)

    import glob
    probs = {}
    for pat in args.prob_dir:
        for d in sorted(glob.glob(pat)):
            for p in sorted(Path(d).glob("*.nii.gz")):
                probs[p.name[: -len(".nii.gz")]] = p
    gt_dir = Path(args.gt_dir)
    subjects = sorted(s for s in probs if (gt_dir / f"{s}.nii.gz").is_file())
    shard = subjects[args.shard_index :: args.shard_count]
    h_values = [float(x) for x in args.h_values.split(",")]
    print(f"[info] shard {args.shard_index}/{args.shard_count}: {len(shard)} cases x {len(h_values)} h", flush=True)

    tasks = [(s, str(gt_dir / f"{s}.nii.gz"), str(probs[s]), args.threshold,
              args.min_cc, h_values, args.min_frag_voxels) for s in shard]
    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            r, e = f.result()
            (errors.append(e) if e else rows.extend(r))
            if e:
                print(f"[error] {e}", file=sys.stderr)
            if i % 25 == 0:
                print(f"[progress] {i}/{len(tasks)}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"shard_{args.shard_index:03d}.json"
    out.write_text(json.dumps({"rows": rows, "errors": errors,
                               "threshold": args.threshold, "min_cc": args.min_cc,
                               "h_values": h_values}))
    print(f"[done] {out} ({len(rows)} rows, {len(errors)} errors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
