#!/usr/bin/env python3
"""Extract TEST-TIME-AVAILABLE features per case, for fitting an adaptive
operating point (per-case probability threshold and min-component size).

The constraint that defines this file: every feature here must be computable at
submission time from the soft map, the image geometry and the challenge-supplied
metadata. Nothing may touch the ground truth. A feature that needs the label is
not a feature, it is a leak.

Why this exists: on the center-held-out surface, an ORACLE per-case choice of
(threshold, min-CC) would score AVD 6.26 -> 4.50 mL and |lesion count diff|
1.86 -> 1.26 versus the best single fixed configuration. Those two metrics are
essentially uncorrelated with Dice (Spearman -0.081 and +0.213), so they are not
improved by making the model better -- only by choosing the operating point
per case. This extracts the inputs a rule would need to try.

Emits, per case:
  - geometry: voxel spacing, anisotropy, volume, grid size
  - per-threshold: predicted volume (mL), connected-component count, and the
    component count surviving each candidate min-size filter
  - soft-map shape: mass, entropy proxy, confidence distribution
and, for supervision only (never as a feature), the ground-truth volume and
component count.
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

_C26 = np.ones((3, 3, 3), dtype=np.uint8)

THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
MIN_CCS = [0, 5, 10, 20, 35, 50]


def features_for_case(subject_id: str, prob_path: str, gt_path: str) -> dict:
    prob_img = nib.load(prob_path)
    prob = np.asarray(prob_img.dataobj, dtype=np.float32)
    zooms = [float(z) for z in prob_img.header.get_zooms()[:3]]
    vox_ml = float(np.prod(zooms)) / 1000.0

    f: dict = {
        "subject_id": subject_id,
        "vox_mm_x": zooms[0], "vox_mm_y": zooms[1], "vox_mm_z": zooms[2],
        "vox_volume_ml": vox_ml,
        "anisotropy": max(zooms) / max(min(zooms), 1e-6),
        "is_iso_1mm": float(all(abs(z - 1.0) < 0.01 for z in zooms)),
        "n_voxels": int(prob.size),
        "grid_volume_ml": float(prob.size) * vox_ml,
    }

    # Soft-map shape. `mass` is the expected lesion volume under the model's own
    # probabilities -- a threshold-free volume estimate, and the single most
    # useful predictor of the true volume.
    f["prob_mass_ml"] = float(prob.sum()) * vox_ml
    f["prob_max"] = float(prob.max())
    fg = prob[prob > 0.05]
    f["prob_fg_mean"] = float(fg.mean()) if fg.size else 0.0
    f["prob_fg_std"] = float(fg.std()) if fg.size else 0.0
    f["prob_fg_frac"] = float(fg.size) / prob.size
    # How decisive is the model? A map concentrated near 0/1 is confident; one
    # smeared through the middle is not, and that is exactly when the optimal
    # threshold moves.
    mid = float(((prob > 0.2) & (prob < 0.8)).sum())
    hi = float((prob >= 0.8).sum())
    f["prob_ambiguous_frac"] = mid / max(hi + mid, 1.0)

    for t in THRESHOLDS:
        m = prob >= t
        n_vox = int(m.sum())
        tag = f"t{int(round(t * 100)):02d}"
        f[f"vol_{tag}_ml"] = n_vox * vox_ml
        if n_vox == 0:
            for k in MIN_CCS:
                f[f"ncomp_{tag}_k{k}"] = 0
            f[f"largest_{tag}_ml"] = 0.0
            continue
        lab, n = ndi.label(m, structure=_C26)
        counts = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        for k in MIN_CCS:
            f[f"ncomp_{tag}_k{k}"] = int((counts >= max(k, 1)).sum())
        f[f"largest_{tag}_ml"] = float(counts.max()) * vox_ml

    # Supervision targets. NOT features -- the fitting script must never put
    # these on the input side.
    gt_img = nib.load(gt_path)
    gt = np.asarray(np.asanyarray(gt_img.dataobj) > 0.5)
    f["target_gt_volume_ml"] = float(gt.sum()) * vox_ml
    _, n_gt = ndi.label(gt, structure=_C26)
    f["target_gt_ncomp"] = int(n_gt)
    return f


def _worker(args):
    sid = args[0]
    try:
        return features_for_case(*args), None
    except Exception:
        return None, f"{sid}: {traceback.format_exc()}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prob-dir", nargs="+", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    args = ap.parse_args(argv)

    import glob
    probs: dict[str, Path] = {}
    for pattern in args.prob_dir:
        for d in sorted(glob.glob(pattern)):
            for p in sorted(Path(d).glob("*.nii.gz")):
                probs[p.name[: -len(".nii.gz")]] = p
    gt_dir = Path(args.gt_dir)

    # Resume support. This job runs on a preemptible partition and takes ~20
    # minutes; without incremental persistence a preemption at minute 19 throws
    # away everything. Partial rows are appended as they complete and reloaded on
    # restart, so a requeue costs only the in-flight cases.
    out = Path(args.out)
    partial = out.with_suffix(".partial.jsonl")
    done: dict[str, dict] = {}
    if partial.is_file():
        for line in partial.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn final line from a kill mid-write
            done[r["subject_id"]] = r
        print(f"[resume] {len(done)} cases already extracted", flush=True)

    tasks = [
        (sid, str(p), str(gt_dir / f"{sid}.nii.gz"))
        for sid, p in sorted(probs.items())
        if (gt_dir / f"{sid}.nii.gz").is_file() and sid not in done
    ]
    print(f"[info] {len(tasks)} cases to do, {args.workers} workers", flush=True)

    rows, errors = list(done.values()), []
    partial.parent.mkdir(parents=True, exist_ok=True)
    sink = partial.open("a")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            r, err = fut.result()
            if err:
                errors.append(err)
                print(f"[error] {err}", file=sys.stderr)
            else:
                rows.append(r)
                sink.write(json.dumps(r) + "\n")
                sink.flush()
            if i % 100 == 0:
                print(f"[progress] {i}/{len(tasks)}", flush=True)
    sink.close()

    rows.sort(key=lambda r: r["subject_id"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if not errors:
        partial.unlink(missing_ok=True)
    print(f"[done] wrote {out} ({len(rows)} rows, {len(errors)} errors)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
