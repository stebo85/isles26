"""End-to-end soop_bench evaluation runner.

Inputs:
    --data-root   path to data/soop_bench
    --pred-dir    a directory containing predictions named:
                      <sub-N>.nii.gz                       (preferred)
                      <sub-N>_space-TRACE_desc-lesion_mask.nii.gz
                  Each prediction is resampled (nearest-neighbour) onto the
                  reference GT grid before metrics are computed.
    --out-dir     output directory for JSON + Markdown report
    --title       title for the Markdown report

Output:
    <out>/per_case.json    one entry per subject with all metrics
    <out>/report.json      aggregated report
    <out>/report.md        human-readable report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from eval.loader import discover_soop_bench, load_mask, resample_to_reference
from eval.metrics import evaluate_case
from eval.report import aggregate, write_json, write_markdown


def find_prediction(pred_dir: Path, subject_id: str) -> Path | None:
    candidates = [
        pred_dir / f"{subject_id}.nii.gz",
        pred_dir / f"{subject_id}_space-TRACE_desc-lesion_mask.nii.gz",
        pred_dir / f"{subject_id}_pred.nii.gz",
        pred_dir / subject_id / f"{subject_id}_pred.nii.gz",
        pred_dir / subject_id / f"{subject_id}.nii.gz",
    ]
    for c in candidates:
        if c.exists():
            return c
    # fallback: any *.nii.gz inside a per-subject folder
    sub_dir = pred_dir / subject_id
    if sub_dir.exists():
        nii = sorted(sub_dir.glob("*.nii.gz"))
        if nii:
            return nii[0]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--title", default="Stroke segmentation benchmark on soop_bench")
    ap.add_argument("--min-overlap-voxels", type=int, default=1)
    ap.add_argument("--allow-missing", action="store_true",
                    help="If set, subjects without a prediction file are evaluated as empty predictions (counts as misses).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    subjects = discover_soop_bench(args.data_root)
    if not subjects:
        print("no subjects found", file=sys.stderr)
        return 2

    per_case: list[dict] = []
    n_missing = 0
    for s in subjects:
        gt_img = nib.load(str(s.lesion_mask))
        gt_data, _, voxel_size = load_mask(s.lesion_mask)
        pred_path = find_prediction(args.pred_dir, s.subject_id)
        if pred_path is None:
            if not args.allow_missing:
                print(f"[warn] no prediction for {s.subject_id} — skipping", file=sys.stderr)
                n_missing += 1
                continue
            pred_data = np.zeros_like(gt_data, dtype=np.uint8)
            print(f"[info] no prediction for {s.subject_id} — evaluating as empty mask", file=sys.stderr)
            n_missing += 1
        else:
            pred_data = resample_to_reference(pred_path, gt_img)
        cm = evaluate_case(
            pred_data, gt_data, voxel_size,
            subject_id=s.subject_id,
            chronicity=s.chronicity,
            center=s.center,
            min_overlap_voxels=args.min_overlap_voxels,
        )
        d = cm.to_dict()
        d["prediction_path"] = str(pred_path) if pred_path else None
        d["gt_path"] = str(s.lesion_mask)
        d["sidecar_metadata"] = s.metadata
        per_case.append(d)
        print(f"{s.subject_id}: Dice={cm.overlap.dice:.4f} F1={cm.lesionwise.lesion_f1:.4f} "
              f"HD95={cm.boundary.hd95_mm:.2f}mm GTml={cm.volume.gt_vol_ml:.2f} "
              f"PredML={cm.volume.pred_vol_ml:.2f} GTcomps={cm.lesionwise.gt_lesion_count} "
              f"PredComps={cm.lesionwise.pred_lesion_count}")

    (args.out_dir / "per_case.json").write_text(json.dumps(per_case, indent=2, default=str))
    report = aggregate(per_case)
    report["n_missing_predictions"] = n_missing
    write_json(report, args.out_dir / "report.json")
    write_markdown(report, args.out_dir / "report.md", title=args.title)

    print(f"\nWrote {args.out_dir / 'report.md'}")
    if n_missing:
        print(f"({n_missing} subjects had no prediction file)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
