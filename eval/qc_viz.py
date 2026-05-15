"""Generate PNG QC overlays of prediction vs ground truth for each subject.

For each subject, picks the axial slice that contains the most GT voxels (so
the lesion is visible), shows the underlying TRACE image in greyscale, and
overlays GT in green and prediction in red (intersection in yellow).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval.loader import discover_soop_bench, load_mask, resample_to_reference
from eval.run_eval import find_prediction


def _norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, (1, 99))
    if hi <= lo:
        return np.zeros_like(x)
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    return x


def _best_slice_axis(mask: np.ndarray) -> tuple[int, int]:
    """Return (axis, slice_index) of the slice with most foreground voxels."""
    if not mask.any():
        return 2, mask.shape[2] // 2
    counts = [mask.sum(axis=(1, 2)), mask.sum(axis=(0, 2)), mask.sum(axis=(0, 1))]
    # Default to axial (axis=2) when tied.
    best_axis = 2
    best_idx = int(np.argmax(counts[2]))
    if counts[2].max() < 1:
        for ax, c in enumerate(counts):
            if c.max() > 0:
                best_axis = ax
                best_idx = int(np.argmax(c))
                break
    return best_axis, best_idx


def _take_slice(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    s = [slice(None)] * 3
    s[axis] = idx
    return vol[tuple(s)]


def make_overlay(trace_path: Path, gt: np.ndarray, pred: np.ndarray,
                 out_png: Path, subject_id: str, dice: float, f1: float) -> None:
    img = nib.load(str(trace_path))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]

    axis, idx = _best_slice_axis(gt)
    bg_slice = _norm01(_take_slice(data, axis, idx))
    gt_slice = _take_slice(gt, axis, idx).astype(bool)
    pr_slice = _take_slice(pred, axis, idx).astype(bool)

    rgb = np.dstack([bg_slice, bg_slice, bg_slice])
    only_gt = gt_slice & ~pr_slice
    only_pr = pr_slice & ~gt_slice
    both = gt_slice & pr_slice
    rgb[only_gt] = [0.1, 0.9, 0.1]   # GT only -> green (FN)
    rgb[only_pr] = [0.95, 0.1, 0.1]  # pred only -> red (FP)
    rgb[both]    = [0.95, 0.9, 0.05] # intersection -> yellow (TP)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(np.rot90(rgb))
    ax.set_title(f"{subject_id}  axis={['sag','cor','ax'][axis]} slice={idx}\nDice={dice:.3f}  LesionF1={f1:.3f}\nGreen=FN  Red=FP  Yellow=TP",
                 fontsize=10)
    ax.set_axis_off()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--report-json", required=True, type=Path,
                    help="report.json produced by run_eval (used to look up per-case Dice/F1).")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    import json
    rep = json.loads(args.report_json.read_text())
    per_case_index = {c["subject_id"]: c for c in rep["per_case"]}

    subjects = discover_soop_bench(args.data_root)
    for s in subjects:
        gt_img = nib.load(str(s.lesion_mask))
        gt_data, _, _ = load_mask(s.lesion_mask)
        pred_path = find_prediction(args.pred_dir, s.subject_id)
        if pred_path is None:
            pred_data = np.zeros_like(gt_data)
        else:
            pred_data = resample_to_reference(pred_path, gt_img)
        info = per_case_index.get(s.subject_id, {"dice": float("nan"), "lesion_f1": float("nan")})
        make_overlay(s.trace, gt_data, pred_data,
                     args.out_dir / f"{s.subject_id}.png",
                     s.subject_id, info["dice"], info["lesion_f1"])
        print(f"wrote {args.out_dir / (s.subject_id + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
