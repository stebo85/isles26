#!/usr/bin/env python3
"""Render QC panels for submission-container predictions.

One row per case: the T1w slice at the lesion's centre of mass, with ground
truth and prediction as contours, plus the soft probability map. Contours rather
than filled overlays, because a filled mask hides whether the boundary is right,
and boundary quality is what separates a 0.6 Dice from a 0.9.

The slice is chosen at the GT centre of mass so the panel cannot be flattered by
picking the best slice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


def best_slice(gt: np.ndarray, pred: np.ndarray) -> tuple[int, int]:
    """Axis and index of the slice through the lesion centre of mass."""
    ref = gt if gt.any() else pred
    if not ref.any():
        return 2, ref.shape[2] // 2
    com = ndi.center_of_mass(ref)
    # Use the axis with the most in-plane lesion area at that centre.
    areas = []
    for ax in range(3):
        idx = int(round(com[ax]))
        idx = max(0, min(ref.shape[ax] - 1, idx))
        areas.append(np.take(ref, idx, axis=ax).sum())
    ax = int(np.argmax(areas))
    return ax, max(0, min(ref.shape[ax] - 1, int(round(com[ax]))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", required=True, type=Path,
                    help="Directory containing per-case native-space NIfTIs")
    ap.add_argument("--report-dir", type=Path,
                    help="Directory containing summary.json and receiving the panel "
                         "(default: --in-dir)")
    args = ap.parse_args()
    report_dir = args.report_dir or args.in_dir

    rows = json.loads((report_dir / "summary.json").read_text())
    suffixes = ("t1w", "ground_truth", "prediction", "probability")
    rows = [
        r for r in rows
        if all((args.in_dir / f"{r['case']}_{suffix}.nii.gz").is_file()
               for suffix in suffixes)
    ]
    if not rows:
        print("[error] no complete native-space NIfTI sets to render")
        return 1

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(10.5, 3.5 * n))
    if n == 1:
        axes = axes[None, :]

    for i, r in enumerate(rows):
        case = r["case"]
        load = lambda suffix: np.asanyarray(
            nib.load(str(args.in_dir / f"{case}_{suffix}.nii.gz")).dataobj
        )
        t1 = load("t1w").astype(np.float32)
        gt = load("ground_truth") > 0.5
        pred = load("prediction") > 0.5
        prob = load("probability").astype(np.float32)
        ax_i, sl = best_slice(gt, pred)
        take = lambda a: np.rot90(np.take(a, sl, axis=ax_i))
        t1s, gts, ps, prs = take(t1), take(gt), take(pred), take(prob)

        lo, hi = np.percentile(t1s[t1s > 0], [1, 99]) if (t1s > 0).any() else (0, 1)

        axes[i, 0].imshow(t1s, cmap="gray", vmin=lo, vmax=hi)
        axes[i, 0].set_ylabel(f"{r['case']}\nfold {r['fold']}", fontsize=9)
        axes[i, 0].set_title("T1w (native)", fontsize=9)

        axes[i, 1].imshow(t1s, cmap="gray", vmin=lo, vmax=hi)
        if gts.any():
            axes[i, 1].contour(gts.astype(float), levels=[0.5], colors="#2ca02c", linewidths=1.1)
        if ps.any():
            axes[i, 1].contour(ps.astype(float), levels=[0.5], colors="#d62728", linewidths=1.1)
        axes[i, 1].set_title(
            f"GT (green) vs prediction (red)\nDice {r['dice']:.3f}  |  "
            f"{r['pred_ml']:.1f} vs {r['gt_ml']:.1f} mL", fontsize=9)

        im = axes[i, 2].imshow(prs, cmap="magma", vmin=0, vmax=1)
        axes[i, 2].set_title("soft probability map", fontsize=9)
        fig.colorbar(im, ax=axes[i, 2], fraction=0.046, shrink=0.85)

        for a in axes[i]:
            a.set_xticks([]); a.set_yticks([])

    fig.suptitle(
        "ISLES'26 submission container — predictions on RAW native-space cases\n"
        "each case predicted with ONLY the fold that held it out; cases sampled across the lesion-size range",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / "qc_panel.png"
    fig.savefig(out, dpi=110)
    print(f"[done] wrote {out}")

    # A compact per-case table alongside the figure.
    lines = ["| case | fold | GT mL | pred mL | Dice |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['case']} | {r['fold']} | {r['gt_ml']:.2f} | {r['pred_ml']:.2f} | {r['dice']:.4f} |")
    (report_dir / "qc_panel.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
