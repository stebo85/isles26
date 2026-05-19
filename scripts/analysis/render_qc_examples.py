"""Render T1 + lesion overlay PNGs for a few representative ATLAS R2.1 cases.

Picks: empty-lesion (if any), smallest non-empty, median, p95, largest, plus a few
random per-site exemplars. Each PNG: axial + coronal + sagittal of slice through
lesion centroid (or volume center if empty).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi


def pick_slice(mask, axis):
    if mask.sum() == 0:
        return mask.shape[axis] // 2
    com = ndi.center_of_mass(mask)
    return int(round(com[axis]))


def overlay(ax, base, mask, title):
    ax.imshow(np.rot90(base), cmap="gray", interpolation="nearest")
    if mask.sum() > 0:
        m = np.ma.masked_where(mask == 0, mask)
        ax.imshow(np.rot90(m), cmap="autumn", alpha=0.5, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def render_case(t1_path, lesion_path, out_png, label):
    t1 = nib.load(str(t1_path))
    les = nib.load(str(lesion_path))
    t1d = np.asanyarray(t1.dataobj).astype(np.float32)
    m = (np.asanyarray(les.dataobj) > 0).astype(np.uint8)
    p1, p99 = np.percentile(t1d, [1, 99])
    t1d = np.clip(t1d, p1, p99)

    z = pick_slice(m, 2)
    y = pick_slice(m, 1)
    x = pick_slice(m, 0)
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    overlay(axes[0], t1d[:, :, z], m[:, :, z], f"axial z={z}")
    overlay(axes[1], t1d[:, y, :], m[:, y, :], f"coronal y={y}")
    overlay(axes[2], t1d[x, :, :], m[x, :, :], f"sagittal x={x}")
    fig.suptitle(label, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = df[df["status"] == "ok"].copy()
    df["lesion_volume_ml"] = pd.to_numeric(df["lesion_volume_ml"], errors="coerce")

    picks = []
    nz = df[df["lesion_volume_ml"] > 0]
    if len(df) > len(nz):
        picks.append(("empty", df[df["lesion_volume_ml"] == 0].iloc[0]))
    picks.append(("smallest_nonzero", nz.nsmallest(1, "lesion_volume_ml").iloc[0]))
    picks.append(("p25", nz.iloc[(nz["lesion_volume_ml"].rank(pct=True) - 0.25).abs().argsort()[:1]].iloc[0]))
    picks.append(("median", nz.iloc[(nz["lesion_volume_ml"].rank(pct=True) - 0.50).abs().argsort()[:1]].iloc[0]))
    picks.append(("p75", nz.iloc[(nz["lesion_volume_ml"].rank(pct=True) - 0.75).abs().argsort()[:1]].iloc[0]))
    picks.append(("p95", nz.iloc[(nz["lesion_volume_ml"].rank(pct=True) - 0.95).abs().argsort()[:1]].iloc[0]))
    picks.append(("largest", nz.nlargest(1, "lesion_volume_ml").iloc[0]))

    # Highest-CC count case (most fragmented).
    df["lesion_cc_count"] = pd.to_numeric(df["lesion_cc_count"], errors="coerce")
    picks.append(("most_fragmented", df.nlargest(1, "lesion_cc_count").iloc[0]))

    # One sample per top-5 site by count.
    top_sites = df["site"].value_counts().head(5).index.tolist()
    rng = np.random.default_rng(7)
    for site in top_sites:
        sub = df[df["site"] == site]
        row = sub.iloc[int(rng.integers(0, len(sub)))]
        picks.append((f"site_{site}_random", row))

    for label, row in picks:
        out_png = out_dir / f"example_{label}__{row['subject']}_{row['session']}.png"
        title = (f"{label} | {row['site']}/{row['subject']}/{row['session']} | "
                 f"DPS={row['days_post_stroke']} | vol={row['lesion_volume_ml']:.2f} mL | "
                 f"CC={row['lesion_cc_count']}")
        print(f"render {label} -> {out_png}", flush=True)
        try:
            render_case(row["t1_path"], row["lesion_path"], out_png, title)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
