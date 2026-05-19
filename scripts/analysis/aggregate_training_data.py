"""Aggregate per-session ATLAS R2.1 characterization into summary JSON + QC plots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summarize_numeric(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {"count": 0}
    return {
        "count": int(s.size),
        "min": float(s.min()),
        "p01": float(s.quantile(0.01)),
        "p05": float(s.quantile(0.05)),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    qc_dir = out_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    n_total = len(df)
    df_ok = df[df["status"] == "ok"].copy()
    n_ok = len(df_ok)

    # Numeric coercions.
    for col in [
        "days_post_stroke", "vox_mm_x", "vox_mm_y", "vox_mm_z", "vox_mm3",
        "t1_min", "t1_p01", "t1_p50", "t1_p99", "t1_max", "t1_mean", "t1_std", "t1_nonzero_frac",
        "lesion_voxels", "lesion_volume_ml", "lesion_cc_count", "lesion_largest_cc_ml",
        "t1_file_bytes", "lesion_file_bytes",
    ]:
        if col in df_ok.columns:
            df_ok[col] = pd.to_numeric(df_ok[col], errors="coerce")

    # Sanity flags.
    flags = {
        "missing_files": int((df["status"] == "missing_files").sum()),
        "errors": int((df["status"] == "error").sum()),
        "shape_mismatch": int((df_ok["shape_match"].astype(str) == "0").sum()),
        "affine_mismatch": int((df_ok["affine_match_t1"].astype(str) == "0").sum()),
        "empty_lesion": int((df_ok["lesion_voxels"] == 0).sum()),
        "tiny_lesion_lt0p05ml": int((df_ok["lesion_volume_ml"] < 0.05).sum()),
        "huge_lesion_gt250ml": int((df_ok["lesion_volume_ml"] > 250).sum()),
        "non_binary_mask": int((~df_ok["lesion_unique_labels"].fillna("").isin(["0", "0,1", "1"])).sum()),
        "missing_days_post_stroke": int(df_ok["days_post_stroke"].isna().sum()),
    }

    # Chronicity bins: DPS<=0 treated as missing/unknown (R049/R050 encode
    # missing as 0; one R049 session is -4). (0, 180) = acute/subacute,
    # [180, inf) = chronic. NaN stays unknown.
    dps = df_ok["days_post_stroke"]
    df_ok["chronicity"] = np.where(
        dps.isna() | (dps <= 0), "unknown",
        np.where(dps < 180, "lt180d", "ge180d"),
    )

    # Per-site rollup.
    per_site = (
        df_ok.groupby("site")
        .agg(
            n=("subject", "size"),
            dice_lesion_ml_median=("lesion_volume_ml", "median"),
            lesion_ml_p95=("lesion_volume_ml", lambda x: float(x.quantile(0.95))),
            empty_lesion=("lesion_voxels", lambda x: int((x == 0).sum())),
            vox_mm3_median=("vox_mm3", "median"),
            iso_1mm_frac=("vox_mm3", lambda x: float(((x > 0.9) & (x < 1.1)).mean())),
            cc_median=("lesion_cc_count", "median"),
            cc_p95=("lesion_cc_count", lambda x: float(x.quantile(0.95))),
            days_post_stroke_median=("days_post_stroke", "median"),
            chronic_frac=("chronicity", lambda x: float((x == "ge180d").mean())),
        )
        .sort_values("n", ascending=False)
    )
    per_site_path = out_dir / "per_site_summary.csv"
    per_site.to_csv(per_site_path)

    # Per-chronicity rollup.
    per_chronicity = (
        df_ok.groupby("chronicity")
        .agg(
            n=("subject", "size"),
            lesion_ml_median=("lesion_volume_ml", "median"),
            lesion_ml_p95=("lesion_volume_ml", lambda x: float(x.quantile(0.95))),
            empty_lesion=("lesion_voxels", lambda x: int((x == 0).sum())),
            cc_median=("lesion_cc_count", "median"),
            days_post_stroke_median=("days_post_stroke", "median"),
        )
    )
    per_chronicity.to_csv(out_dir / "per_chronicity_summary.csv")

    summary = {
        "n_sessions_total": n_total,
        "n_sessions_ok": n_ok,
        "n_sites": int(df_ok["site"].nunique()),
        "n_subjects": int(df_ok["subject"].nunique()),
        "flags": flags,
        "shapes_unique_top": (
            df_ok["t1_shape"].value_counts().head(10).to_dict()
        ),
        "orient_unique": df_ok["orient"].value_counts().to_dict(),
        "dtype_t1_unique": df_ok["t1_dtype"].value_counts().to_dict(),
        "dtype_lesion_unique": df_ok["lesion_dtype"].value_counts().to_dict(),
        "label_sets_unique": df_ok["lesion_unique_labels"].fillna("").value_counts().to_dict(),
        "voxel_mm_x": summarize_numeric(df_ok["vox_mm_x"]),
        "voxel_mm_y": summarize_numeric(df_ok["vox_mm_y"]),
        "voxel_mm_z": summarize_numeric(df_ok["vox_mm_z"]),
        "voxel_mm3": summarize_numeric(df_ok["vox_mm3"]),
        "lesion_volume_ml": summarize_numeric(df_ok["lesion_volume_ml"]),
        "lesion_largest_cc_ml": summarize_numeric(df_ok["lesion_largest_cc_ml"]),
        "lesion_cc_count": summarize_numeric(df_ok["lesion_cc_count"]),
        "days_post_stroke": summarize_numeric(df_ok["days_post_stroke"]),
        "t1_p50": summarize_numeric(df_ok["t1_p50"]),
        "t1_p99": summarize_numeric(df_ok["t1_p99"]),
        "t1_nonzero_frac": summarize_numeric(df_ok["t1_nonzero_frac"]),
        "chronicity_counts": df_ok["chronicity"].value_counts().to_dict(),
        "isotropic_1mm_fraction": float(
            ((df_ok["vox_mm3"] > 0.9) & (df_ok["vox_mm3"] < 1.1)).mean()
        ),
        "per_site_subject_count": (
            df_ok.groupby("site")["subject"].nunique().to_dict()
        ),
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # QC plots --- size-stratified, log-scaled lesion volumes.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vols = df_ok["lesion_volume_ml"].clip(lower=1e-3)
    ax.hist(np.log10(vols), bins=60, color="#3366cc", alpha=0.85)
    ax.set_xlabel("log10(lesion volume, mL)")
    ax.set_ylabel("# sessions")
    ax.set_title(f"Lesion volume distribution (n={n_ok})")
    for x, label in [(np.log10(0.1), "0.1 mL"), (np.log10(1), "1 mL"),
                     (np.log10(10), "10 mL"), (np.log10(100), "100 mL")]:
        ax.axvline(x, color="k", lw=0.5, ls="--")
        ax.text(x, ax.get_ylim()[1] * 0.95, label, rotation=90, fontsize=8, va="top")
    fig.tight_layout()
    fig.savefig(qc_dir / "lesion_volume_hist.png", dpi=130)
    plt.close(fig)

    # Voxel size cloud.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(df_ok["vox_mm_x"], df_ok["vox_mm_z"], s=4, alpha=0.3, color="#993322")
    ax.set_xlabel("in-plane voxel size x (mm)")
    ax.set_ylabel("slice thickness z (mm)")
    ax.set_title("T1w voxel geometry (x vs z)")
    ax.axvline(1, ls=":", color="k", lw=0.5)
    ax.axhline(1, ls=":", color="k", lw=0.5)
    fig.tight_layout()
    fig.savefig(qc_dir / "voxel_xz_scatter.png", dpi=130)
    plt.close(fig)

    # Days-post-stroke distribution.
    dps = pd.to_numeric(df_ok["days_post_stroke"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(dps.clip(upper=3650), bins=60, color="#229955")
    ax.axvline(180, color="k", ls="--", lw=0.8, label="180 d cutoff")
    ax.set_xlabel("Days post-stroke (clipped at 3650)")
    ax.set_ylabel("# sessions")
    ax.set_title("DAYS_POST_STROKE distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(qc_dir / "days_post_stroke_hist.png", dpi=130)
    plt.close(fig)

    # Per-site counts.
    site_counts = df_ok["site"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(4.5, 0.25 * len(site_counts))))
    ax.barh(site_counts.index, site_counts.values, color="#445577")
    ax.set_xlabel("# sessions")
    ax.set_title("Sessions per center")
    fig.tight_layout()
    fig.savefig(qc_dir / "sessions_per_site.png", dpi=130)
    plt.close(fig)

    # CC count vs volume.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cc = df_ok["lesion_cc_count"].clip(lower=0)
    vol = df_ok["lesion_volume_ml"].clip(lower=1e-3)
    ax.scatter(np.log10(vol), cc, s=4, alpha=0.3, color="#552288")
    ax.set_xlabel("log10(lesion volume, mL)")
    ax.set_ylabel("# connected components")
    ax.set_title("Lesion fragmentation vs total volume")
    fig.tight_layout()
    fig.savefig(qc_dir / "cc_vs_volume.png", dpi=130)
    plt.close(fig)

    print(json.dumps(summary, indent=2)[:1500])
    print(f"wrote {out_dir / 'summary.json'}")
    print(f"wrote {per_site_path}")


if __name__ == "__main__":
    main()
