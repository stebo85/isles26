"""Per-session characterization of ATLAS R2.1 raw training data.

For each session, record:
  - subject_id, session_id, site, days_post_stroke, chronicity (computed: <180d => 1)
  - T1w voxel sizes (mm), shape, orientation, affine determinant, data dtype, intensity stats
  - lesion mask shape match, voxel count, lesion volume (mL), connected-component count,
    largest-CC volume (mL), background match with T1, label uniqueness
  - file sizes (bytes)

Writes one CSV row per session to OUT_CSV (created if missing, appended otherwise).
This is parallelizable: each worker handles a slice of the subject list.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi


FIELDS = [
    "site", "subject", "session", "days_post_stroke", "chronicity_lt180d",
    "t1_path", "lesion_path",
    "t1_shape", "lesion_shape",
    "vox_mm_x", "vox_mm_y", "vox_mm_z", "vox_mm3",
    "orient", "det_sign",
    "t1_dtype",
    "t1_min", "t1_p01", "t1_p50", "t1_p99", "t1_max", "t1_mean", "t1_std",
    "t1_nonzero_frac", "t1_brain_p50", "t1_brain_p99",
    "lesion_dtype",
    "lesion_unique_labels", "lesion_voxels", "lesion_volume_ml",
    "lesion_cc_count", "lesion_largest_cc_ml",
    "shape_match", "affine_match_t1",
    "t1_file_bytes", "lesion_file_bytes",
    "status", "error",
]


def find_sessions(root: Path):
    """Yield (site, subject, session, t1_path, lesion_path, metadata_csv) tuples."""
    for site_dir in sorted(root.iterdir()):
        if not site_dir.is_dir() or not site_dir.name.startswith("R"):
            continue
        site = site_dir.name
        for sub_dir in sorted(site_dir.iterdir()):
            if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
                continue
            subject = sub_dir.name
            for ses_dir in sorted(sub_dir.iterdir()):
                if not ses_dir.is_dir() or not ses_dir.name.startswith("ses-"):
                    continue
                session = ses_dir.name
                anat = ses_dir / "anat"
                if not anat.is_dir():
                    continue
                t1s = sorted(anat.glob("*_T1w.nii.gz"))
                masks = sorted(anat.glob("*label-lesion*_mask.nii.gz"))
                metas = sorted(anat.glob("*_metadata.csv"))
                t1 = t1s[0] if t1s else None
                mask = masks[0] if masks else None
                meta = metas[0] if metas else None
                yield site, subject, session, t1, mask, meta


def parse_metadata(path: Path):
    """Return (days_post_stroke or None, site or None) from the per-session csv."""
    if path is None or not path.is_file():
        return None, None
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None, None
        r = rows[0]
        dps_raw = r.get("DAYS_POST_STROKE", "").strip()
        dps = float(dps_raw) if dps_raw not in ("", "NA", "NaN") else None
        site = r.get("SITE", "").strip() or None
        return dps, site
    except Exception:
        return None, None


def affine_orient(affine: np.ndarray) -> str:
    try:
        return "".join(nib.orientations.aff2axcodes(affine))
    except Exception:
        return "?"


def percentiles(a: np.ndarray):
    a = a.ravel()
    if a.size == 0:
        return tuple([np.nan] * 5)
    # nan* variants: guard against silent NaN propagation from float-stored T1s.
    return (
        float(np.nanmin(a)), float(np.nanpercentile(a, 1)), float(np.nanpercentile(a, 50)),
        float(np.nanpercentile(a, 99)), float(np.nanmax(a)),
    )


# 26-connectivity matches eval/metrics.py:_CONNECTIVITY_26 so CC counts here
# are directly comparable to the lesion-wise F1 metric.
_CC26 = ndi.generate_binary_structure(3, 3)


def analyze_session(site, subject, session, t1_path, lesion_path, meta_path):
    row = {k: "" for k in FIELDS}
    row.update({"site": site, "subject": subject, "session": session,
                "t1_path": str(t1_path) if t1_path else "",
                "lesion_path": str(lesion_path) if lesion_path else ""})
    dps, meta_site = parse_metadata(meta_path)
    row["days_post_stroke"] = "" if dps is None else f"{dps:g}"
    # Treat DPS<=0 as missing/unknown (R049/R050 encode missing as 0; one R049
    # session is -4). Anything in (0, 180) is lt180d; everything else is ge180d.
    if dps is None or dps <= 0:
        row["chronicity_lt180d"] = ""
    else:
        row["chronicity_lt180d"] = "1" if dps < 180 else "0"
    if meta_site and meta_site != site:
        # Folder name vs csv site disagreement — surface but trust folder name.
        row["error"] = f"site_csv={meta_site}!=folder={site}"

    if t1_path is None or lesion_path is None:
        row["status"] = "missing_files"
        return row

    try:
        row["t1_file_bytes"] = str(os.path.getsize(t1_path))
        row["lesion_file_bytes"] = str(os.path.getsize(lesion_path))

        t1_img = nib.load(str(t1_path))
        lesion_img = nib.load(str(lesion_path))

        t1_shape = tuple(int(x) for x in t1_img.shape)
        lesion_shape = tuple(int(x) for x in lesion_img.shape)
        row["t1_shape"] = "x".join(map(str, t1_shape))
        row["lesion_shape"] = "x".join(map(str, lesion_shape))
        row["shape_match"] = "1" if t1_shape == lesion_shape else "0"
        row["affine_match_t1"] = "1" if np.allclose(t1_img.affine, lesion_img.affine, atol=1e-3) else "0"

        zooms = t1_img.header.get_zooms()[:3]
        vx, vy, vz = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
        row["vox_mm_x"], row["vox_mm_y"], row["vox_mm_z"] = f"{vx:.4f}", f"{vy:.4f}", f"{vz:.4f}"
        row["vox_mm3"] = f"{vx*vy*vz:.6f}"
        row["orient"] = affine_orient(t1_img.affine)
        row["det_sign"] = "1" if np.linalg.det(t1_img.affine[:3, :3]) > 0 else "-1"

        t1_data = np.asanyarray(t1_img.dataobj)
        row["t1_dtype"] = str(t1_data.dtype)
        t1f = t1_data.astype(np.float32, copy=False)
        mn, p01, p50, p99, mx = percentiles(t1f)
        row["t1_min"], row["t1_p01"], row["t1_p50"], row["t1_p99"], row["t1_max"] = (
            f"{mn:.4f}", f"{p01:.4f}", f"{p50:.4f}", f"{p99:.4f}", f"{mx:.4f}"
        )
        row["t1_mean"] = f"{float(np.nanmean(t1f)):.4f}"
        row["t1_std"] = f"{float(np.nanstd(t1f)):.4f}"
        row["t1_nonzero_frac"] = f"{float((t1f > 0).mean()):.6f}"
        # Percentiles over the brain (non-zero) voxels: useful as the normalization
        # anchor; whole-volume p50 is background-dominated (~0).
        brain = t1f[t1f > 0]
        if brain.size:
            row["t1_brain_p50"] = f"{float(np.nanpercentile(brain, 50)):.4f}"
            row["t1_brain_p99"] = f"{float(np.nanpercentile(brain, 99)):.4f}"

        lesion_data = np.asanyarray(lesion_img.dataobj)
        row["lesion_dtype"] = str(lesion_data.dtype)
        uniq = np.unique(lesion_data)
        # Round (not truncate) before int-cast so float-encoded labels like
        # 0.9999999776482582 (R039) read as "1" not "0".
        row["lesion_unique_labels"] = ",".join(
            str(int(round(float(u)))) for u in uniq if np.isfinite(u)
        )
        lesion_bin = lesion_data > 0
        nvox = int(lesion_bin.sum())
        row["lesion_voxels"] = str(nvox)
        row["lesion_volume_ml"] = f"{nvox * (vx * vy * vz) / 1000.0:.4f}"
        if nvox > 0:
            lbl, ncc = ndi.label(lesion_bin, structure=_CC26)
            row["lesion_cc_count"] = str(int(ncc))
            if ncc > 0:
                sizes = ndi.sum(lesion_bin, lbl, range(1, ncc + 1))
                largest_vox = float(np.max(sizes))
                row["lesion_largest_cc_ml"] = f"{largest_vox * (vx*vy*vz) / 1000.0:.4f}"
        else:
            row["lesion_cc_count"] = "0"
            row["lesion_largest_cc_ml"] = "0"

        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["error"] = (row.get("error", "") + ";" + repr(exc)).strip(";")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to Training_Raw directory")
    ap.add_argument("--out-csv", required=True, help="Output CSV path")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    args = ap.parse_args()

    root = Path(args.root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    sessions = list(find_sessions(root))
    sessions = [s for i, s in enumerate(sessions) if i % args.shard_count == args.shard_index]
    print(f"[shard {args.shard_index}/{args.shard_count}] {len(sessions)} sessions",
          flush=True)

    write_header = not out_csv.exists() or out_csv.stat().st_size == 0
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        for i, (site, subject, session, t1, mask, meta) in enumerate(sessions):
            row = analyze_session(site, subject, session, t1, mask, meta)
            w.writerow(row)
            if (i + 1) % 25 == 0:
                f.flush()
                print(f"[shard {args.shard_index}] {i+1}/{len(sessions)} done", flush=True)
    print(f"[shard {args.shard_index}] complete", flush=True)


if __name__ == "__main__":
    sys.exit(main())
