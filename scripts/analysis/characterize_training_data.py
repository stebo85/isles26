"""Per-session characterization of ATLAS R2.1 raw training data.

For each session, record:
  - subject_id, session_id, site, days_post_stroke, raw CHRONICITY flag, and the
    derived chronicity bin (DAYS_POST_STROKE first, CHRONICITY==1 as fallback)
  - T1w voxel sizes (mm), shape, orientation, affine determinant, data dtype, intensity stats
  - lesion mask shape match, voxel count, lesion volume (mL), connected-component count,
    largest-CC volume (mL), background match with T1, label uniqueness
  - file sizes (bytes)

Writes one CSV row per session to OUT_CSV (created if missing, appended otherwise).
This is parallelizable: each worker handles a slice of the subject list. Each shard
also drops a small chronicity audit JSON beside OUT_CSV so the DAYS_POST_STROKE vs
CHRONICITY agreement can be checked without re-reading every metadata file.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi


FIELDS = [
    "site", "subject", "session", "days_post_stroke", "chronicity_lt180d",
    "chronicity_raw", "chronicity_bin", "chronicity_source",
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
        # ATLAS site dirs are R0xx; ATLAS R3.0 adds a "SOOP" cohort dir with the
        # identical sub-*/ses-*/anat layout. Include both; skip stray files
        # (e.g. .DS_Store).
        if not site_dir.is_dir() or not (
            site_dir.name.startswith("R") or site_dir.name == "SOOP"
        ):
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


_MISSING_TOKENS = frozenset({"", "na", "n/a", "nan", "none", "null"})


def _parse_float(raw: str | None) -> float | None:
    """Parse a metadata cell to float, mapping the release's missing tokens to None.

    Case-insensitive and non-finite-rejecting so this agrees with the downstream
    readers (convert_atlas_r21_to_nnunet.parse_float, run_atlas_eval.parse_float).
    Without the isfinite guard a cell spelled "Nan" would slip past the token list
    and become a float NaN, which compares False against every threshold and would
    be silently binned as ge180d by any consumer that only tests `dps < 180`.
    """
    text = (raw or "").strip()
    if text.lower() in _MISSING_TOKENS:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def parse_metadata(path: Path):
    """Return (days_post_stroke, site, chronicity_raw, chronicity_value) from the csv.

    chronicity_raw is the verbatim CHRONICITY cell (kept so the derived bin stays
    auditable); chronicity_value is that cell as a float, or None when absent.
    """
    if path is None or not path.is_file():
        return None, None, "", None
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None, None, "", None
        r = rows[0]
        dps = _parse_float(r.get("DAYS_POST_STROKE"))
        site = (r.get("SITE") or "").strip() or None
        chron_raw = (r.get("CHRONICITY") or "").strip()
        return dps, site, chron_raw, _parse_float(chron_raw)
    except Exception:
        return None, None, "", None


def chronicity_bin(dps: float | None, chronicity: float | None):
    """Return (bin, source) for one session: bin in {lt180d, ge180d, unknown}.

    DAYS_POST_STROKE is authoritative whenever it is usable. DPS<=0 counts as
    missing (R049/R050 encode missing as 0; one R049 session is -4).

    CHRONICITY==1 means CHRONIC (>=180 days), so it can only ever rescue a case
    into ge180d. This direction was established empirically over all 1453
    per-session metadata CSVs of the ATLAS R3.0 raw release: CHRONICITY is only
    ever 1 (296 cases) or blank, and of the 26 cases carrying BOTH a usable
    DAYS_POST_STROKE and CHRONICITY==1, all 26 have DAYS_POST_STROKE >= 180.
    docs/challenge/overview.md documents the same rule. Note this contradicts
    the parenthetical "CHRONICITY (1 if < 180 days)" in docs/challenge/overview.md, which
    the data does not support; the audit counters emitted by main() will fire if
    a future release actually flips the encoding.
    """
    if dps is not None and dps > 0:
        return ("lt180d" if dps < 180 else "ge180d"), "days_post_stroke"
    if chronicity is not None and chronicity == 1:
        return "ge180d", "chronicity_flag"
    return "unknown", ""


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
    dps, meta_site, chron_raw, chron_val = parse_metadata(meta_path)
    row["days_post_stroke"] = "" if dps is None else f"{dps:g}"
    row["chronicity_raw"] = chron_raw
    cbin, csource = chronicity_bin(dps, chron_val)
    row["chronicity_bin"] = cbin
    row["chronicity_source"] = csource
    # Legacy column, kept for downstream readers: "" when the bin is unknown.
    row["chronicity_lt180d"] = "" if cbin == "unknown" else ("1" if cbin == "lt180d" else "0")
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


def new_chronicity_audit() -> dict:
    return {
        "n_sessions": 0,
        "dps_usable": 0,                 # DAYS_POST_STROKE present and > 0
        "chronicity_present": 0,         # CHRONICITY cell non-empty
        "both_present": 0,               # usable DPS *and* CHRONICITY
        "both_agree_ge180d": 0,          # CHRONICITY==1 and DPS >= 180
        "both_disagree": 0,              # CHRONICITY==1 but DPS < 180
        "rescued_by_chronicity": 0,      # no usable DPS, CHRONICITY==1 -> ge180d
        "unexpected_chronicity_values": {},  # anything other than 1 / blank
        "bins": {"lt180d": 0, "ge180d": 0, "unknown": 0},
    }


def update_chronicity_audit(audit: dict, dps: float | None, chron_raw: str,
                            chron_val: float | None, cbin: str) -> None:
    """Accumulate the DPS-vs-CHRONICITY consistency counts for one session.

    both_disagree and unexpected_chronicity_values must stay at 0 / empty. If a
    future data release flips the CHRONICITY encoding (or adds a second code),
    they go non-zero and the derived ge180d fallback in chronicity_bin() is no
    longer safe — that is the whole point of carrying these counters.
    """
    audit["n_sessions"] += 1
    audit["bins"][cbin] = audit["bins"].get(cbin, 0) + 1
    dps_usable = dps is not None and dps > 0
    if dps_usable:
        audit["dps_usable"] += 1
    if chron_raw:
        audit["chronicity_present"] += 1
    if chron_val is None:
        if chron_raw:
            # Non-empty but unparseable — still a deviation from the known encoding.
            audit["unexpected_chronicity_values"][chron_raw] = (
                audit["unexpected_chronicity_values"].get(chron_raw, 0) + 1
            )
        return
    if chron_val != 1:
        audit["unexpected_chronicity_values"][chron_raw] = (
            audit["unexpected_chronicity_values"].get(chron_raw, 0) + 1
        )
        return
    if dps_usable:
        audit["both_present"] += 1
        if dps >= 180:
            audit["both_agree_ge180d"] += 1
        else:
            audit["both_disagree"] += 1
    else:
        audit["rescued_by_chronicity"] += 1


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

    audit = new_chronicity_audit()
    write_header = not out_csv.exists() or out_csv.stat().st_size == 0
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        for i, (site, subject, session, t1, mask, meta) in enumerate(sessions):
            row = analyze_session(site, subject, session, t1, mask, meta)
            update_chronicity_audit(
                audit, _parse_float(row["days_post_stroke"]), row["chronicity_raw"],
                _parse_float(row["chronicity_raw"]), row["chronicity_bin"],
            )
            w.writerow(row)
            if (i + 1) % 25 == 0:
                f.flush()
                print(f"[shard {args.shard_index}] {i+1}/{len(sessions)} done", flush=True)

    # Per-shard sidecar (shards append to one CSV, so they must not share a file).
    audit_path = out_csv.with_suffix(f".chronicity_audit.shard{args.shard_index}.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"[shard {args.shard_index}] chronicity audit: {json.dumps(audit, sort_keys=True)}",
          flush=True)
    if audit["both_disagree"] or audit["unexpected_chronicity_values"]:
        print(f"[shard {args.shard_index}] WARNING: CHRONICITY encoding assumption "
              f"violated — chronicity_bin() fallback needs review", flush=True)
    print(f"[shard {args.shard_index}] complete", flush=True)


if __name__ == "__main__":
    sys.exit(main())
