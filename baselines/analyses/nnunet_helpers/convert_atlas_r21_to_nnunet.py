#!/usr/bin/env python3
"""Convert ATLAS R2.1 sessions to nnU-Net v2 format and write custom splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np


CONVERSION_VERSION = "v1"

SIZE_BINS_ML: tuple[tuple[str, float, float], ...] = (
    ("vsmall_<0.1mL", 0.0, 0.1),
    ("small_0.1-1mL", 0.1, 1.0),
    ("med_1-10mL", 1.0, 10.0),
    ("large_10-50mL", 10.0, 50.0),
    ("vlarge_>=50mL", 50.0, 1e9),
)


MANIFEST_FIELDS = [
    "nnunet_id",
    "conversion_version",
    "image",
    "label",
    "t1_source_path",
    "lesion_source_path",
    "t1_source_sha256_prefix",
    "lesion_source_sha256_prefix",
    "site",
    "subject",
    "session",
    "days_post_stroke",
    "chronicity_bin",
    "lesion_volume_ml",
    "size_bin",
    "t1_path",
    "lesion_path",
]


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "na", "n/a"}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def parse_bool(value: object) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1", "1.0"}:
        return True
    if text in {"false", "f", "no", "n", "0", "0.0"}:
        return False
    return None


def chronicity_bin(row: dict[str, str]) -> str:
    days = parse_float(row.get("days_post_stroke"))
    if days is None or days <= 0:
        return "unknown"

    lt180 = parse_bool(row.get("chronicity_lt180d"))
    if lt180 is None:
        lt180 = days < 180
    return "lt180d" if lt180 else "ge180d"


def size_bin_ml(volume_ml: float | None) -> str:
    if volume_ml is None or volume_ml < 0:
        return "unknown"
    for name, lo, hi in SIZE_BINS_ML:
        if lo <= volume_ml < hi:
            return name
    return SIZE_BINS_ML[-1][0]


def resolve_path(value: str | None, repo_root: Path) -> Path | None:
    if value is None:
        return None
    text = value.strip()
    if text == "" or text.lower() in {"nan", "none", "na", "n/a"}:
        return None
    path = Path(text)
    return path if path.is_absolute() else repo_root / path


def is_nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def sha256_prefix(path: Path, n_hex: int = 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:n_hex]


def write_json_atomic(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def save_nifti_atomic(img: nib.spatialimages.SpatialImage, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.nii.gz")
    if tmp.exists():
        tmp.unlink()
    nib.save(img, str(tmp))
    if not is_nonempty(tmp):
        raise RuntimeError(f"nibabel wrote an empty file: {tmp}")
    os.replace(tmp, path)


def load_inventory(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def make_manifest_row(
    case_id: str,
    row: dict[str, str],
    image_path: Path,
    label_path: Path,
    t1_path: Path,
    lesion_path: Path,
    t1_sha_prefix: str,
    lesion_sha_prefix: str,
) -> dict[str, str]:
    lesion_volume = parse_float(row.get("lesion_volume_ml"))
    return {
        "nnunet_id": case_id,
        "conversion_version": CONVERSION_VERSION,
        "image": str(image_path),
        "label": str(label_path),
        "t1_source_path": str(t1_path),
        "lesion_source_path": str(lesion_path),
        "t1_source_sha256_prefix": t1_sha_prefix,
        "lesion_source_sha256_prefix": lesion_sha_prefix,
        "site": row.get("site", ""),
        "subject": row.get("subject", ""),
        "session": row.get("session", ""),
        "days_post_stroke": row.get("days_post_stroke", ""),
        "chronicity_bin": chronicity_bin(row),
        "lesion_volume_ml": "" if lesion_volume is None else f"{lesion_volume:.8g}",
        "size_bin": size_bin_ml(lesion_volume),
        "t1_path": str(t1_path),
        "lesion_path": str(lesion_path),
    }


def manifest_matches_sources(
    manifest_row: dict[str, str] | None,
    t1_path: Path,
    lesion_path: Path,
    t1_sha_prefix: str,
    lesion_sha_prefix: str,
) -> bool:
    if manifest_row is None:
        return False
    expected = {
        "conversion_version": CONVERSION_VERSION,
        "t1_source_path": str(t1_path),
        "lesion_source_path": str(lesion_path),
        "t1_source_sha256_prefix": t1_sha_prefix,
        "lesion_source_sha256_prefix": lesion_sha_prefix,
    }
    return all(manifest_row.get(k) == v for k, v in expected.items())


def unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def write_csv_atomic(rows: Iterable[dict[str, str]], fields: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def convert(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    dataset_dir = args.dataset_dir.resolve()
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    rows = load_inventory(args.inventory)
    existing_manifest_path = dataset_dir / "conversion_manifest.csv"
    existing_manifest = {}
    if existing_manifest_path.exists():
        existing_manifest = {
            row.get("nnunet_id", ""): row
            for row in load_inventory(existing_manifest_path)
        }

    manifest_rows: list[dict[str, str]] = []
    skip_rows: list[dict[str, str]] = []

    for row_index, row in enumerate(rows, start=1):
        case_id = f"ATLAS_{row_index:04d}"
        t1_path = resolve_path(row.get("t1_path"), repo_root)
        lesion_path = resolve_path(row.get("lesion_path"), repo_root)

        def log_skip(reason: str, detail: str = "") -> None:
            skip_rows.append({
                "nnunet_id": case_id,
                "site": row.get("site", ""),
                "subject": row.get("subject", ""),
                "session": row.get("session", ""),
                "reason": reason,
                "detail": detail,
                "t1_path": "" if t1_path is None else str(t1_path),
                "lesion_path": "" if lesion_path is None else str(lesion_path),
            })

        if t1_path is None or lesion_path is None:
            log_skip("missing_inventory_path")
            continue
        if not t1_path.exists():
            log_skip("missing_t1_file")
            continue
        if not lesion_path.exists():
            log_skip("missing_lesion_file")
            continue

        t1_sha_prefix = sha256_prefix(t1_path)
        lesion_sha_prefix = sha256_prefix(lesion_path)

        image_out = images_tr / f"{case_id}_0000.nii.gz"
        label_out = labels_tr / f"{case_id}.nii.gz"

        if is_nonempty(image_out) and is_nonempty(label_out):
            if manifest_matches_sources(
                existing_manifest.get(case_id),
                t1_path,
                lesion_path,
                t1_sha_prefix,
                lesion_sha_prefix,
            ):
                manifest_rows.append(
                    make_manifest_row(
                        case_id,
                        row,
                        image_out,
                        label_out,
                        t1_path,
                        lesion_path,
                        t1_sha_prefix,
                        lesion_sha_prefix,
                    )
                )
                continue
            print(f"[stale] {case_id}: removing stale converted outputs before reconversion")
            unlink_if_exists(image_out)
            unlink_if_exists(label_out)

        try:
            t1_img = nib.as_closest_canonical(nib.load(str(t1_path)))
            lesion_img = nib.as_closest_canonical(nib.load(str(lesion_path)))
            if len(t1_img.shape) != 3 or len(lesion_img.shape) != 3:
                log_skip("not_3d", f"T1 shape {t1_img.shape}; lesion shape {lesion_img.shape}")
                continue
            if tuple(t1_img.shape) != tuple(lesion_img.shape):
                log_skip("shape_mismatch", f"T1 shape {t1_img.shape}; lesion shape {lesion_img.shape}")
                continue
            if not np.allclose(t1_img.affine, lesion_img.affine, atol=1e-3):
                log_skip("affine_mismatch", "T1 and lesion affines differ after RAS canonicalization")
                continue

            lesion_data = (np.asanyarray(lesion_img.dataobj) > 0.5).astype(np.uint8)
            lesion_header = lesion_img.header.copy()
            lesion_header.set_data_dtype(np.uint8)
            lesion_clean = nib.Nifti1Image(lesion_data, lesion_img.affine, lesion_header)

            save_nifti_atomic(t1_img, image_out)
            save_nifti_atomic(lesion_clean, label_out)
            manifest_rows.append(
                make_manifest_row(
                    case_id,
                    row,
                    image_out,
                    label_out,
                    t1_path,
                    lesion_path,
                    t1_sha_prefix,
                    lesion_sha_prefix,
                )
            )
            print(f"[convert] {case_id}: {row.get('site', '')}/{row.get('subject', '')}/{row.get('session', '')}")
        except Exception as exc:  # noqa: BLE001 - log the session and continue converting the rest.
            log_skip(type(exc).__name__, str(exc))

    dataset_json = {
        "name": "Dataset501_ATLASR21",
        "description": "ATLAS R2.1 T1w brain-extracted stroke lesion segmentation baseline dataset",
        "channel_names": {"0": "T1w"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(manifest_rows),
        "file_ending": ".nii.gz",
    }
    write_json_atomic(dataset_json, dataset_dir / "dataset.json")
    write_csv_atomic(manifest_rows, MANIFEST_FIELDS, dataset_dir / "conversion_manifest.csv")
    write_csv_atomic(
        skip_rows,
        ["nnunet_id", "site", "subject", "session", "reason", "detail", "t1_path", "lesion_path"],
        dataset_dir / "conversion_skips.csv",
    )

    print(f"[done] converted/available cases: {len(manifest_rows)}")
    print(f"[done] skipped sessions: {len(skip_rows)}")
    print(f"[done] dataset: {dataset_dir}")
    return 0 if manifest_rows else 2


def count_field(rows: Iterable[dict[str, str]], field: str) -> dict[str, int]:
    counts = Counter((row.get(field) or "unknown") for row in rows)
    return dict(sorted(counts.items()))


def write_splits(args: argparse.Namespace) -> int:
    manifest_path = args.dataset_dir / "conversion_manifest.csv"
    if not manifest_path.exists():
        print(f"missing manifest: {manifest_path}", file=sys.stderr)
        return 2

    manifest = load_inventory(manifest_path)
    if not manifest:
        print(f"empty manifest: {manifest_path}", file=sys.stderr)
        return 2

    folds = int(args.folds)
    rng = random.Random(args.seed)
    strata: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    row_by_case = {}
    for row in manifest:
        case_id = row["nnunet_id"]
        row_by_case[case_id] = row
        strata[(row.get("site") or "unknown", row.get("chronicity_bin") or "unknown", row.get("size_bin") or "unknown")].append(case_id)

    val_by_fold: list[list[str]] = [[] for _ in range(folds)]
    strata_summary = []
    for key in sorted(strata):
        cases = sorted(strata[key])
        rng.shuffle(cases)
        offset = rng.randrange(folds)
        fold_counts = {str(i): 0 for i in range(folds)}
        for i, case_id in enumerate(cases):
            fold = (offset + i) % folds
            val_by_fold[fold].append(case_id)
            fold_counts[str(fold)] += 1
        strata_summary.append({
            "site": key[0],
            "chronicity_bin": key[1],
            "size_bin": key[2],
            "n": len(cases),
            "val_by_fold": fold_counts,
        })

    all_cases = sorted(row_by_case)
    all_case_set = set(all_cases)
    splits = []
    summary_folds = []
    for fold, val_cases_unsorted in enumerate(val_by_fold):
        val_cases = sorted(val_cases_unsorted)
        val_set = set(val_cases)
        train_cases = sorted(all_case_set - val_set)
        train_rows = [row_by_case[c] for c in train_cases]
        val_rows = [row_by_case[c] for c in val_cases]
        splits.append({"train": train_cases, "val": val_cases})
        summary_folds.append({
            "fold": fold,
            "n_train": len(train_cases),
            "n_val": len(val_cases),
            "train_by_site": count_field(train_rows, "site"),
            "val_by_site": count_field(val_rows, "site"),
            "train_by_chronicity": count_field(train_rows, "chronicity_bin"),
            "val_by_chronicity": count_field(val_rows, "chronicity_bin"),
            "train_by_size_bin": count_field(train_rows, "size_bin"),
            "val_by_size_bin": count_field(val_rows, "size_bin"),
        })

    args.preprocessed_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(splits, args.preprocessed_dir / "splits_final.json")
    write_json_atomic(
        {
            "dataset": args.dataset_dir.name,
            "seed": args.seed,
            "folds": folds,
            "num_folds": folds,
            "n_cases": len(all_cases),
            "num_total_cases": len(all_cases),
            "stratification": "site x chronicity_bin x size_bin, shuffled within stratum and round-robin assigned",
            "fold_summary": summary_folds,
            "strata": strata_summary,
        },
        args.preprocessed_dir / "splits_summary.json",
    )
    print(f"[done] wrote {args.preprocessed_dir / 'splits_final.json'}")
    print(f"[done] wrote {args.preprocessed_dir / 'splits_summary.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_convert = sub.add_parser("convert", help="convert ATLAS R2.1 to nnU-Net raw dataset format")
    p_convert.add_argument("--inventory", required=True, type=Path)
    p_convert.add_argument("--dataset-dir", required=True, type=Path)
    p_convert.add_argument("--repo-root", required=True, type=Path)
    p_convert.set_defaults(func=convert)

    p_splits = sub.add_parser("splits", help="write stratified nnU-Net splits_final.json")
    p_splits.add_argument("--dataset-dir", required=True, type=Path)
    p_splits.add_argument("--preprocessed-dir", required=True, type=Path)
    p_splits.add_argument("--folds", default=5, type=int)
    p_splits.add_argument("--seed", default=42, type=int)
    p_splits.set_defaults(func=write_splits)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
