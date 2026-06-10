#!/usr/bin/env python3
"""Convert nnU-Net probability NPZ outputs to native-space lesion NIfTIs."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np


PROBABILITIES_KEY = "probabilities"
MAX_MISMATCH_FRACTION = 1e-3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-dir", required=True, type=Path)
    parser.add_argument("--seg-dir", type=Path, default=None)
    parser.add_argument("--ref-dir", type=Path, default=Path("work/predictions_atlas_val"))
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--lesion-channel", type=int, default=1)
    args = parser.parse_args(argv)
    if args.seg_dir is None:
        args.seg_dir = args.npz_dir
    if args.lesion_channel < 0:
        parser.error("--lesion-channel must be >= 0")
    return args


def save_nifti_atomic(data: np.ndarray, ref: nib.spatialimages.SpatialImage, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ref.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1.0, 0.0)
    img = nib.Nifti1Image(data, ref.affine, header=header)
    tmp = path.with_name(path.name + ".tmp.nii.gz")
    if tmp.exists():
        tmp.unlink()
    nib.save(img, str(tmp))
    os.replace(tmp, path)


def load_probabilities(path: Path) -> np.ndarray:
    with np.load(path) as payload:
        if PROBABILITIES_KEY not in payload:
            keys = ", ".join(sorted(payload.files))
            raise ValueError(f"missing NPZ key {PROBABILITIES_KEY!r}; found keys: {keys}")
        return np.asarray(payload[PROBABILITIES_KEY])


def mismatch_fraction(probs: np.ndarray, seg: np.ndarray, spatial_axes: tuple[int, ...]) -> float:
    axes = (0,) + tuple(axis + 1 for axis in spatial_axes)
    permuted_probs = np.transpose(probs, axes=axes)
    pred = np.argmax(permuted_probs, axis=0)
    seg_labels = np.rint(seg).astype(pred.dtype, copy=False)
    return float(np.count_nonzero(pred != seg_labels) / pred.size)


def resolve_spatial_axes(probs: np.ndarray, seg_shape: tuple[int, ...], seg: np.ndarray) -> tuple[int, ...]:
    spatial_shape = tuple(probs.shape[1:])
    candidates = [
        axes
        for axes in itertools.permutations(range(len(spatial_shape)))
        if tuple(spatial_shape[axis] for axis in axes) == seg_shape
    ]
    if not candidates:
        raise ValueError(f"no spatial axis permutation maps probability shape {spatial_shape} to seg shape {seg_shape}")

    validated = []
    mismatch_by_axes = {}
    for axes in candidates:
        mismatch = mismatch_fraction(probs, seg, axes)
        mismatch_by_axes[axes] = mismatch
        if mismatch < MAX_MISMATCH_FRACTION:
            validated.append(axes)

    if len(validated) != 1:
        details = ", ".join(f"{axes}: {mismatch_by_axes[axes]:.6g}" for axes in candidates)
        raise ValueError(
            f"expected exactly one validated spatial axis permutation, got {len(validated)} "
            f"from {len(candidates)} shape candidate(s); mismatch fractions: {details}"
        )
    return validated[0]


def convert_case(
    npz_path: Path,
    seg_dir: Path,
    ref_dir: Path,
    out_dir: Path,
    lesion_channel: int,
) -> None:
    case_id = npz_path.stem
    seg_path = seg_dir / f"{case_id}.nii.gz"
    ref_path = ref_dir / f"{case_id}.nii.gz"
    out_path = out_dir / f"{case_id}.nii.gz"

    if not seg_path.is_file():
        raise FileNotFoundError(f"missing nnU-Net segmentation: {seg_path}")
    if not ref_path.is_file():
        raise FileNotFoundError(f"missing reference NIfTI: {ref_path}")

    probs = load_probabilities(npz_path)
    if probs.ndim < 2:
        raise ValueError(f"probabilities must be class-first with spatial axes, got shape {probs.shape}")
    if lesion_channel >= probs.shape[0]:
        raise ValueError(f"lesion channel {lesion_channel} outside probability class axis with size {probs.shape[0]}")

    seg_img = nib.load(str(seg_path))
    seg = np.asanyarray(seg_img.dataobj)
    ref = nib.load(str(ref_path))
    if seg.ndim != probs.ndim - 1:
        raise ValueError(f"seg ndim {seg.ndim} is incompatible with probability shape {probs.shape}")

    spatial_axes = resolve_spatial_axes(probs, tuple(seg.shape), seg)
    permuted_probs = np.transpose(probs, axes=(0,) + tuple(axis + 1 for axis in spatial_axes))
    lesion = np.ascontiguousarray(permuted_probs[lesion_channel], dtype=np.float32)

    if lesion.shape != tuple(ref.shape[:3]):
        raise ValueError(f"final lesion shape {lesion.shape} does not match reference shape {ref.shape[:3]}")

    save_nifti_atomic(lesion, ref, out_path)


def write_report(path: Path, n_cases: int, ok: int, failed: dict[str, str]) -> None:
    payload = {"n": n_cases, "ok": ok, "failed": failed}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    npz_paths = sorted(args.npz_dir.glob("*.npz"))
    failed: dict[str, str] = {}
    ok = 0

    for npz_path in npz_paths:
        case_id = npz_path.stem
        try:
            convert_case(npz_path, args.seg_dir, args.ref_dir, args.out_dir, args.lesion_channel)
        except Exception as exc:
            out_path = args.out_dir / f"{case_id}.nii.gz"
            if out_path.exists():
                out_path.unlink()
            failed[case_id] = str(exc)
            print(f"[fail] {case_id}: {exc}")
            continue
        ok += 1
        print(f"[ok] {case_id} -> {args.out_dir / f'{case_id}.nii.gz'}")

    write_report(args.out_dir / "_convert_report.json", len(npz_paths), ok, failed)
    print(f"[done] converted {ok}/{len(npz_paths)} case(s); failed={len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
