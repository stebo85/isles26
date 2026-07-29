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


class AmbiguousAxesError(ValueError):
    """The segmentation cannot discriminate between candidate permutations."""


def resolve_spatial_axes(
    probs: np.ndarray,
    seg_shape: tuple[int, ...],
    seg: np.ndarray,
    fallback_axes: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    spatial_shape = tuple(probs.shape[1:])
    candidates = [
        axes
        for axes in itertools.permutations(range(len(spatial_shape)))
        if tuple(spatial_shape[axis] for axis in axes) == seg_shape
    ]
    if not candidates:
        raise ValueError(f"no spatial axis permutation maps probability shape {spatial_shape} to seg shape {seg_shape}")

    # When two spatial dims are equal, multiple permutations match the shape and a
    # WRONG transpose still agrees on ~99.9% of voxels (background dominates), so a
    # loose threshold admits several. The CORRECT permutation aligns with nnU-Net's
    # own seg, giving the (uniquely) minimal mismatch -- essentially 0. Pick the
    # argmin and require it to be both ~0 and strictly better than any runner-up.
    mismatch_by_axes = {axes: mismatch_fraction(probs, seg, axes) for axes in candidates}
    ordered = sorted(candidates, key=lambda axes: mismatch_by_axes[axes])
    best = ordered[0]
    best_mismatch = mismatch_by_axes[best]
    if best_mismatch > MAX_MISMATCH_FRACTION:
        details = ", ".join(f"{axes}: {mismatch_by_axes[axes]:.6g}" for axes in ordered)
        raise ValueError(
            f"no spatial axis permutation matches nnU-Net seg within {MAX_MISMATCH_FRACTION:g}; "
            f"mismatch fractions: {details}"
        )
    if len(ordered) > 1 and mismatch_by_axes[ordered[1]] <= best_mismatch:
        details = ", ".join(f"{axes}: {mismatch_by_axes[axes]:.6g}" for axes in ordered)
        # A tie means the segmentation cannot discriminate. Overwhelmingly this
        # is because the segmentation is ENTIRELY BACKGROUND: argmax == 0
        # everywhere agrees with seg == 0 everywhere under every permutation, so
        # the mismatch metric carries no information at all. That is different
        # from a genuinely contradictory case, and it must not be resolved by
        # picking candidates[0] -- an arbitrary transpose silently corrupts the
        # soft map, which is the input to the official PR-AUC metric.
        #
        # The caller can supply a `fallback_axes` derived from the cases in the
        # same batch whose segmentations DO discriminate; the geometry is a
        # property of how nnU-Net wrote the NPZ, so it is constant within a fold.
        if fallback_axes is not None and fallback_axes in candidates and not seg.any():
            return tuple(fallback_axes)
        reason = (
            "segmentation is entirely background, so no permutation can be "
            "distinguished" if not seg.any() else "several permutations agree equally well"
        )
        raise AmbiguousAxesError(
            f"ambiguous spatial axis permutation ({reason}); mismatch fractions: {details}"
        )
    return best


def convert_case(
    npz_path: Path,
    seg_dir: Path,
    ref_dir: Path,
    out_dir: Path,
    lesion_channel: int,
    fallback_axes: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
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

    spatial_axes = resolve_spatial_axes(probs, tuple(seg.shape), seg, fallback_axes=fallback_axes)
    permuted_probs = np.transpose(probs, axes=(0,) + tuple(axis + 1 for axis in spatial_axes))
    lesion = np.ascontiguousarray(permuted_probs[lesion_channel], dtype=np.float32)

    if lesion.shape != tuple(ref.shape[:3]):
        raise ValueError(f"final lesion shape {lesion.shape} does not match reference shape {ref.shape[:3]}")

    save_nifti_atomic(lesion, ref, out_path)
    return spatial_axes


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

    # Two passes. The first converts every case that its own segmentation can
    # disambiguate and records which permutation was chosen. The axis layout is a
    # property of how nnU-Net serialised the NPZ, so it is constant across a
    # batch; if the unambiguous cases agree unanimously, that consensus is used
    # to rescue the cases whose segmentation is entirely background (13 of 1452
    # in Dataset507). If they do NOT agree unanimously, something is wrong at a
    # level a fallback must not paper over, and every ambiguous case stays failed.
    deferred: list[Path] = []
    observed: set[tuple[int, ...]] = set()

    for npz_path in npz_paths:
        case_id = npz_path.stem
        try:
            axes = convert_case(npz_path, args.seg_dir, args.ref_dir, args.out_dir, args.lesion_channel)
        except AmbiguousAxesError:
            deferred.append(npz_path)
            continue
        except Exception as exc:
            out_path = args.out_dir / f"{case_id}.nii.gz"
            if out_path.exists():
                out_path.unlink()
            failed[case_id] = str(exc)
            print(f"[fail] {case_id}: {exc}")
            continue
        observed.add(tuple(axes))
        ok += 1
        print(f"[ok] {case_id} -> {args.out_dir / f'{case_id}.nii.gz'}")

    consensus = next(iter(observed)) if len(observed) == 1 else None
    if deferred:
        if consensus is None:
            print(
                f"[warn] {len(deferred)} case(s) are ambiguous and the unambiguous cases "
                f"do NOT agree on one permutation (saw {sorted(observed)}); refusing to guess"
            )
        else:
            print(f"[info] rescuing {len(deferred)} all-background case(s) with consensus axes {consensus}")
        for npz_path in deferred:
            case_id = npz_path.stem
            if consensus is None:
                failed[case_id] = "ambiguous axes and no unanimous consensus available"
                print(f"[fail] {case_id}: {failed[case_id]}")
                continue
            try:
                convert_case(
                    npz_path, args.seg_dir, args.ref_dir, args.out_dir, args.lesion_channel,
                    fallback_axes=consensus,
                )
            except Exception as exc:
                out_path = args.out_dir / f"{case_id}.nii.gz"
                if out_path.exists():
                    out_path.unlink()
                failed[case_id] = str(exc)
                print(f"[fail] {case_id}: {exc}")
                continue
            ok += 1
            print(f"[ok] {case_id} -> {args.out_dir / f'{case_id}.nii.gz'} (consensus axes)")

    write_report(args.out_dir / "_convert_report.json", len(npz_paths), ok, failed)
    print(f"[done] converted {ok}/{len(npz_paths)} case(s); failed={len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
