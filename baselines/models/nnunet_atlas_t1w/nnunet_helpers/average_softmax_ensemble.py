#!/usr/bin/env python3
"""Average nnU-Net softmax predictions from one or more model families."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path

import nibabel as nib
import numpy as np


PROBABILITIES_KEY = "probabilities"


def case_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def load_probabilities(path: Path) -> np.ndarray:
    data = np.load(path)
    if PROBABILITIES_KEY not in data:
        raise KeyError(f"{path} does not contain key {PROBABILITIES_KEY!r}")
    probs = data[PROBABILITIES_KEY]
    if probs.ndim != 4:
        raise ValueError(f"{path}: expected class-first 4D probabilities, got shape {probs.shape}")
    if probs.shape[0] < 2:
        raise ValueError(f"{path}: expected at least two classes, got shape {probs.shape}")
    return probs.astype(np.float32, copy=False)


def mismatch_fraction(probs: np.ndarray, seg: np.ndarray, spatial_axes: tuple[int, int, int]) -> float:
    axes = (0,) + tuple(axis + 1 for axis in spatial_axes)
    pred = np.argmax(np.transpose(probs, axes=axes), axis=0)
    return float(np.mean(pred != seg))


def resolve_spatial_axes(
    probs: np.ndarray,
    ref_shape: tuple[int, int, int],
    member_seg: np.ndarray | None,
    member_name: str,
    case_id: str,
) -> tuple[int, int, int]:
    spatial_shape = tuple(int(v) for v in probs.shape[1:])
    candidates = [axes for axes in itertools.permutations(range(3)) if tuple(spatial_shape[i] for i in axes) == ref_shape]
    if not candidates:
        raise ValueError(
            f"{member_name}/{case_id}: no spatial axis permutation maps probability shape "
            f"{spatial_shape} to NIfTI shape {ref_shape}"
        )
    if len(candidates) == 1 or member_seg is None:
        return candidates[0]

    mismatch_by_axes = {axes: mismatch_fraction(probs, member_seg, axes) for axes in candidates}
    return min(mismatch_by_axes, key=mismatch_by_axes.get)


def probabilities_in_nifti_space(
    probs: np.ndarray,
    ref_shape: tuple[int, int, int],
    member_seg: np.ndarray | None,
    member_name: str,
    case_id: str,
) -> np.ndarray:
    spatial_axes = resolve_spatial_axes(probs, ref_shape, member_seg, member_name, case_id)
    axes = (0,) + tuple(axis + 1 for axis in spatial_axes)
    return np.asarray(np.transpose(probs, axes=axes), dtype=np.float32)


def finite_float(value: str, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise argparse.ArgumentTypeError(f"{name} must be finite")
    return out


def parse_member(values: list[str]) -> tuple[str, Path]:
    name, path = values
    if not name:
        raise argparse.ArgumentTypeError("member name must not be empty")
    return name, Path(path)


def collect_cases(first_dir: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return sorted(set(requested))
    return sorted(case_id_from_path(p) for p in first_dir.glob("*.npz"))


def load_member_seg(member_dir: Path, case_id: str) -> tuple[nib.Nifti1Image, np.ndarray] | tuple[None, None]:
    seg_path = member_dir / f"{case_id}.nii.gz"
    if not seg_path.is_file():
        return None, None
    img = nib.load(str(seg_path))
    seg = np.asanyarray(img.dataobj)
    while seg.ndim > 3 and seg.shape[-1] == 1:
        seg = seg[..., 0]
    return img, seg


def check_same_grid(
    member_img: nib.Nifti1Image | None,
    ref_img: nib.Nifti1Image,
    member_name: str,
    case_id: str,
    strict_affine: bool,
) -> None:
    if member_img is None:
        return
    if tuple(member_img.shape[:3]) != tuple(ref_img.shape[:3]):
        raise ValueError(f"{member_name}/{case_id}: member NIfTI shape {member_img.shape} != reference {ref_img.shape}")
    if strict_affine and not np.allclose(member_img.affine, ref_img.affine, atol=1e-3):
        raise ValueError(f"{member_name}/{case_id}: member affine differs from reference affine")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--member",
        nargs=2,
        action="append",
        metavar=("NAME", "PROB_DIR"),
        required=True,
        help="Model member name and nnU-Net prediction directory containing <case>.npz and <case>.nii.gz.",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", type=Path, help="Directory containing reference <case>.nii.gz grids.")
    parser.add_argument("--case-id", action="append", help="Restrict to one case id; may be repeated.")
    parser.add_argument("--lesion-channel", type=int, default=1)
    parser.add_argument("--threshold", type=lambda s: finite_float(s, "threshold"), default=0.5)
    parser.add_argument("--decision", choices=("threshold", "argmax"), default="threshold")
    parser.add_argument("--save-prob-nifti", action="store_true", help="Save averaged lesion probability NIfTIs.")
    parser.add_argument("--save-softmax-npz", action="store_true", help="Save averaged class probabilities as NPZ.")
    parser.add_argument("--allow-missing-members", action="store_true", help="Average available members for each case.")
    parser.add_argument("--no-strict-affine", action="store_true", help="Do not fail on member/reference affine mismatch.")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    members = [parse_member(m) for m in args.member]
    if len(members) < 1:
        parser.error("at least one --member entry is required")
    if args.threshold < 0.0 or args.threshold > 1.0:
        parser.error("--threshold must be in [0, 1]")

    first_dir = members[0][1]
    cases = collect_cases(first_dir, args.case_id)
    if not cases:
        raise SystemExit(f"no cases found in {first_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prob_dir = args.out_dir / "probabilities"
    softmax_dir = args.out_dir / "softmax_npz"
    if args.save_prob_nifti:
        prob_dir.mkdir(parents=True, exist_ok=True)
    if args.save_softmax_npz:
        softmax_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    strict_affine = not args.no_strict_affine
    for i, case_id in enumerate(cases, start=1):
        ref_dir = args.reference_dir or first_dir
        ref_path = ref_dir / f"{case_id}.nii.gz"
        if not ref_path.is_file():
            raise FileNotFoundError(f"missing reference NIfTI for {case_id}: {ref_path}")
        ref_img = nib.load(str(ref_path))
        ref_shape = tuple(int(v) for v in ref_img.shape[:3])

        avg: np.ndarray | None = None
        used_members: list[str] = []
        missing_members: list[str] = []
        for member_name, member_dir in members:
            npz_path = member_dir / f"{case_id}.npz"
            if not npz_path.is_file():
                missing_members.append(member_name)
                if args.allow_missing_members:
                    continue
                raise FileNotFoundError(f"missing softmax for {member_name}/{case_id}: {npz_path}")

            member_img, member_seg = load_member_seg(member_dir, case_id)
            check_same_grid(member_img, ref_img, member_name, case_id, strict_affine)
            probs = load_probabilities(npz_path)
            if args.lesion_channel >= probs.shape[0]:
                raise ValueError(f"{member_name}/{case_id}: lesion channel {args.lesion_channel} outside {probs.shape[0]} classes")
            probs_nii = probabilities_in_nifti_space(probs, ref_shape, member_seg, member_name, case_id)

            if avg is None:
                avg = np.zeros_like(probs_nii, dtype=np.float32)
            if probs_nii.shape != avg.shape:
                raise ValueError(f"{member_name}/{case_id}: probabilities shape {probs_nii.shape} != ensemble shape {avg.shape}")
            avg += probs_nii
            used_members.append(member_name)

        if avg is None or not used_members:
            raise ValueError(f"{case_id}: no usable members")
        avg /= float(len(used_members))

        if args.decision == "argmax":
            mask = np.asarray(np.argmax(avg, axis=0) == args.lesion_channel, dtype=np.uint8)
        else:
            mask = np.asarray(avg[args.lesion_channel] >= args.threshold, dtype=np.uint8)

        out_img = nib.Nifti1Image(mask, ref_img.affine, ref_img.header)
        out_img.set_data_dtype(np.uint8)
        nib.save(out_img, str(args.out_dir / f"{case_id}.nii.gz"))

        if args.save_prob_nifti:
            prob_img = nib.Nifti1Image(avg[args.lesion_channel].astype(np.float32), ref_img.affine, ref_img.header)
            prob_img.set_data_dtype(np.float32)
            nib.save(prob_img, str(prob_dir / f"{case_id}.nii.gz"))

        if args.save_softmax_npz:
            np.savez_compressed(softmax_dir / f"{case_id}.npz", probabilities=avg.astype(np.float32, copy=False))

        manifest_rows.append(
            {
                "case_id": case_id,
                "used_members": ";".join(used_members),
                "missing_members": ";".join(missing_members),
                "n_members": len(used_members),
                "prediction": str(args.out_dir / f"{case_id}.nii.gz"),
            }
        )
        if args.progress_every > 0 and (i == 1 or i == len(cases) or i % args.progress_every == 0):
            print(f"[info] averaged {i}/{len(cases)} cases; latest={case_id} members={','.join(used_members)}")

    with (args.out_dir / "ensemble_manifest.csv").open("w", newline="") as f:
        fieldnames = ["case_id", "used_members", "missing_members", "n_members", "prediction"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    write_json(
        args.out_dir / "ensemble_manifest.json",
        {
            "n_cases": len(cases),
            "members": [{"name": name, "prob_dir": str(path)} for name, path in members],
            "lesion_channel": args.lesion_channel,
            "decision": args.decision,
            "threshold": args.threshold,
            "save_prob_nifti": args.save_prob_nifti,
            "save_softmax_npz": args.save_softmax_npz,
            "allow_missing_members": args.allow_missing_members,
        },
    )
    print(f"[done] wrote {len(cases)} ensemble masks to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
