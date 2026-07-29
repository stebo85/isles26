#!/usr/bin/env python3
"""Average nnU-Net softmax predictions from one or more model families.

Two things here are subtler than they look:

* The probability NPZ can be stored in a different axis order than the NIfTI grid, and
  when two spatial dims are equal several permutations fit the shape. The permutation
  is therefore resolved ONCE per case from a member whose segmentation actually
  constrains it, and reused for members (typically all-background ones) that do not --
  their binary mask is transpose-invariant but their soft map is not, and the soft map
  is what PR-AUC scores.
* Members are averaged with optional per-member weights (``--member-weight NAME=W``,
  equal by default) so member-subset/weight searches need no code change.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import NamedTuple

import nibabel as nib
import numpy as np


PROBABILITIES_KEY = "probabilities"

# Same guard (and rationale) as npz_lesion_prob_to_nii.py: when two spatial dims are
# equal, several axis permutations match the shape and a WRONG transpose still agrees
# with nnU-Net's own segmentation on ~99.9% of voxels because background dominates.
# Only a mismatch fraction of essentially zero identifies the correct permutation, so a
# minimum above this threshold means none of the candidates is right and we must fail.
MAX_MISMATCH_FRACTION = 1e-3

IDENTITY_AXES: tuple[int, int, int] = (0, 1, 2)


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
    seg_labels = np.rint(seg).astype(pred.dtype, copy=False)
    return float(np.count_nonzero(pred != seg_labels) / pred.size)


class AxisResolution(NamedTuple):
    """Chosen probability->NIfTI axis permutation plus whether the data pinned it down.

    ``determined`` is False when the member cannot distinguish the shape-compatible
    candidates -- typically an all-background segmentation, whose label field is
    invariant under transposition. Such a member must never drive the choice, because
    the binary mask is unaffected while the SOFT map (the PR-AUC input) is not.
    """

    axes: tuple[int, int, int]
    determined: bool


def axis_candidates(probs: np.ndarray, ref_shape: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    spatial_shape = tuple(int(v) for v in probs.shape[1:])
    return [
        axes
        for axes in itertools.permutations(range(3))
        if tuple(spatial_shape[axis] for axis in axes) == ref_shape
    ]


def fallback_spatial_axes(candidates: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    """Deterministic choice when no member constrains the permutation.

    Prefer the identity so that ambiguity degrades to "leave the array as nnU-Net
    wrote it" rather than to an arbitrary shuffle of the soft map.
    """
    return IDENTITY_AXES if IDENTITY_AXES in candidates else candidates[0]


def resolve_spatial_axes(
    probs: np.ndarray,
    ref_shape: tuple[int, int, int],
    member_seg: np.ndarray | None,
    member_name: str,
    case_id: str,
) -> AxisResolution:
    candidates = axis_candidates(probs, ref_shape)
    if not candidates:
        raise ValueError(
            f"{member_name}/{case_id}: no spatial axis permutation maps probability shape "
            f"{tuple(int(v) for v in probs.shape[1:])} to NIfTI shape {ref_shape}"
        )
    if len(candidates) == 1:
        return AxisResolution(candidates[0], True)
    if member_seg is None:
        return AxisResolution(fallback_spatial_axes(candidates), False)

    mismatch_by_axes = {axes: mismatch_fraction(probs, member_seg, axes) for axes in candidates}
    ordered = sorted(candidates, key=lambda axes: mismatch_by_axes[axes])
    best = ordered[0]
    best_mismatch = mismatch_by_axes[best]
    if best_mismatch > MAX_MISMATCH_FRACTION:
        details = ", ".join(f"{axes}: {mismatch_by_axes[axes]:.6g}" for axes in ordered)
        raise ValueError(
            f"{member_name}/{case_id}: no spatial axis permutation matches the member "
            f"segmentation within {MAX_MISMATCH_FRACTION:g}; mismatch fractions: {details}"
        )
    if mismatch_by_axes[ordered[1]] <= best_mismatch:
        return AxisResolution(fallback_spatial_axes(candidates), False)
    return AxisResolution(best, True)


def probabilities_in_nifti_space(probs: np.ndarray, spatial_axes: tuple[int, int, int]) -> np.ndarray:
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


def parse_member_weight(value: str) -> tuple[str, float]:
    name, sep, raw = value.partition("=")
    if not sep or not name:
        raise argparse.ArgumentTypeError(f"--member-weight expects NAME=WEIGHT, got {value!r}")
    try:
        weight = finite_float(raw, "--member-weight")
    except (ValueError, argparse.ArgumentTypeError) as exc:
        # ArgumentTypeError is not a ValueError, so both have to be caught for the
        # offending member name to survive into the message.
        raise argparse.ArgumentTypeError(f"--member-weight {name}: {exc}") from exc
    if weight < 0.0:
        raise argparse.ArgumentTypeError(f"--member-weight {name}: weight must be >= 0")
    return name, weight


def resolve_member_weights(
    member_names: list[str],
    overrides: list[tuple[str, float]] | None,
) -> dict[str, float]:
    """Map member name -> relative weight, defaulting to a uniform average.

    Members are not interchangeable (out-of-fold Dice differs by family), so a uniform
    average is a choice, not a given. Exposing weights lets a member-subset/weight
    search be run against the official metrics without editing this file. Only ratios
    matter -- the averaging divides by the sum of the weights actually used -- so the
    returned values are raw and the normalised view is reported in the manifest.
    """
    duplicates = sorted({name for name in member_names if member_names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate --member names are ambiguous for weighting: {', '.join(duplicates)}")

    weights = {name: 1.0 for name in member_names}
    seen: set[str] = set()
    for name, weight in overrides or []:
        if name not in weights:
            raise ValueError(
                f"--member-weight {name}: no such member; members are {', '.join(member_names)}"
            )
        if name in seen:
            raise ValueError(f"--member-weight {name}: given more than once")
        seen.add(name)
        weights[name] = weight
    if sum(weights.values()) <= 0.0:
        raise ValueError("member weights must sum to a positive value")
    return weights


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


def accumulate_member(
    avg: np.ndarray | None,
    probs_nii: np.ndarray,
    weight: float,
    member_name: str,
    case_id: str,
) -> np.ndarray:
    if avg is None:
        avg = np.zeros_like(probs_nii, dtype=np.float32)
    if probs_nii.shape != avg.shape:
        raise ValueError(f"{member_name}/{case_id}: probabilities shape {probs_nii.shape} != ensemble shape {avg.shape}")
    # weight == 1.0 is the equal-weight default; adding without the multiply keeps that
    # path bit-identical to the previous unweighted implementation.
    if weight == 1.0:
        avg += probs_nii
    else:
        avg += probs_nii * weight
    return avg


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
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--member",
        nargs=2,
        action="append",
        metavar=("NAME", "PROB_DIR"),
        required=True,
        help="Model member name and nnU-Net prediction directory containing <case>.npz and <case>.nii.gz.",
    )
    parser.add_argument(
        "--member-weight",
        action="append",
        type=parse_member_weight,
        metavar="NAME=WEIGHT",
        help="Relative weight for one --member (default: equal weights); may be repeated.",
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
    try:
        weights = resolve_member_weights([name for name, _ in members], args.member_weight)
    except ValueError as exc:
        parser.error(str(exc))
    weight_total = sum(weights.values())
    normalised_weights = {name: weight / weight_total for name, weight in weights.items()}
    # A zero weight drops the member outright, so a subset search does not have to pay
    # for reading its softmax (or even have it on disk).
    active_members = [(name, path) for name, path in members if weights[name] > 0.0]
    if not active_members:
        parser.error("no member has a positive weight")

    first_dir = active_members[0][1]
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
        # Members whose own segmentation cannot distinguish the shape-compatible axis
        # permutations are held back until a member that can has been seen, so that an
        # all-background member never picks the permutation for the case.
        undetermined: list[tuple[str, np.ndarray]] = []
        case_axes: tuple[int, int, int] | None = None
        axes_source = "fallback"
        for member_name, member_dir in active_members:
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

            resolution = resolve_spatial_axes(probs, ref_shape, member_seg, member_name, case_id)
            used_members.append(member_name)
            if not resolution.determined:
                undetermined.append((member_name, probs))
                continue
            if case_axes is None:
                case_axes, axes_source = resolution.axes, member_name
            avg = accumulate_member(
                avg,
                probabilities_in_nifti_space(probs, resolution.axes),
                weights[member_name],
                member_name,
                case_id,
            )

        if not used_members:
            raise ValueError(f"{case_id}: no usable members")
        fallback_axes: tuple[int, int, int] | None = None
        for member_name, probs in undetermined:
            candidates = axis_candidates(probs, ref_shape)
            if case_axes is None:
                spatial_axes = fallback_spatial_axes(candidates)
                fallback_axes = spatial_axes
            elif case_axes in candidates:
                spatial_axes = case_axes
            else:
                raise ValueError(
                    f"{member_name}/{case_id}: probability shape {tuple(int(v) for v in probs.shape[1:])} "
                    f"is incompatible with the axis permutation {case_axes} resolved from {axes_source}"
                )
            avg = accumulate_member(
                avg,
                probabilities_in_nifti_space(probs, spatial_axes),
                weights[member_name],
                member_name,
                case_id,
            )
        if case_axes is None:
            case_axes = fallback_axes or IDENTITY_AXES
            if fallback_axes is not None:
                # stderr, not stdout: this says the saved soft map may be transposed
                # wrongly, which is exactly the kind of line that must not be lost in
                # the bulk [info] progress output.
                print(
                    f"[warn] {case_id}: no member constrains the spatial axis permutation "
                    f"(all shape-compatible transposes agree with the member segmentations); "
                    f"used {case_axes}",
                    file=sys.stderr,
                )

        used_weight_total = sum(weights[name] for name in used_members)
        if avg is None or used_weight_total <= 0.0:
            raise ValueError(f"{case_id}: no usable members")
        avg /= float(used_weight_total)

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
                "used_weights": ";".join(f"{weights[name] / used_weight_total:.6g}" for name in used_members),
                "missing_members": ";".join(missing_members),
                "n_members": len(used_members),
                # ";" like the other multi-value columns: a "," would force the csv
                # writer to quote the field and break naive awk/cut readers.
                "spatial_axes": ";".join(str(axis) for axis in case_axes),
                "axes_source": axes_source,
                "prediction": str(args.out_dir / f"{case_id}.nii.gz"),
            }
        )
        if args.progress_every > 0 and (i == 1 or i == len(cases) or i % args.progress_every == 0):
            print(f"[info] averaged {i}/{len(cases)} cases; latest={case_id} members={','.join(used_members)}")

    with (args.out_dir / "ensemble_manifest.csv").open("w", newline="") as f:
        fieldnames = [
            "case_id",
            "used_members",
            "used_weights",
            "missing_members",
            "n_members",
            "spatial_axes",
            "axes_source",
            "prediction",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    write_json(
        args.out_dir / "ensemble_manifest.json",
        {
            "n_cases": len(cases),
            "members": [
                {
                    "name": name,
                    "prob_dir": str(path),
                    "weight": weights[name],
                    "normalised_weight": normalised_weights[name],
                }
                for name, path in members
            ],
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
