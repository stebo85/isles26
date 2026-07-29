#!/usr/bin/env python3
"""ISLES'26 submission entrypoint.

Task: native-space, already skull-stripped T1w MRI -> binary ischemic infarct
mask. The organizers' evaluator scores five metrics, and the fifth (PR-AUC) is
computed over a CONTINUOUS map, so this algorithm emits BOTH a binary mask and a
float32 probability map.

Design decisions that are load-bearing -- read before changing anything:

1. GEOMETRY. The models were trained on RAS-canonicalised volumes and nnU-Net
   does not reorient at inference (plans record transpose_forward [0,1,2]). The
   test data is native-space with at least four storage orientations. Every case
   is therefore reoriented into the training orientation and the prediction is
   mapped back with the exact inverse; see geometry.py. The round-trip is
   asserted per case because this failure mode is silent.

2. WE REUSE nnU-Net's OWN FILE PATH. The volume is written to a temporary
   `*_0000.nii.gz` and fed through `predict_from_files`, which is the identical
   code path that produced every cross-validation number in this repo. Passing
   arrays in-process would be faster but would re-derive the axis and spacing
   conventions by hand, and that is precisely the class of bug that has already
   cost this project once.

3. SOFT MAP EMISSION. PR-AUC is invariant to any strictly monotone transform, so
   calibration/temperature scaling cannot move it -- do not bother. The one
   intervention that does move it is the empty-case rule: the official
   `compute_pr_auc` returns 1.0 on a lesion-free ground truth ONLY if the soft
   map is perfectly constant, and 0.0 otherwise. So when the thresholded
   prediction is empty we emit an exactly-constant (all-zero) map. This is gated
   behind FLATTEN_SOFT_MAP_IF_EMPTY so it can be switched off if the organizers
   say it is not intended behaviour (question posted to the challenge forum).

4. ONE BAD CASE MUST NOT FAIL THE RUN. Every case is wrapped: on any exception
   we emit an all-zero mask on the correct grid and continue. An all-zero map is
   also constant, so a failed case still scores correctly on a lesion-free
   ground truth rather than crashing the whole submission.

5. THE PREDICTOR IS BUILT ONCE. Model loading dominates per-case wall clock
   (measured ~30 s/case end-to-end for 5 folds + TTA on Sherlock GPUs, of which
   only ~8 s is GPU compute).

Environment knobs (all optional, defaults are the shipped configuration):
    ISLES26_MODEL_DIR        nnU-Net trained-model folder (default /opt/ml/model)
    ISLES26_FOLDS            comma-separated folds            (default 0,1,2,3,4)
    ISLES26_THRESHOLD        binarisation threshold           (default 0.35)
    ISLES26_MIN_CC_VOXELS    min connected-component size     (default 20)
    ISLES26_MIN_CC_MM3       spacing-invariant alternative    (default 0 = use voxels)
    ISLES26_DISABLE_TTA      set to 1 to disable mirroring TTA
    ISLES26_FLATTEN_EMPTY    1/0, soft-map flattening rule    (default 1)
    ISLES26_INPUT_DIR        default /input
    ISLES26_OUTPUT_DIR       default /output
    ISLES26_MASK_SLUG        output socket dir for the mask
    ISLES26_PROB_SLUG        output socket dir for the soft map
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry import (  # noqa: E402
    InputGeometry,
    assert_roundtrip,
    from_training_orientation,
    read_image,
    to_training_orientation,
    write_like_input,
)

# --- configuration ----------------------------------------------------------

INPUT_DIR = Path(os.environ.get("ISLES26_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("ISLES26_OUTPUT_DIR", "/output"))
MODEL_DIR = Path(os.environ.get("ISLES26_MODEL_DIR", "/opt/ml/model"))

# Grand Challenge names output sockets by slug. The organizers had not published
# the interface when this was written (question posted to the challenge forum on
# 2026-07-27), so the slugs are overridable and we default to the naming ISLES
# has used in previous editions.
MASK_SLUG = os.environ.get("ISLES26_MASK_SLUG", "stroke-lesion-segmentation")
PROB_SLUG = os.environ.get("ISLES26_PROB_SLUG", "stroke-lesion-probability-map")

FOLDS = tuple(int(f) for f in os.environ.get("ISLES26_FOLDS", "0,1,2,3,4").split(",") if f != "")

# Operating point selected on the CENTER-HELD-OUT surface (Dataset507, n=1452)
# against all five official metrics -- not on Dice, and not on the center-leaking
# random-fold surface. See baselines/reports/official_metrics_d507_topk10/.
#
# Measured against the previous default (0.50, no pruning):
#     Dice            0.6268 -> 0.6283   (+0.0015)
#     |lesion count|  2.0000 -> 1.8685   (-0.1315, better)
#     lesion-F1 (RQ)  0.5491 -> 0.5791   (+0.0300, the big one)
#     AVD mL          6.2780 -> 6.3732   (+0.0952, the only regression)
#     PR-AUC          unchanged (threshold-free)
# Mean rank across all five metrics: 28.29 -> 26.55 of 54 configurations, and it
# is the rank-optimal configuration on both evaluation surfaces.
#
# The min-component filter is the lever nnU-Net's own postprocessing search never
# contained: it only ever tests remove_all_but_largest_component, and only accepts
# it on a Dice gain, so min-size pruning was never a candidate -- while two of the
# five ranked metrics depend on it directly.
THRESHOLD = float(os.environ.get("ISLES26_THRESHOLD", "0.35"))
# In VOXELS, because that is what was swept. Note only 783/1453 training sessions
# are 1 mm isotropic, so this is not a constant physical volume; ISLES26_MIN_CC_MM3
# expresses the same filter in mm^3 if a spacing-invariant rule is preferred.
MIN_CC_VOXELS = int(os.environ.get("ISLES26_MIN_CC_VOXELS", "20"))
MIN_CC_MM3 = float(os.environ.get("ISLES26_MIN_CC_MM3", "0"))
DISABLE_TTA = os.environ.get("ISLES26_DISABLE_TTA", "0") == "1"
FLATTEN_EMPTY = os.environ.get("ISLES26_FLATTEN_EMPTY", "1") == "1"
CHECKPOINT = os.environ.get("ISLES26_CHECKPOINT", "checkpoint_final.pth")

LESION_CHANNEL = 1  # dataset.json: {"background": 0, "lesion": 1}

# The volume is compressed float32 on the caller's grid; a whole-brain T1w at
# 0.5 mm is ~200 MB uncompressed, which is fine, but refuse anything absurd
# rather than OOM the node.
MAX_VOXELS = 600_000_000


# --- model ------------------------------------------------------------------


def build_predictor():
    """Initialise the nnU-Net predictor once, up front."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device} folds={FOLDS} tta={'off' if DISABLE_TTA else 'on'}", flush=True)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=not DISABLE_TTA,
        perform_everything_on_device=device.type == "cuda",
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(MODEL_DIR), use_folds=FOLDS, checkpoint_name=CHECKPOINT
    )
    return predictor


# --- postprocessing ---------------------------------------------------------


def min_size_filter(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    """Drop connected components below `min_voxels`, 26-connectivity.

    26-connectivity is not a guess: `eval/probe_panoptica_semantics.py` measures
    that the official evaluator's ConnectedComponentsInstanceApproximator treats
    a corner-adjacent voxel pair as ONE instance. Filtering with a different
    connectivity than the scorer counts with would make the two disagree about
    what a lesion is.
    """
    if min_voxels <= 0:
        return mask
    from scipy import ndimage as ndi

    lab, n = ndi.label(mask.astype(bool), structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n == 0:
        return mask
    counts = np.bincount(lab.ravel(), minlength=n + 1)
    keep = counts >= min_voxels
    keep[0] = False
    return keep[lab].astype(np.uint8)


# --- per-case ---------------------------------------------------------------


def find_input_images(input_dir: Path) -> list[Path]:
    """Locate the input volumes.

    Grand Challenge mounts each input socket at /input/images/<slug>/ and names
    the file with a UUID, so the filename must never be hard-coded. We glob for
    any supported volume under images/ and fall back to the directory root for
    local testing.
    """
    exts = ("*.mha", "*.mhd", "*.nii.gz", "*.nii", "*.nrrd")
    images_root = input_dir / "images"
    roots = [images_root] if images_root.is_dir() else [input_dir]
    found: list[Path] = []
    for root in roots:
        for ext in exts:
            found.extend(sorted(root.rglob(ext)))
    # De-duplicate while preserving order (*.nii matches inside *.nii.gz globs
    # on some filesystems).
    seen: set[str] = set()
    unique = []
    for p in found:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def read_case_metadata(input_dir: Path) -> dict:
    """Read the tabular covariates the challenge supplies (DAYS_POST_STROKE,
    CHRONICITY, CENTER).

    The shipped model is T1w-only and does not consume these -- a measured
    post-hoc analysis found no gain, and CHRONICITY is fully redundant with
    DAYS_POST_STROKE where both exist. We still parse and log them so that (a) a
    malformed metadata file cannot crash the run, and (b) if a later submission
    conditions on chronicity, the plumbing is already correct.
    """
    meta: dict = {}
    for candidate in sorted(input_dir.rglob("*.json")):
        try:
            payload = json.loads(candidate.read_text())
        except Exception:  # noqa: BLE001 - metadata must never break inference
            continue
        if isinstance(payload, dict):
            meta.update(payload)
    return meta


def predict_case(predictor, image_path: Path, workdir: Path) -> tuple[np.ndarray, InputGeometry]:
    """Run the model and return (lesion probability on the caller's grid, geometry)."""
    import nibabel as nib

    data, geom = read_image(str(image_path))
    if int(np.prod(geom.original_shape)) > MAX_VOXELS:
        raise ValueError(f"{image_path}: {geom.original_shape} exceeds the supported size")

    # Fail before predicting, not after, if we could not map the result back.
    assert_roundtrip(data, geom)

    canonical, canonical_affine = to_training_orientation(data, geom)
    print(
        f"[case] {image_path.name} shape={geom.original_shape} axcodes={''.join(geom.axcodes)}",
        flush=True,
    )

    in_dir = workdir / "nnunet_in"
    out_dir = workdir / "nnunet_out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # nnU-Net's channel-suffix convention; dataset.json declares one channel.
    nib.save(
        nib.Nifti1Image(canonical.astype(np.float32), canonical_affine),
        str(in_dir / "case_0000.nii.gz"),
    )

    predictor.predict_from_files(
        [[str(in_dir / "case_0000.nii.gz")]],
        [str(out_dir / "case")],
        save_probabilities=True,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )

    npz_path = out_dir / "case.npz"
    seg_path = out_dir / "case.nii.gz"
    if not npz_path.is_file() or not seg_path.is_file():
        raise RuntimeError(f"nnU-Net produced no output for {image_path.name}")

    with np.load(npz_path) as payload:
        probs = np.asarray(payload["probabilities"])

    seg_canonical = np.asanyarray(nib.load(str(seg_path)).dataobj)
    lesion_prob = _align_probabilities(probs, seg_canonical)

    # Back onto the caller's grid.
    return from_training_orientation(lesion_prob, geom).astype(np.float32), geom


def _align_probabilities(probs: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Extract the lesion channel from nnU-Net's probability array, resolving the
    spatial-axis permutation against the segmentation it was written with.

    nnU-Net writes probabilities as (channel, *spatial) but the spatial axis
    order has bitten this project before (see `npz_lesion_prob_to_nii.py`, and
    the ensemble helper's `resolve_spatial_axes`). Rather than assume, we check
    every permutation against the segmentation nnU-Net itself produced and
    require exactly one to agree. On a cubic volume several permutations can be
    consistent, so an ambiguous match is an error, not a coin flip.
    """
    import itertools

    if probs.ndim != seg.ndim + 1:
        raise ValueError(f"probability array {probs.shape} incompatible with seg {seg.shape}")

    candidates = []
    for perm in itertools.permutations(range(seg.ndim)):
        moved = np.transpose(probs, axes=(0,) + tuple(a + 1 for a in perm))
        if moved.shape[1:] != seg.shape:
            continue
        mismatch = float(np.mean(np.argmax(moved, axis=0) != np.rint(seg))) if seg.size else 0.0
        candidates.append((mismatch, perm, moved))

    if not candidates:
        raise ValueError(f"no axis permutation maps {probs.shape} onto {seg.shape}")

    candidates.sort(key=lambda c: c[0])
    best_mismatch, _, best = candidates[0]
    if best_mismatch > 1e-3:
        raise ValueError(
            f"probability/segmentation disagree on {best_mismatch:.4%} of voxels; "
            "refusing to emit a soft map that does not match the mask"
        )
    ties = [c for c in candidates if abs(c[0] - best_mismatch) < 1e-12]
    if len(ties) > 1 and seg.any():
        raise ValueError(
            f"{len(ties)} axis permutations are equally consistent; the soft map "
            "orientation is ambiguous"
        )
    return np.asarray(best[LESION_CHANNEL], dtype=np.float32)


# --- main -------------------------------------------------------------------


def main() -> int:
    started = time.time()
    images = find_input_images(INPUT_DIR)
    if not images:
        print(f"[error] no input volumes found under {INPUT_DIR}", file=sys.stderr)
        return 1

    metadata = read_case_metadata(INPUT_DIR)
    if metadata:
        print(f"[init] metadata keys: {sorted(metadata)}", flush=True)

    mask_dir = OUTPUT_DIR / "images" / MASK_SLUG
    prob_dir = OUTPUT_DIR / "images" / PROB_SLUG
    mask_dir.mkdir(parents=True, exist_ok=True)
    prob_dir.mkdir(parents=True, exist_ok=True)

    predictor = build_predictor()
    print(f"[init] model ready in {time.time() - started:.1f}s", flush=True)

    n_failed = 0
    for image_path in images:
        case_started = time.time()
        out_name = image_path.name
        for suffix in (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd"):
            if out_name.endswith(suffix):
                out_name = out_name[: -len(suffix)]
                break
        out_name = f"{out_name}.mha"

        geom: InputGeometry | None = None
        try:
            with tempfile.TemporaryDirectory(prefix="isles26_") as tmp:
                prob, geom = predict_case(predictor, image_path, Path(tmp))

            mask = (prob >= THRESHOLD).astype(np.uint8)
            if MIN_CC_MM3 > 0:
                vox_mm3 = float(np.prod(geom.sitk_image.GetSpacing()))
                mask = min_size_filter(mask, int(math.ceil(MIN_CC_MM3 / max(vox_mm3, 1e-9))))
            else:
                mask = min_size_filter(mask, MIN_CC_VOXELS)

            soft = prob
            if FLATTEN_EMPTY and not mask.any():
                # Exactly constant, so the official PR-AUC scores a lesion-free
                # ground truth as 1.0 instead of 0.0.
                soft = np.zeros_like(prob)

            write_like_input(mask, geom, str(mask_dir / out_name), np.uint8)
            write_like_input(soft, geom, str(prob_dir / out_name), np.float32)
            print(
                f"[ok] {image_path.name} lesion_voxels={int(mask.sum())} "
                f"in {time.time() - case_started:.1f}s",
                flush=True,
            )
        except Exception:  # noqa: BLE001 - never fail the whole submission
            n_failed += 1
            print(f"[FAILED] {image_path.name}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            try:
                if geom is None:
                    _, geom = read_image(str(image_path))
                zeros = np.zeros(geom.original_shape, dtype=np.uint8)
                write_like_input(zeros, geom, str(mask_dir / out_name), np.uint8)
                write_like_input(zeros, geom, str(prob_dir / out_name), np.float32)
                print(f"[fallback] wrote empty prediction for {image_path.name}", flush=True)
            except Exception:  # noqa: BLE001
                print(
                    f"[FATAL] could not even write a fallback for {image_path.name}\n"
                    f"{traceback.format_exc()}",
                    file=sys.stderr,
                    flush=True,
                )

    elapsed = time.time() - started
    print(
        f"[done] {len(images)} case(s), {n_failed} failed, {elapsed:.1f}s total "
        f"({elapsed / max(len(images), 1):.1f}s/case)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
