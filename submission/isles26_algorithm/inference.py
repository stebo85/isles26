"""ISLES'26 algorithm — nnU-Net v2 stroke lesion segmentation.

Grand Challenge runs this as an HTTP server (see app.py): `init_model()` is
called ONCE at startup, then `run(model)` once per case via POST /invoke.
Model loading therefore belongs in `init_model()` — the /invoke call has its own
timeout and must not be spent loading weights.

Interface (from the official template's inputs.json):

    inputs   /input/images/t1-brain-mri        slug: t1-brain-mri
             /input/stroke-metadata.json       slug: stroke-metadata
    outputs  /output/images/stroke-lesion-segmentation
             /output/images/lesion-probability-map

Both outputs are required. The evaluator ranks five metrics and the fifth,
PR-AUC, is computed over the CONTINUOUS map — a mask-only submission silently
forfeits it.

Load-bearing design decisions, read before changing anything:

1. GEOMETRY. The models were trained on RAS-canonicalised volumes and nnU-Net
   does not reorient at inference (plans record transpose_forward [0,1,2]). The
   test data is native-space; the public training release alone contains RAS,
   LAS, PSR and PSL storage orientations. Every case is reoriented into the
   training orientation and mapped back with the exact inverse (geometry.py),
   and the round-trip is asserted per case because this failure mode is SILENT:
   a mirror flip still emits a well-formed mask of plausible size. It cost this
   project 0.016 Dice against a correct 0.2576 once already.

2. WE REUSE nnU-Net's OWN FILE PATH. The volume is written to a temporary
   `*_0000.nii.gz` and fed through `predict_from_files`, the identical code path
   that produced every cross-validation number we have. Passing arrays in-process
   would be faster but would re-derive the axis and spacing conventions by hand,
   which is exactly the class of bug above.

3. SOFT MAP EMISSION. PR-AUC is invariant to any strictly monotone transform, so
   temperature scaling and Platt calibration cannot move it — do not bother. The
   one intervention that does move it is the empty-case rule: the official
   `compute_pr_auc` returns 1.0 on a lesion-free ground truth ONLY if the map is
   perfectly constant, else 0.0. So when the thresholded mask is empty we emit an
   exactly-constant map. Gated behind ISLES26_FLATTEN_EMPTY.

4. ONE BAD CASE MUST NOT FAIL THE RUN. The whole handler is wrapped: on any
   exception we emit an all-zero mask on the caller's grid and return normally.
   An all-zero map is also constant, so a failed lesion-free case still scores
   correctly rather than failing the submission.

5. METADATA IS PARSED BUT NOT USED. Measured post-hoc gain from conditioning on
   chronicity was exactly 0.0000, and CHRONICITY is redundant with
   DAYS_POST_STROKE wherever both exist. Values may be null (organizers' note),
   and the site key is CENTER here versus SITE in the training CSVs. Parsed and
   logged so a future variant has correct plumbing and a malformed file can never
   break inference.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import SimpleITK

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry import (  # noqa: E402
    assert_roundtrip,
    from_training_orientation,
    read_image,
    to_training_orientation,
)

INPUT_PATH = Path(os.environ.get("ISLES26_INPUT_DIR", "/input"))
OUTPUT_PATH = Path(os.environ.get("ISLES26_OUTPUT_DIR", "/output"))
MODEL_DIR = Path(os.environ.get("ISLES26_MODEL_DIR", "/opt/ml/model"))

# Slugs are fixed by the challenge interface; overridable only for local testing.
INPUT_IMAGE_SLUG = os.environ.get("ISLES26_INPUT_SLUG", "t1-brain-mri")
MASK_SLUG = os.environ.get("ISLES26_MASK_SLUG", "stroke-lesion-segmentation")
PROB_SLUG = os.environ.get("ISLES26_PROB_SLUG", "lesion-probability-map")

FOLDS = tuple(int(f) for f in os.environ.get("ISLES26_FOLDS", "0,1,2,3,4").split(",") if f != "")

# Operating point selected on the CENTER-HELD-OUT surface (Dataset507, n=1452)
# against all five official metrics -- not on Dice, and not on the center-leaking
# random-fold surface. Versus the naive default (0.50, no pruning):
#     Dice           0.6268 -> 0.6283   (+0.0015)
#     |lesion count| 2.0000 -> 1.8685   (-0.1315, better)
#     lesion-F1 (RQ) 0.5491 -> 0.5791   (+0.0300, the big one)
#     AVD mL         6.2780 -> 6.3732   (+0.0952, the only regression)
# Rank-optimal across all five metrics on both evaluation surfaces.
#
# Min-size pruning is the lever nnU-Net's own postprocessing search never
# contained -- it only tests remove_all_but_largest_component, and only accepts
# it on a Dice gain, while two of the five ranked metrics depend on it directly.
THRESHOLD = float(os.environ.get("ISLES26_THRESHOLD", "0.35"))
MIN_CC_VOXELS = int(os.environ.get("ISLES26_MIN_CC_VOXELS", "20"))
MIN_CC_MM3 = float(os.environ.get("ISLES26_MIN_CC_MM3", "0"))

# TTA IS OFF BY DEFAULT, AND THAT IS A RUNTIME DECISION, NOT AN ACCURACY ONE.
# Measured CPU cost per case (analysis_nnunet_60_cpu_runtime_budget.sh), against
# the guidelines' ~10 minute budget, on the largest cases:
#     5 folds + TTA   12.9 min   OVER BUDGET
#     5 folds, no TTA  2.0 min
#     1 fold  + TTA    2.8 min
#     1 fold,  no TTA  0.5 min
# Grand Challenge does not guarantee a GPU (the official template says the GPU is
# used "when enabled"), so the CPU figure is the one that must fit. Of the two
# affordable ways to get there, keeping the 5-fold ensemble and dropping TTA is
# the safer trade: ensembling is the more reliable of the two effects, and it
# leaves 5x headroom rather than 3.5x.
#
# CAVEAT, recorded so nobody is surprised: every out-of-fold number in this repo
# was measured WITH TTA, and the operating point below was tuned on TTA-smoothed
# probabilities. Shipping without TTA means the real score is slightly below the
# quoted 0.6283 Dice, and the ideal threshold may shift a little.
# analysis_nnunet_61_tta_ablation.sh measures exactly that gap.
DISABLE_TTA = os.environ.get("ISLES26_DISABLE_TTA", "1") == "1"
FLATTEN_EMPTY = os.environ.get("ISLES26_FLATTEN_EMPTY", "1") == "1"
CHECKPOINT = os.environ.get("ISLES26_CHECKPOINT", "checkpoint_final.pth")

LESION_CHANNEL = 1  # dataset.json: {"background": 0, "lesion": 1}
MAX_VOXELS = 600_000_000


# --- startup ----------------------------------------------------------------


def init_model():
    """Build the nnU-Net predictor once, before /health reports ready."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=+=" * 10, flush=True)
    print(f"[init] torch {torch.__version__} | cuda available: {torch.cuda.is_available()}", flush=True)
    if device.type == "cpu":
        # Grand Challenge does not guarantee a GPU, and the guidelines quote a
        # ~10 minute CPU budget. Give torch every core it has.
        torch.set_num_threads(os.cpu_count() or 4)
        print(f"[init] running on CPU with {torch.get_num_threads()} threads", flush=True)
    print(f"[init] device={device} folds={FOLDS} tta={'off' if DISABLE_TTA else 'on'}", flush=True)
    print(f"[init] threshold={THRESHOLD} min_cc_voxels={MIN_CC_VOXELS} "
          f"flatten_empty={FLATTEN_EMPTY}", flush=True)

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
    print(f"[init] model ready in {time.time() - started:.1f}s", flush=True)
    print("=+=" * 10, flush=True)
    return predictor


# --- dispatch ---------------------------------------------------------------


def run(model):
    """Called once per case by POST /invoke."""
    interface_key = get_interface_key()
    handler = {
        ("stroke-metadata", "t1-brain-mri"): interf0_handler,
    }[interface_key]
    return handler(model)


def get_interface_key():
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    return tuple(sorted(sv["socket"]["slug"] for sv in inputs))


# --- the case handler -------------------------------------------------------


def interf0_handler(model):
    started = time.time()
    image_dir = INPUT_PATH / "images" / INPUT_IMAGE_SLUG
    image_path = find_input_image(image_dir)
    print(f"[case] {image_path.name}", flush=True)

    metadata = load_stroke_metadata(INPUT_PATH / "stroke-metadata.json")
    print(f"[case] metadata: {json.dumps(metadata)}", flush=True)

    geom = None
    try:
        prob, geom = predict(model, image_path)

        mask = (prob >= THRESHOLD).astype(np.uint8)
        if MIN_CC_MM3 > 0:
            vox_mm3 = float(np.prod(geom.sitk_image.GetSpacing()))
            mask = min_size_filter(mask, int(math.ceil(MIN_CC_MM3 / max(vox_mm3, 1e-9))))
        else:
            mask = min_size_filter(mask, MIN_CC_VOXELS)

        soft = prob
        if FLATTEN_EMPTY and not mask.any():
            # Exactly constant, so the official PR-AUC scores a lesion-free
            # ground truth as 1.0 rather than 0.0.
            soft = np.zeros_like(prob)

        write_output(MASK_SLUG, mask, geom, np.uint8)
        write_output(PROB_SLUG, soft, geom, np.float32)
        print(f"[case] lesion_voxels={int(mask.sum())} in {time.time() - started:.1f}s", flush=True)
    except Exception:  # noqa: BLE001 - one bad case must not fail the submission
        print(f"[FAILED] {image_path.name}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        if geom is None:
            _, geom = read_image(str(image_path))
        zeros = np.zeros(geom.original_shape, dtype=np.uint8)
        write_output(MASK_SLUG, zeros, geom, np.uint8)
        write_output(PROB_SLUG, zeros, geom, np.float32)
        print("[fallback] wrote an empty prediction on the correct grid", flush=True)

    return 0


def predict(model, image_path: Path):
    """Run nnU-Net and return (lesion probability on the caller's grid, geometry)."""
    import nibabel as nib

    data, geom = read_image(str(image_path))
    if int(np.prod(geom.original_shape)) > MAX_VOXELS:
        raise ValueError(f"{image_path}: {geom.original_shape} exceeds the supported size")

    # Fail before predicting, not after, if the result could not be mapped back.
    assert_roundtrip(data, geom)
    canonical, canonical_affine = to_training_orientation(data, geom)
    print(f"[case] shape={geom.original_shape} axcodes={''.join(geom.axcodes)}", flush=True)

    with tempfile.TemporaryDirectory(prefix="isles26_") as tmp:
        tmp = Path(tmp)
        in_dir, out_dir = tmp / "in", tmp / "out"
        in_dir.mkdir(); out_dir.mkdir()
        nib.save(nib.Nifti1Image(canonical.astype(np.float32), canonical_affine),
                 str(in_dir / "case_0000.nii.gz"))

        model.predict_from_files(
            [[str(in_dir / "case_0000.nii.gz")]],
            [str(out_dir / "case")],
            save_probabilities=True,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )

        npz_path, seg_path = out_dir / "case.npz", out_dir / "case.nii.gz"
        if not npz_path.is_file() or not seg_path.is_file():
            raise RuntimeError(f"nnU-Net produced no output for {image_path.name}")
        with np.load(npz_path) as payload:
            probs = np.asarray(payload["probabilities"])
        seg_canonical = np.asanyarray(nib.load(str(seg_path)).dataobj)

    lesion_prob = align_probabilities(probs, seg_canonical)
    return from_training_orientation(lesion_prob, geom).astype(np.float32), geom


def align_probabilities(probs: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Extract the lesion channel, resolving the spatial-axis permutation against
    the segmentation nnU-Net itself produced.

    nnU-Net writes probabilities as (channel, *spatial), and the spatial order has
    bitten this project before. Rather than assume, check every permutation
    against the segmentation and require exactly one to agree; on a cubic volume
    several can be consistent, so an ambiguous match is an error, not a coin flip.
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
            "refusing to emit a soft map that does not match the mask")
    ties = [c for c in candidates if abs(c[0] - best_mismatch) < 1e-12]
    if len(ties) > 1 and seg.any():
        raise ValueError(f"{len(ties)} axis permutations are equally consistent; ambiguous")
    return np.asarray(best[LESION_CHANNEL], dtype=np.float32)


def min_size_filter(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    """Drop connected components below `min_voxels`, 26-connectivity.

    26-connectivity is measured, not assumed: eval/probe_panoptica_semantics.py
    shows the official scorer treats a corner-adjacent voxel pair as ONE instance.
    Filtering with a different connectivity than the metric counts with would
    train and score against different definitions of "a lesion".
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


# --- io ---------------------------------------------------------------------


def find_input_image(location: Path) -> Path:
    """Grand Challenge names the file with a UUID, so never hard-code it."""
    files = (glob.glob(str(location / "*.mha"))
             + glob.glob(str(location / "*.nii.gz"))
             + glob.glob(str(location / "*.nii")))
    if not files:
        raise FileNotFoundError(f"no valid image file found in {location}")
    return Path(sorted(files)[0])


def load_json_file(*, location: Path):
    with open(location) as f:
        return json.loads(f.read())


def load_stroke_metadata(location: Path) -> dict:
    """Read the tabular covariates, tolerating absence and nulls.

    Organizers' notes: the site key is CENTER here but SITE in the training CSVs,
    and any value may be null. Inference must never depend on this file being
    well-formed.
    """
    try:
        meta = load_json_file(location=location)
    except Exception:  # noqa: BLE001 - metadata must never break inference
        print(f"[warn] could not read {location}; continuing without metadata", flush=True)
        return {}
    if not isinstance(meta, dict):
        return {}
    return {k: meta.get(k) for k in ("CENTER", "CHRONICITY", "DAYS_POST_STROKE")}


def write_output(slug: str, array: np.ndarray, geom, dtype) -> None:
    """Write onto the caller's exact grid.

    Copying spacing/origin/direction from the input image is what guarantees the
    evaluator sees an identical grid; recomputing them from our own affine would
    reintroduce floating-point drift.
    """
    location = OUTPUT_PATH / "images" / slug
    location.mkdir(parents=True, exist_ok=True)
    if tuple(array.shape) != geom.original_shape:
        raise ValueError(f"refusing to write {slug}: shape {array.shape} != {geom.original_shape}")
    image = SimpleITK.GetImageFromArray(np.ascontiguousarray(array.astype(dtype).transpose(2, 1, 0)))
    image.CopyInformation(geom.sitk_image)
    SimpleITK.WriteImage(image, str(location / "output.mha"), useCompression=True)


if __name__ == "__main__":
    raise SystemExit(run(model=init_model()))
