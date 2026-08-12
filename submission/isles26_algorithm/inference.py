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

# Where the weights are. /opt/ml/model is Grand Challenge's mount point for a
# SEPARATELY uploaded model tarball; GC bind-mounts it read-only even when no
# tarball was uploaded, so it is an EMPTY directory that shadows anything the
# image baked there. We bake to /opt/app/model instead and use the mount only
# when it really carries a model, which supports both deployment styles.
#
# This is not hypothetical: submission 0b843f9e (2026-08-11) died here. The
# lookup failed inside the FastAPI lifespan, so uvicorn never bound port 4743 and
# the only symptom Grand Challenge surfaced was "health check not passed within
# 300 seconds" -- the actual FileNotFoundError was three screens up in the log.
GC_MODEL_MOUNT = Path("/opt/ml/model")
BAKED_MODEL_DIR = Path(__file__).resolve().parent / "model"


def resolve_model_dir() -> Path:
    """Explicit override, else GC's tarball mount if populated, else the image."""
    override = os.environ.get("ISLES26_MODEL_DIR", "")
    if override:
        return Path(override)
    if (GC_MODEL_MOUNT / "dataset.json").is_file():
        return GC_MODEL_MOUNT
    return BAKED_MODEL_DIR


MODEL_DIR = resolve_model_dir()

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
#
# It is rank 1 of 54 by mean rank across the five metrics on that surface. It is
# rank 16 of 54 on the DEPLOYMENT-MATCHED surface (no-TTA probabilities, fold 0),
# where the argmax is thr0.25_k50 -- and it stays anyway, deliberately:
#   - that surface is n=281 and single-fold, and the "win" is an in-sample argmax
#     over a 54-point grid worth 0.0030 Dice, which is the exact error this
#     project already retracted once (the threshold-0.54 entry in
#     docs/models/status.md was chosen the same way and did not survive);
#   - the top 20 configs there span mean rank 24.9-26.8, i.e. inside the noise
#     the table is measured with;
#   - PR-AUC is computed over the soft map and is IDENTICAL (0.7417) at every
#     threshold, so a threshold change cannot move one of the five metrics at all.
# thr0.35_k50 was the serious alternative -- rank 5 on n=1452 and rank 3 on the
# no-TTA surface, trading ~0.003 Dice for lesion-F1 -- and was rejected under the
# same rule: the gap is smaller than the noise, so do not churn the operating
# point in the last fortnight. Evidence:
# baselines/reports/official_metrics_d507_topk10/ and .../tta_ablation/.
#
# Min-size pruning is the lever nnU-Net's own postprocessing search never
# contained -- it only tests remove_all_but_largest_component, and only accepts
# it on a Dice gain, while two of the five ranked metrics depend on it directly.
THRESHOLD = float(os.environ.get("ISLES26_THRESHOLD", "0.35"))
MIN_CC_VOXELS = int(os.environ.get("ISLES26_MIN_CC_VOXELS", "20"))
MIN_CC_MM3 = float(os.environ.get("ISLES26_MIN_CC_MM3", "0"))

# TTA FOLLOWS THE DEVICE: on when CUDA is present, off on CPU. Unset means auto;
# ISLES26_DISABLE_TTA=0/1 forces it either way.
#
# This reverses the previous "TTA is off, resolved, do not restore it" default,
# and the reason is a changed premise rather than a changed mind. That decision
# rested on one stated assumption -- "Grand Challenge does not guarantee a GPU, so
# the CPU figure is the one that must fit". The algorithm can in fact be
# configured to request a T4 (16 GB), and on GPU a 5-fold ensemble WITH mirroring
# TTA runs in roughly 30 s/case against a ~10 minute budget. When the binding
# constraint disappears, so does the conclusion it forced.
#
# It stays automatic rather than simply on, because the first real submission
# (0b843f9e, 2026-08-11) was scheduled with "Requested GPU: No GPU" and reported
# `cuda available: False`. If the GPU request is ever absent, unavailable, or
# silently dropped, unconditional TTA would put 5 folds + 8 mirror passes on CPU
# at 12.9 min/case and time the job out. Degrading to no-TTA costs 0.0141 Dice;
# timing out costs the submission.
#
# Measured CPU cost per case (analysis_nnunet_60_cpu_runtime_budget.sh), against
# the guidelines' ~10 minute budget, on the largest cases, 8 cores:
#     5 folds + TTA   12.9 min   OVER BUDGET
#     5 folds, no TTA  2.0 min
#     1 fold  + TTA    2.8 min
#     1 fold,  no TTA  0.5 min
# NOTE those are 8-core figures and Grand Challenge gave us TWO (ml.r7i.large,
# 2 CPU / 16 GB / no GPU). MEASURED on GC 2026-08-12, a 325x512x512 SOOP case:
# 235.5 s for 5-fold no-TTA, against 100.1 s for the same shape on 8 cores. That
# is 2.35x, not the ~4x a core count would suggest -- the workload is
# memory-bound, not thread-starved, and CPU sat at ~97% of both threads with
# memory at 28% of 16 GB. So the CPU fallback runs at ~3.9 min against a ~10 min
# budget. Applying that same 2.35x to the mirroring table below still rules out
# every TTA subset on CPU: one axis ~9.4 min, two axes ~17-19 min, full ~30 min.
#
# The original no-TTA choice was made under a pre-registered trigger: "if the
# ablation shows TTA was worth more than ~0.005 Dice, revisit in favour of 1 fold
# + TTA".
#
# The ablation landed (analysis_nnunet_61, fold 0, n=281, same weights, TTA the
# only variable) and TTA is worth +0.0141 Dice, so the trigger fired and the
# revisit was done. It did NOT overturn the choice at the time, for a reason the
# trigger did not know: 1 fold + TTA is SLOWER than 5 folds without it. Measured
# through this very file on CPU, 8 cores (analysis_nnunet_66, two real cases):
#     ATLAS_0001   fold4+TTA 183.9s   fold4 no-TTA 32.9s   5-fold no-TTA 114.2s
#     ATLAS_1316   fold4+TTA 129.9s   fold4 no-TTA 59.4s   5-fold no-TTA 100.1s
# So the alternative costs ~60% more wall time to deliver a SINGLE-model
# prediction. To win it would need the 5-fold ensemble to be worth less than
# TTA's +0.0141 Dice, and fold-ensembling is nnU-Net's better-established effect
# -- it also cuts variance across unseen centers, which is precisely our risk
# (the test set spans 60+ centers and our honest surface is center-held-out).
# The ensemble gain cannot be measured here (every case was seen by 4 of the 5
# members -- work/nested_holdout/RESULT.md), so this is a judgement under one
# unmeasured quantity, made explicit rather than hidden.
#
# WHAT IT IS WORTH, per baselines/reports/tta_ablation/: TTA buys Dice +0.0141,
# lesion-F1 +0.0205, PR-AUC +0.0191 and AVD -0.86 mL -- four of the five ranked
# metrics, and PR-AUC is the one no threshold change can touch. Every out-of-fold
# number in this repo was measured WITH TTA, so on the GPU path 0.6283 is the
# matched forecast; on the CPU fallback it is an optimistic bound.
#
# REDUCED MIRRORING: MEASURED AND CLOSED (analysis_nnunet_68/69/70). A subset of
# allowed_mirroring_axes costs 2^k passes instead of 8, so one axis fits the
# budget. It does not buy enough -- the gain is strongly super-additive across
# axes, and no single axis recovers half of it:
#     axes        passes  min/case@5f   dDice vs no-TTA
#     (0,) L-R      2        ~4.0         +0.0041
#     (1,) P-A      2        ~4.0         +0.0046
#     (2,) I-S      2        ~4.0         +0.0023
#     (0,1)         4        ~7-8         +0.0113
#     (0,1,2)       8       12.9 (over)   +0.0141
# Note AVD -- one of the five ranked metrics -- improves ONLY under full
# mirroring (-0.856 mL, t=-3.46); every subset is slightly worse than no-TTA on
# it. The pre-registered rule (recover >= half the Dice gain, regress nothing
# else, keep >= 2x CPU headroom) rejects all four. That analysis was a search for
# a TTA subset affordable ON CPU; it is moot on the GPU path, where full
# mirroring fits and is strictly better. It still governs the CPU fallback, where
# the answer remains no-TTA.
#
# "" / unset -> auto (on iff CUDA), "1" -> force off, "0" -> force on.
_disable_tta = os.environ.get("ISLES26_DISABLE_TTA", "").strip()
DISABLE_TTA = None if _disable_tta == "" else _disable_tta == "1"

# Which spatial axes mirroring TTA may flip, when it is on at all. Unset means
# "whatever the checkpoint was trained with" (all three), i.e. no behaviour
# change. A subset costs 2^len(axes) forward passes instead of 8, which is the
# only way to have both the 5-fold ensemble and some TTA inside the CPU budget.
# Measured by analysis_nnunet_68/69 before any non-empty value ships.
_axes = os.environ.get("ISLES26_MIRROR_AXES", "").strip()
MIRROR_AXES = tuple(int(a) for a in _axes.split(",") if a.strip() != "") if _axes else None
FLATTEN_EMPTY = os.environ.get("ISLES26_FLATTEN_EMPTY", "1") == "1"
CHECKPOINT = os.environ.get("ISLES26_CHECKPOINT", "checkpoint_final.pth")

LESION_CHANNEL = 1  # dataset.json: {"background": 0, "lesion": 1}
MAX_VOXELS = 600_000_000


# --- startup ----------------------------------------------------------------


def _cgroup_cpu_quota():
    """CPUs granted by the cgroup, or None if unlimited/unreadable.

    `docker run --cpus=N` -- which is how Grand Challenge caps CPU -- sets a
    cgroup CFS quota. It does NOT change os.cpu_count(), which keeps reporting
    the host's core count. Reading the quota is the only way to see the real
    allowance from inside the container.
    """
    try:  # cgroup v2
        raw = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if raw[0] != "max":
            return max(1, int(int(raw[0]) / int(raw[1])))
        return None
    except Exception:  # noqa: BLE001 - absent or v1; fall through
        pass
    try:  # cgroup v1
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0 and period > 0:
            return max(1, int(quota / period))
    except Exception:  # noqa: BLE001
        pass
    return None


def usable_cpus() -> int:
    """How many threads torch may actually run without thrashing.

    Measured, not theoretical (analysis_nnunet_67, same harness, one variable):
    on a 5-CPU cgroup on a 24-core node, asking for os.cpu_count()=24 threads
    made one real case take 240.6 s where 5 threads took 146.9 s -- a 1.6x
    slowdown purely from oversubscription, against a ~10 minute budget, and both
    runs emitted the identical 81668-voxel mask. Prefer an explicit override,
    then the cpuset affinity mask, then the CFS quota, and only fall back to the
    host core count.
    """
    for var in ("ISLES26_NUM_THREADS", "OMP_NUM_THREADS"):
        value = os.environ.get(var, "")
        if value.isdigit() and int(value) > 0:
            return int(value)
    try:
        allowed = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        allowed = os.cpu_count() or 4
    quota = _cgroup_cpu_quota()
    return max(1, min(allowed, quota) if quota else allowed)


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
        # ~10 minute CPU budget. Give torch every core it is ALLOWED -- see
        # usable_cpus(); os.cpu_count() reports the host and oversubscribes.
        torch.set_num_threads(usable_cpus())
        print(f"[init] running on CPU with {torch.get_num_threads()} threads "
              f"(host reports {os.cpu_count()})", flush=True)
    # TTA is affordable on GPU (~30 s/case at 5 folds) and not on CPU (12.9 min at
    # 8 cores, worse at the 2 threads Grand Challenge actually gave us).
    use_tta = (device.type == "cuda") if DISABLE_TTA is None else not DISABLE_TTA
    tta_source = "auto" if DISABLE_TTA is None else "ISLES26_DISABLE_TTA"
    if use_tta and device.type == "cpu":
        print("[warn] TTA forced ON without a GPU: expect ~13 min/case at 8 cores "
              "and far worse at 2. This will likely exceed the runtime budget.",
              flush=True)
    print(f"[init] device={device} folds={FOLDS} "
          f"tta={'on' if use_tta else 'off'} ({tta_source})", flush=True)
    # Say where the weights came from, and say it BEFORE the load. When this goes
    # wrong Grand Challenge shows only a health-check timeout, so the log has to
    # be self-explanatory without a debugger.
    print(f"[init] model_dir={MODEL_DIR} (gc_mount_populated="
          f"{(GC_MODEL_MOUNT / 'dataset.json').is_file()})", flush=True)
    if not (MODEL_DIR / "dataset.json").is_file():
        raise FileNotFoundError(
            f"no nnU-Net model at {MODEL_DIR}: dataset.json is missing. "
            f"Weights are baked at {BAKED_MODEL_DIR}; {GC_MODEL_MOUNT} is Grand "
            "Challenge's read-only tarball mount and is empty unless a model was "
            "uploaded. Set ISLES26_MODEL_DIR to override.")
    print(f"[init] threshold={THRESHOLD} min_cc_voxels={MIN_CC_VOXELS} "
          f"flatten_empty={FLATTEN_EMPTY}", flush=True)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=use_tta,
        perform_everything_on_device=device.type == "cuda",
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(MODEL_DIR), use_folds=FOLDS, checkpoint_name=CHECKPOINT
    )
    if MIRROR_AXES is not None:
        # Set AFTER initialization: the call above overwrites this attribute with
        # the checkpoint's inference_allowed_mirroring_axes.
        predictor.allowed_mirroring_axes = MIRROR_AXES
    print(f"[init] mirror_axes={predictor.allowed_mirroring_axes if use_tta else None}",
          flush=True)
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
        prob, geom, stage = predict(model, image_path)

        t0 = time.time()
        mask = (prob >= THRESHOLD).astype(np.uint8)
        raw_voxels = int(mask.sum())

        # Resolve the pruning threshold once so it can be logged in both units.
        vox_mm3 = float(np.prod(geom.sitk_image.GetSpacing()))
        if MIN_CC_MM3 > 0:
            min_voxels = int(math.ceil(MIN_CC_MM3 / max(vox_mm3, 1e-9)))
        else:
            min_voxels = MIN_CC_VOXELS
        mask, n_before, n_after = min_size_filter(mask, min_voxels)

        soft = prob
        if FLATTEN_EMPTY and not mask.any():
            # Exactly constant, so the official PR-AUC scores a lesion-free
            # ground truth as 1.0 rather than 0.0.
            soft = np.zeros_like(prob)
        stage["post"] = time.time() - t0

        t0 = time.time()
        write_output(MASK_SLUG, mask, geom, np.uint8)
        write_output(PROB_SLUG, soft, geom, np.float32)
        stage["write"] = time.time() - t0

        # Everything below is diagnostics. The sanity phase is the only
        # out-of-distribution signal this project will get before the deadline
        # and each submission costs nothing, so make every case self-describing:
        # a volume in mL can be compared against a known ground truth, whereas a
        # raw voxel count cannot without knowing the spacing.
        lesion_voxels = int(mask.sum())
        print(f"[case] spacing={tuple(round(s, 4) for s in geom.sitk_image.GetSpacing())} mm "
              f"voxel={vox_mm3:.5f} mm3", flush=True)
        print(f"[case] lesion_voxels={lesion_voxels} volume={lesion_voxels * vox_mm3 / 1000.0:.2f} mL",
              flush=True)
        print(f"[case] components={n_after} (pruned {n_before - n_after} of {n_before} "
              f"below {min_voxels} vox = {min_voxels * vox_mm3:.1f} mm3); "
              f"thresholding kept {raw_voxels} vox before pruning", flush=True)
        print(f"[case] soft_map max={float(prob.max()):.4f} mean={float(prob.mean()):.6f} "
              f"frac>0.5={float((prob > 0.5).mean()):.6f} "
              f"emitted={'flattened-constant' if soft is not prob else 'raw'}", flush=True)
        # The line to diff between a CPU run and a GPU+TTA run.
        print(f"[time] read+orient {stage['read']:.1f}s | network {stage['network']:.1f}s | "
              f"invert {stage['invert']:.1f}s | post {stage['post']:.1f}s | "
              f"write {stage['write']:.1f}s | TOTAL {time.time() - started:.1f}s "
              f"[device={model.device.type} tta={'on' if getattr(model, 'use_mirroring', False) else 'off'} "
              f"axes={model.allowed_mirroring_axes if getattr(model, 'use_mirroring', False) else None} "
              f"folds={len(FOLDS)}]", flush=True)
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
    """Run nnU-Net.

    Returns (lesion probability on the caller's grid, geometry, stage timings).
    The timings exist to make the GPU-vs-CPU and TTA-on-vs-off comparison
    readable from the Grand Challenge log alone: mirroring multiplies only the
    network stage by 2^len(axes), so a total runtime that moves for any other
    reason is a different problem wearing TTA's clothes.
    """
    import nibabel as nib

    stage = {}
    t0 = time.time()
    data, geom = read_image(str(image_path))
    if int(np.prod(geom.original_shape)) > MAX_VOXELS:
        raise ValueError(f"{image_path}: {geom.original_shape} exceeds the supported size")

    # Fail before predicting, not after, if the result could not be mapped back.
    assert_roundtrip(data, geom)
    canonical, canonical_affine = to_training_orientation(data, geom)
    stage["read"] = time.time() - t0
    print(f"[case] shape={geom.original_shape} axcodes={''.join(geom.axcodes)}", flush=True)

    with tempfile.TemporaryDirectory(prefix="isles26_") as tmp:
        tmp = Path(tmp)
        in_dir, out_dir = tmp / "in", tmp / "out"
        in_dir.mkdir(); out_dir.mkdir()
        nib.save(nib.Nifti1Image(canonical.astype(np.float32), canonical_affine),
                 str(in_dir / "case_0000.nii.gz"))

        t0 = time.time()
        model.predict_from_files(
            [[str(in_dir / "case_0000.nii.gz")]],
            [str(out_dir / "case")],
            save_probabilities=True,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )
        stage["network"] = time.time() - t0

        npz_path, seg_path = out_dir / "case.npz", out_dir / "case.nii.gz"
        if not npz_path.is_file() or not seg_path.is_file():
            raise RuntimeError(f"nnU-Net produced no output for {image_path.name}")
        with np.load(npz_path) as payload:
            probs = np.asarray(payload["probabilities"])
        seg_canonical = np.asanyarray(nib.load(str(seg_path)).dataobj)

    t0 = time.time()
    lesion_prob = align_probabilities(probs, seg_canonical)
    out = from_training_orientation(lesion_prob, geom).astype(np.float32)
    stage["invert"] = time.time() - t0
    return out, geom, stage


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


def min_size_filter(mask: np.ndarray, min_voxels: int):
    """Drop connected components below `min_voxels`, 26-connectivity.

    Returns (filtered mask, components before, components kept). The counts come
    free from the labelling we already do -- counting them separately would mean
    a second ndi.label pass over an ~85 M voxel volume -- and two of the five
    ranked metrics are lesion-count sensitive, so how many components pruning
    removed is the diagnostic most worth having per case.

    26-connectivity is measured, not assumed: eval/probe_panoptica_semantics.py
    shows the official scorer treats a corner-adjacent voxel pair as ONE instance.
    Filtering with a different connectivity than the metric counts with would
    train and score against different definitions of "a lesion".
    """
    from scipy import ndimage as ndi

    lab, n = ndi.label(mask.astype(bool), structure=np.ones((3, 3, 3), dtype=np.uint8))
    if min_voxels <= 0 or n == 0:
        return mask, n, n
    counts = np.bincount(lab.ravel(), minlength=n + 1)
    keep = counts >= min_voxels
    keep[0] = False
    return keep[lab].astype(np.uint8), n, int(keep.sum())


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
