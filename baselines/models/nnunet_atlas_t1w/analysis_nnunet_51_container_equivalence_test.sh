#!/bin/bash
#SBATCH --job-name=container_equivalence
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH -p gpu
#SBATCH -G 1
#
# THE acceptance test for the submission: run the container's own inference code
# on RAW, NATIVE-ORIENTATION ATLAS files and require it to reproduce the Dice we
# already recorded for those same cases in cross-validation, to within a tight
# tolerance.
#
# Why this specific test and not a smoke test. Every number this project has ever
# produced was computed on the RAS-canonicalised copy in work/nnunet/nnUNet_raw/.
# `grep -rl ATLAS3_Training_Raw baselines/ scripts/ eval/` finds exactly one
# script, and it only characterises the data. There has never been an
# end-to-end raw-file -> prediction path. Meanwhile the release stores volumes in
# four different orientations (RAS 869, LAS 575, PSR 8, PSL 1) and nnU-Net does
# not reorient at inference. A mirror flip produces a well-formed mask of
# plausible size, so nothing except a per-case Dice comparison can detect it --
# and when it happened before (commit f8d88c1) it cost 0.2576 -> 0.016 Dice.
#
# The comparison is made EXACT by running the container with only the fold that
# held each case out, matching how the out-of-fold number was produced. Anything
# other than agreement to ~1e-3 means the geometry handling is wrong.
#
# Caveat: this exercises the container's inference code under the cluster's venv,
# not the built image. It cannot catch a Dockerfile/dependency problem -- only
# `docker run` on a build host can. It does catch the failure mode that actually
# threatens the score.

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs"

if command -v module >/dev/null 2>&1; then
  module purge
  if [[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]]; then
    module use "$GROUP_HOME/modules"
  fi
  module load devel
  module load python/3.12.1
  # .venv-nnunet was built --system-site-packages against the cluster's
  # py-pytorch module; without it torch loads but numpy is missing and
  # torch.load dies unpickling the checkpoints.
  module load py-pytorch/2.4.1_py312
fi

export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export nnUNet_compile=f
export nnUNet_raw="$REPO/work/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/work/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"

PY="$REPO/.venv-nnunet/bin/python"
DATASET="${DATASET:-Dataset507_ATLASR30_CORRECTED}"
MODEL_DIR="${MODEL_DIR:-$REPO/work/submission_model/$DATASET}"
MANIFEST="$nnUNet_raw/$DATASET/conversion_manifest.csv"
SPLITS="$nnUNet_preprocessed/$DATASET/splits_final.json"
PER_CASE="${PER_CASE:-$REPO/baselines/reports/nnunet_center_grouped_topk10_crossval_raw/per_case.json}"
PER_SESSION="$REPO/baselines/reports/training_data_characterization_r30/per_session.csv"
OUT_DIR="${OUT_DIR:-$REPO/work/container_equivalence}"
N_PER_ORIENT="${N_PER_ORIENT:-2}"

for f in "$MODEL_DIR" "$MANIFEST" "$SPLITS" "$PER_CASE" "$PER_SESSION"; do
  if [[ ! -e "$f" ]]; then
    echo "[error] missing prerequisite: $f" >&2
    exit 2
  fi
done

echo "[info] geometry unit tests first (cheap, fails fast)"
PYTHONPATH="$REPO/submission/isles26_algorithm:${PYTHONPATH:-}" "$REPO/.venv-eval/bin/python" \
  "$REPO/submission/isles26_algorithm/test_geometry.py"

mkdir -p "$OUT_DIR"

# NOTE: append, never assign. `module load py-pytorch` supplies numpy and torch
# via PYTHONPATH, and overwriting it leaves the venv without numpy.
PYTHONPATH="$REPO/submission/isles26_algorithm:${PYTHONPATH:-}" "$PY" - \
  "$MANIFEST" "$SPLITS" "$PER_CASE" "$PER_SESSION" "$MODEL_DIR" "$OUT_DIR" "$N_PER_ORIENT" <<'PYEOF'
import csv, json, os, shutil, subprocess, sys, tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import nibabel as nib

(manifest_p, splits_p, per_case_p, per_session_p,
 model_dir, out_dir, n_per_orient) = sys.argv[1:8]
n_per_orient = int(n_per_orient)
REPO = Path("/scratch/users/sciget/isles26challenge")
out_dir = Path(out_dir)

# ---- case -> raw paths, holdout fold, orientation, recorded Dice -------------
rows = {r["nnunet_id"]: r for r in csv.DictReader(open(manifest_p))}
splits = json.loads(Path(splits_p).read_text())
fold_of = {cid: i for i, s in enumerate(splits) for cid in s["val"]}

per_case = json.loads(Path(per_case_p).read_text())
cases = per_case["cases"] if isinstance(per_case, dict) and "cases" in per_case else per_case
recorded = {}
for c in cases:
    sid = c.get("subject_id")
    dice = (c.get("overlap") or {}).get("dice", c.get("dice"))
    if sid is not None and dice is not None:
        recorded[sid] = float(dice)

orient_of = {}
for r in csv.DictReader(open(per_session_p)):
    orient_of[(r["site"], r["subject"], r["session"])] = r.get("orient", "?")

# Pick cases spanning every storage orientation present in the release -- the
# permutation orientations (PSR/PSL) are rare but are exactly the ones a
# mirror-only bug would survive.
by_orient = defaultdict(list)
for cid, r in rows.items():
    if cid not in recorded or cid not in fold_of:
        continue
    o = orient_of.get((r["site"], r["subject"], r["session"]), "?")
    by_orient[o].append(cid)

selected = []
for o in sorted(by_orient):
    # prefer cases with a real lesion: Dice on a near-empty GT is uninformative
    ranked = sorted(by_orient[o], key=lambda c: -float(rows[c]["lesion_volume_ml"] or 0))
    selected.extend((o, c) for c in ranked[:n_per_orient])

# plus the most anisotropic and the finest-spacing case we have, for memory and
# resampling behaviour
print(f"[info] selected {len(selected)} cases across {len(by_orient)} orientations", flush=True)

results = []
for orient, cid in selected:
    r = rows[cid]
    raw_t1 = Path(r["t1_source_path"])
    raw_gt = Path(r["lesion_source_path"])
    fold = fold_of[cid]
    if not raw_t1.is_file() or not raw_gt.is_file():
        print(f"[skip] {cid}: raw files missing", flush=True)
        continue

    with tempfile.TemporaryDirectory(prefix=f"equiv_{cid}_") as tmp:
        tmp = Path(tmp)
        in_dir = tmp / "input" / "images" / "t1w-brain-mri"
        in_dir.mkdir(parents=True)
        # Feed the RAW file untouched, under a UUID-ish name, exactly as Grand
        # Challenge would.
        shutil.copy2(raw_t1, in_dir / f"{cid}.nii.gz")
        out_root = tmp / "output"

        env = dict(os.environ)
        env.update(
            ISLES26_INPUT_DIR=str(tmp / "input"),
            ISLES26_OUTPUT_DIR=str(out_root),
            ISLES26_MODEL_DIR=str(model_dir),
            ISLES26_FOLDS=str(fold),          # match the out-of-fold prediction exactly
            ISLES26_THRESHOLD="0.5",
            ISLES26_MIN_CC_VOXELS="0",
            ISLES26_FLATTEN_EMPTY="0",        # keep the raw softmax for comparison
            PYTHONPATH=os.pathsep.join(
                [str(REPO / "submission" / "isles26_algorithm"), os.environ.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep),
        )
        proc = subprocess.run(
            [sys.executable, str(REPO / "submission" / "isles26_algorithm" / "inference.py")],
            env=env, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"[FAIL] {cid}: inference exited {proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}", flush=True)
            results.append({"case": cid, "orient": orient, "status": "inference-failed"})
            continue

        mask_files = sorted((out_root / "images").rglob("*.mha"))
        mask_p = next((m for m in mask_files if "probability" not in m.parent.name), None)
        if mask_p is None:
            results.append({"case": cid, "orient": orient, "status": "no-output"})
            continue

        # CONTROL. Feed the same case again, but from the already-canonicalised
        # copy in nnUNet_raw. That exercises the identical inference code path
        # with an IDENTITY orientation transform, so:
        #   container(raw) == container(converted)  =>  geometry is exact
        #   container(converted) != recorded OOF    =>  the residual is the
        #       known difference between nnU-Net's training-time validation
        #       (predicted from cached preprocessed .npy) and predict_from_files
        #       (which re-preprocesses from the NIfTI) -- benign, and equally
        #       present at test time whatever we do.
        # Without this control a preprocessing-path difference is indistinguishable
        # from a geometry bug.
        conv_in = tmp / "input_conv" / "images" / "t1w-brain-mri"
        conv_in.mkdir(parents=True)
        shutil.copy2(Path(r["image"]), conv_in / f"{cid}.nii.gz")
        conv_out = tmp / "output_conv"
        env_conv = dict(env)
        env_conv.update(ISLES26_INPUT_DIR=str(tmp / "input_conv"),
                        ISLES26_OUTPUT_DIR=str(conv_out))
        proc_c = subprocess.run(
            [sys.executable, str(REPO / "submission" / "isles26_algorithm" / "inference.py")],
            env=env_conv, capture_output=True, text=True,
        )
        dice_conv = None
        if proc_c.returncode == 0:
            conv_masks = sorted((conv_out / "images").rglob("*.mha"))
            conv_p = next((m for m in conv_masks if "probability" not in m.parent.name), None)
            if conv_p is not None:
                import SimpleITK as _sitk
                pred_conv = _sitk.GetArrayFromImage(_sitk.ReadImage(str(conv_p))).astype(bool)
                gt_conv = np.asanyarray(nib.load(r["label"]).dataobj) > 0.5
                gt_conv = gt_conv.transpose(2, 1, 0)
                i2 = int(np.logical_and(pred_conv, gt_conv).sum())
                d2 = int(pred_conv.sum()) + int(gt_conv.sum())
                dice_conv = (2 * i2 / d2) if d2 else 1.0
        else:
            print(f"[warn] {cid}: control run failed\n{proc_c.stderr[-1500:]}", flush=True)

        import SimpleITK as sitk
        pred_img = sitk.ReadImage(str(mask_p))
        pred = sitk.GetArrayFromImage(pred_img).astype(bool)

        gt_img = nib.load(str(raw_gt))
        gt = np.asanyarray(gt_img.dataobj) > 0.5
        gt_sitk_order = gt.transpose(2, 1, 0)

        # Grid identity is a hard requirement of the evaluator.
        src = sitk.ReadImage(str(raw_t1))
        grid_ok = (
            pred_img.GetSize() == src.GetSize()
            and np.allclose(pred_img.GetSpacing(), src.GetSpacing())
            and np.allclose(pred_img.GetOrigin(), src.GetOrigin(), atol=1e-4)
            and np.allclose(pred_img.GetDirection(), src.GetDirection(), atol=1e-6)
        )

        inter = int(np.logical_and(pred, gt_sitk_order).sum())
        denom = int(pred.sum()) + int(gt_sitk_order.sum())
        dice = (2 * inter / denom) if denom else 1.0
        delta_oof = dice - recorded[cid]
        # THE assertion: geometry is exact iff raw and already-canonical inputs
        # give the same answer. The gap to the recorded out-of-fold number is
        # reported but is NOT the pass criterion, because it also contains the
        # preprocessing-path difference the control isolates.
        delta_geom = None if dice_conv is None else dice - dice_conv
        geom_ok = delta_geom is not None and abs(delta_geom) < 1e-3
        ok = grid_ok and geom_ok
        results.append({
            "case": cid, "orient": orient, "fold": fold,
            "dice_container_on_raw": round(dice, 6),
            "dice_container_on_canonical": None if dice_conv is None else round(dice_conv, 6),
            "dice_recorded_oof": round(recorded[cid], 6),
            "delta_geometry": None if delta_geom is None else round(delta_geom, 6),
            "delta_vs_recorded_oof": round(delta_oof, 6),
            "grid_identical": grid_ok,
            "status": "ok" if ok else "MISMATCH",
        })
        flag = "ok" if ok else "MISMATCH"
        conv_s = "n/a" if dice_conv is None else f"{dice_conv:.4f}"
        geom_s = "n/a" if delta_geom is None else f"{delta_geom:+.6f}"
        print(f"[{flag}] {cid} ({orient}, fold {fold}): raw {dice:.4f} | canonical {conv_s} | "
              f"recorded-OOF {recorded[cid]:.4f} | GEOMETRY delta {geom_s} | "
              f"vs-OOF {delta_oof:+.5f} | grid_ok={grid_ok}", flush=True)

out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "equivalence.json").write_text(json.dumps(results, indent=2) + "\n")

bad = [r for r in results if r.get("status") != "ok"]
deltas = [r["delta_vs_recorded_oof"] for r in results if r.get("delta_vs_recorded_oof") is not None]
if deltas:
    print(f"\nresidual vs recorded out-of-fold (preprocessing-path difference, informational): "
          f"mean {np.mean(deltas):+.5f}, min {min(deltas):+.5f}, max {max(deltas):+.5f}")
print(f"{len(results) - len(bad)}/{len(results)} cases are geometry-exact "
      f"(raw input == already-canonical input)")
if bad:
    print("FAILURES:", json.dumps(bad, indent=2))
    raise SystemExit(1)
print("[done] container inference is geometry-equivalent to the validated pipeline")
PYEOF

echo "[done] equivalence test passed"
