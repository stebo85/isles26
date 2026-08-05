#!/bin/bash
#SBATCH --job-name=two_case_cpu_arms
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH -p normal
#
# What do we give up by shipping CPU + no-TTA, measured on the two named cases?
#
#   sub-r001s001 -> ATLAS_0001 (site R001)
#   sub-soop0468 -> ATLAS_1316 (site SOOP)
#
# Both are fold-4 validation cases in Dataset507's center-grouped split, so
# fold 4 is the only member that never saw them.
#
# Three arms, all run through the CONTAINER'S OWN inference.py on CPU, so the
# runtime numbers are the shipped code path, not a proxy:
#
#   fold4_tta     fold 4 only, mirroring TTA ON   -- the honest reference; this
#                 is how every out-of-fold number in this repo was produced.
#   fold4_notta   fold 4 only, mirroring TTA OFF  -- same weights, same case,
#                 TTA the only variable. fold4_tta - fold4_notta IS the loss.
#   ship5_notta   folds 0-4, TTA OFF -- the literal shipped configuration.
#                 *** CONTAMINATED ON THESE TWO CASES ***: folds 0-3 trained on
#                 both subjects, so this arm is part-memorisation and must NOT
#                 be read as an estimate of test performance. It is here only to
#                 show what the container will actually emit for these files.
#
# GPU vs CPU is not an accuracy variable -- same weights, same sliding window,
# same math to float tolerance. The only accuracy-relevant thing the CPU budget
# forced is TTA. That is what this measures.

set -euo pipefail
REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"; mkdir -p "$REPO/logs"

module purge
[[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]] && module use "$GROUP_HOME/modules"
module load devel; module load python/3.12.1; module load py-pytorch/2.4.1_py312
export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1 nnUNet_compile=f
export CUDA_VISIBLE_DEVICES=""          # force the CPU path
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export nnUNet_raw="$REPO/work/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/work/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"

DATASET=Dataset507_ATLASR30_CORRECTED
MODEL_DIR="$REPO/work/submission_model/$DATASET"
OUT="$REPO/work/qc_two_cases"
mkdir -p "$OUT"

[[ -d "$MODEL_DIR" ]] || { echo "[error] missing $MODEL_DIR" >&2; exit 2; }

"$REPO/.venv-nnunet/bin/python" - "$MODEL_DIR" "$OUT" <<'PYEOF'
import json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

model_dir, out = sys.argv[1], Path(sys.argv[2])
REPO = Path("/scratch/users/sciget/isles26challenge")
RAW = REPO / "work/nnunet/nnUNet_raw/Dataset507_ATLASR30_CORRECTED"

CASES = [("ATLAS_0001", "sub-r001s001"), ("ATLAS_1316", "sub-soop0468")]
ARMS = {                       # name -> (folds, disable_tta)
    "fold4_tta":   ("4", "0"),
    "fold4_notta": ("4", "1"),
    "ship5_notta": ("0,1,2,3,4", "1"),
}
THR, MIN_CC = 0.35, 20
STRUCT = np.ones((3, 3, 3), dtype=bool)

# Confirm the split really holds these cases out of fold 4 before claiming it.
splits = json.loads((REPO / "work/nnunet/nnUNet_preprocessed"
                     / "Dataset507_ATLASR30_CORRECTED/splits_final.json").read_text())
for cid, _ in CASES:
    assert cid in splits[4]["val"], f"{cid} is not a fold-4 validation case"
    assert cid not in splits[4]["train"], f"{cid} leaked into fold-4 train"
print("[check] both cases confirmed out-of-fold for fold 4", flush=True)


def min_size_filter(mask, k):
    lab, n = ndimage.label(mask, structure=STRUCT)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = np.zeros(sizes.shape, bool)
    keep[1:] = sizes[1:] >= k
    return keep[lab].astype(np.uint8)


def dice(a, b):
    d = a.sum() + b.sum()
    return 1.0 if d == 0 else 2.0 * np.logical_and(a, b).sum() / d


results = {}
for cid, subject in CASES:
    gt_img = nib.load(str(RAW / "labelsTr" / f"{cid}.nii.gz"))
    gt = (np.asanyarray(gt_img.dataobj) > 0.5).astype(np.uint8)
    vox_ml = float(np.prod(nib.affines.voxel_sizes(gt_img.affine))) / 1000.0
    ref = nib.load(str(RAW / "imagesTr" / f"{cid}_0000.nii.gz"))
    results[subject] = {"case": cid, "gt_ml": round(float(gt.sum() * vox_ml), 3),
                        "gt_components": int(ndimage.label(gt, structure=STRUCT)[1]),
                        "voxel_ml": round(vox_ml, 6), "arms": {}}

    for arm, (folds, no_tta) in ARMS.items():
        with tempfile.TemporaryDirectory(prefix=f"{arm}_{cid}_") as t:
            tmp = Path(t)
            ind = tmp / "in" / "images" / "t1-brain-mri"; ind.mkdir(parents=True)
            shutil.copy2(RAW / "imagesTr" / f"{cid}_0000.nii.gz", ind / f"{cid}.nii.gz")
            (tmp / "in" / "inputs.json").write_text(json.dumps(
                [{"socket": {"slug": "t1-brain-mri"}}, {"socket": {"slug": "stroke-metadata"}}]))
            (tmp / "in" / "stroke-metadata.json").write_text(json.dumps(
                {"CENTER": None, "CHRONICITY": None, "DAYS_POST_STROKE": None}))
            env = dict(os.environ)
            env.update(ISLES26_INPUT_DIR=str(tmp / "in"), ISLES26_OUTPUT_DIR=str(tmp / "out"),
                       ISLES26_MODEL_DIR=model_dir, ISLES26_FOLDS=folds,
                       ISLES26_DISABLE_TTA=no_tta, ISLES26_FLATTEN_EMPTY="0",
                       PYTHONPATH=os.pathsep.join(
                           [str(REPO / "submission" / "isles26_algorithm"),
                            os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep))
            t0 = time.time()
            p = subprocess.run(
                [sys.executable, "-c", "import inference; inference.run(inference.init_model())"],
                env=env, capture_output=True, text=True,
                cwd=str(REPO / "submission" / "isles26_algorithm"))
            wall = time.time() - t0
            if p.returncode != 0:
                print(f"[FAIL] {subject} {arm}\n{p.stderr[-2000:]}", flush=True)
                results[subject]["arms"][arm] = {"failed": True}
                continue
            pp = next(m for m in (tmp / "out" / "images").rglob("*.mha")
                      if "probability" in m.parent.name)
            prob = sitk.GetArrayFromImage(sitk.ReadImage(str(pp))
                                          ).transpose(2, 1, 0).astype(np.float32)

        nib.save(nib.Nifti1Image(prob, ref.affine, ref.header),
                 str(out / f"{subject}_prob_{arm}.nii.gz"))
        mask = min_size_filter((prob >= THR).astype(np.uint8), MIN_CC)
        nib.save(nib.Nifti1Image(mask, ref.affine, ref.header),
                 str(out / f"{subject}_mask_{arm}.nii.gz"))

        rec = {
            "folds": folds, "tta": no_tta == "0",
            "dice": round(float(dice(gt, mask)), 4),
            "pred_ml": round(float(mask.sum() * vox_ml), 3),
            "abs_volume_diff_ml": round(float(abs(mask.sum() - gt.sum()) * vox_ml), 3),
            "pred_components": int(ndimage.label(mask, structure=STRUCT)[1]),
            "wall_seconds": round(wall, 1),
            "contaminated": folds != "4",
        }
        results[subject]["arms"][arm] = rec
        print(f"[ok] {subject} {arm}: {json.dumps(rec)}", flush=True)

# Paired TTA loss, out-of-fold only.
deltas = {}
for subject, r in results.items():
    a, b = r["arms"].get("fold4_tta"), r["arms"].get("fold4_notta")
    if a and b and not a.get("failed") and not b.get("failed"):
        deltas[subject] = {
            "dice_tta": a["dice"], "dice_notta": b["dice"],
            "dice_delta_notta_minus_tta": round(b["dice"] - a["dice"], 4),
            "avd_delta_ml": round(b["abs_volume_diff_ml"] - a["abs_volume_diff_ml"], 3),
            "seconds_tta": a["wall_seconds"], "seconds_notta": b["wall_seconds"],
        }

payload = {"threshold": THR, "min_cc_voxels": MIN_CC, "device": "cpu",
           "cases": results, "tta_loss_out_of_fold": deltas}
(out / "cpu_arms_summary.json").write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
PYEOF

echo "[done] outputs in $OUT"
