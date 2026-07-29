#!/bin/bash
#SBATCH --job-name=deployment_gap
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH -G 1
#
# Diagnose the -0.017 mean Dice gap between what nnU-Net recorded during
# training-time validation and what the deployment path produces for the same
# case, same fold, same weights (measured in analysis_nnunet_51).
#
# What is already excluded:
#   - geometry: the raw-vs-canonical control in step 51 gives delta 0.000000
#   - predictor settings: nnUNetTrainer.perform_actual_validation builds
#     nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
#     perform_everything_on_device=True) -- identical to ours (verified in source)
#
# What remains: training-time validation feeds the CACHED PREPROCESSED arrays
# (self.dataset_class(self.preprocessed_dataset_folder, ...)), while
# predict_from_files re-preprocesses from the NIfTI. This job measures the
# resulting disagreement directly, prediction-vs-prediction, so the size and the
# character of the difference are pinned rather than argued.
#
# If the two predictions are near-identical, the -0.017 is export/resampling
# noise and can be ignored. If they differ structurally, the CV numbers are not
# an estimate of deployed performance and every historical comparison needs the
# caveat.

set -euo pipefail
REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"; mkdir -p "$REPO/logs"

if command -v module >/dev/null 2>&1; then
  module purge
  [[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]] && module use "$GROUP_HOME/modules"
  module load devel; module load python/3.12.1; module load py-pytorch/2.4.1_py312
fi
export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1 nnUNet_compile=f
export nnUNet_raw="$REPO/work/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/work/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"

DATASET=Dataset507_ATLASR30_CORRECTED
TRAINER=nnUNetTrainerDiceTopK10Loss
RESULT_DIR="$nnUNet_results/$DATASET/${TRAINER}__nnUNetPlans__3d_fullres"
N_CASES="${N_CASES:-40}"

PYTHONPATH="$REPO/submission/isles26_algorithm:${PYTHONPATH:-}" \
"$REPO/.venv-nnunet/bin/python" - "$RESULT_DIR" "$nnUNet_raw/$DATASET" \
  "$REPO/work/submission_model/$DATASET" "$N_CASES" <<'PYEOF'
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import nibabel as nib

result_dir, raw_dir, model_dir, n_cases = sys.argv[1:5]
result_dir, raw_dir = Path(result_dir), Path(raw_dir)
n_cases = int(n_cases)
REPO = Path("/scratch/users/sciget/isles26challenge")

splits = json.loads((Path(os.environ["nnUNet_preprocessed"]) / raw_dir.name / "splits_final.json").read_text())
per_fold = max(1, n_cases // len(splits))
forced = [c for c in os.environ.get("CASES", "").split(",") if c]
if forced:
    fold_of = {c: k for k, s in enumerate(splits) for c in s["val"]}
    sel = [(fold_of[c], c) for c in forced if c in fold_of]
else:
    sel = [(k, c) for k, s in enumerate(splits) for c in sorted(s["val"])[:per_fold]]
print(f"[info] {len(sel)} cases", flush=True)

def dice(a, b):
    d = int(a.sum()) + int(b.sum())
    return 1.0 if d == 0 else 2 * int(np.logical_and(a, b).sum()) / d

rows = []
for fold, cid in sel:
    stored_p = result_dir / f"fold_{fold}" / "validation" / f"{cid}.nii.gz"
    img_p = raw_dir / "imagesTr" / f"{cid}_0000.nii.gz"
    gt_p = raw_dir / "labelsTr" / f"{cid}.nii.gz"
    if not stored_p.is_file():
        continue
    with tempfile.TemporaryDirectory(prefix=f"gap_{cid}_") as t:
        tmp = Path(t)
        ind = tmp / "in" / "images" / "t1w"; ind.mkdir(parents=True)
        shutil.copy2(img_p, ind / f"{cid}.nii.gz")
        env = dict(os.environ)
        env.update(ISLES26_INPUT_DIR=str(tmp / "in"), ISLES26_OUTPUT_DIR=str(tmp / "out"),
                   ISLES26_MODEL_DIR=str(model_dir), ISLES26_FOLDS=str(fold),
                   ISLES26_THRESHOLD="0.5", ISLES26_MIN_CC_VOXELS="0",
                   ISLES26_FLATTEN_EMPTY="0",
                   PYTHONPATH=os.pathsep.join([str(REPO / "submission" / "isles26_algorithm"),
                                               os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep))
        p = subprocess.run([sys.executable, str(REPO / "submission" / "isles26_algorithm" / "inference.py")],
                           env=env, capture_output=True, text=True)
        if p.returncode != 0:
            print(f"[warn] {cid}: {p.stderr[-800:]}", flush=True); continue
        import SimpleITK as sitk
        masks = sorted((tmp / "out" / "images").rglob("*.mha"))
        mp = next((m for m in masks if "probability" not in m.parent.name), None)
        deployed = sitk.GetArrayFromImage(sitk.ReadImage(str(mp))).transpose(2, 1, 0).astype(bool)

    stored = np.asanyarray(nib.load(str(stored_p)).dataobj) > 0.5
    gt = np.asanyarray(nib.load(str(gt_p)).dataobj) > 0.5
    zooms = [float(z) for z in nib.load(str(gt_p)).header.get_zooms()[:3]]
    vox_ml = float(np.prod(zooms)) / 1000.0

    rows.append({
        "case": cid, "fold": fold,
        "dice_stored_vs_gt": round(dice(stored, gt), 6),
        "dice_deployed_vs_gt": round(dice(deployed, gt), 6),
        "dice_stored_vs_deployed": round(dice(stored, deployed), 6),
        "vol_stored_ml": round(float(stored.sum()) * vox_ml, 4),
        "vol_deployed_ml": round(float(deployed.sum()) * vox_ml, 4),
        "vol_gt_ml": round(float(gt.sum()) * vox_ml, 4),
    })
    r = rows[-1]
    print(f"[{cid}] stored {r['dice_stored_vs_gt']:.4f} | deployed {r['dice_deployed_vs_gt']:.4f} | "
          f"agree {r['dice_stored_vs_deployed']:.4f} | vol {r['vol_stored_ml']:.2f} -> "
          f"{r['vol_deployed_ml']:.2f} (gt {r['vol_gt_ml']:.2f})", flush=True)

out = REPO / "work" / "deployment_gap"
out.mkdir(parents=True, exist_ok=True)
(out / "per_case.json").write_text(json.dumps(rows, indent=2) + "\n")

a = np.array([r["dice_stored_vs_gt"] for r in rows])
b = np.array([r["dice_deployed_vs_gt"] for r in rows])
g = np.array([r["dice_stored_vs_deployed"] for r in rows])
vs = np.array([r["vol_stored_ml"] for r in rows]); vd = np.array([r["vol_deployed_ml"] for r in rows])
print(f"\nn={len(rows)}")
print(f"  stored-vs-GT   Dice mean {a.mean():.4f}")
print(f"  deployed-vs-GT Dice mean {b.mean():.4f}   (delta {b.mean()-a.mean():+.4f})")
print(f"  stored-vs-deployed AGREEMENT Dice mean {g.mean():.4f}  min {g.min():.4f}")
print(f"  predicted volume: stored {vs.mean():.3f} mL -> deployed {vd.mean():.3f} mL "
      f"({100*(vd.mean()-vs.mean())/max(vs.mean(),1e-9):+.1f}%)")
print("\nInterpretation: agreement >~0.98 => numerical/resampling noise;")
print("                agreement <~0.95 => the two paths genuinely disagree.")
PYEOF
