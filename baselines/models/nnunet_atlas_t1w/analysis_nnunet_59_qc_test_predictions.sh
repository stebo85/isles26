#!/bin/bash
#SBATCH --job-name=qc_test_predictions
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH -G 1
#
# Run the SUBMISSION CONTAINER'S OWN inference code on RAW, native-space cases
# and render QC overlays, so the predictions we would actually submit can be
# looked at rather than only summarised as numbers.
#
# Honesty constraints:
#  - inputs are the RAW files under data/ATLAS3_Training_Raw/, never the
#    RAS-canonicalised training copies, so this exercises the real geometry path;
#  - each case is predicted with ONLY the fold that held it out, so nothing shown
#    here is a memorised training case;
#  - cases are sampled across the lesion-size range, not cherry-picked. Stroke
#    segmentation looks great on big confluent lesions and bad on small ones, and
#    a QC panel of large lesions would be misleading.

set -euo pipefail
REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"; mkdir -p "$REPO/logs"

module purge
[[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]] && module use "$GROUP_HOME/modules"
module load devel; module load python/3.12.1; module load py-pytorch/2.4.1_py312

export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1 nnUNet_compile=f
export nnUNet_raw="$REPO/work/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/work/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"

DATASET=Dataset507_ATLASR30_CORRECTED
MODEL_DIR="$REPO/work/submission_model/$DATASET"
VOLUME_DIR="$REPO/work/qc_test_predictions"
REPORT_DIR="$REPO/baselines/reports/qc_test_predictions"
N="${N:-9}"

PYTHONPATH="$REPO/submission/isles26_algorithm:${PYTHONPATH:-}" \
"$REPO/.venv-nnunet/bin/python" - "$MODEL_DIR" "$VOLUME_DIR" "$REPORT_DIR" "$N" <<'PYEOF'
import csv, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, nibabel as nib

model_dir = sys.argv[1]
volume_dir, report_dir = Path(sys.argv[2]), Path(sys.argv[3])
n = int(sys.argv[4])
REPO = Path("/scratch/users/sciget/isles26challenge")
RAW = REPO / "work/nnunet/nnUNet_raw/Dataset507_ATLASR30_CORRECTED"
volume_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)

man = {r["nnunet_id"]: r for r in csv.DictReader(open(RAW / "conversion_manifest.csv"))}
splits = json.loads((REPO / "work/nnunet/nnUNet_preprocessed/Dataset507_ATLASR30_CORRECTED/splits_final.json").read_text())
fold_of = {c: k for k, s in enumerate(splits) for c in s["val"]}

# Sample across the lesion-size range rather than picking winners.
cands = sorted((c for c in man if c in fold_of), key=lambda c: float(man[c]["lesion_volume_ml"] or 0))
idx = np.linspace(0, len(cands) - 1, n).astype(int)
sel = [cands[i] for i in idx]
print(f"[info] {len(sel)} cases, GT volumes "
      f"{[round(float(man[c]['lesion_volume_ml'] or 0),2) for c in sel]}", flush=True)

import SimpleITK as sitk
rows = []
for cid in sel:
    r = man[cid]; fold = fold_of[cid]
    raw_t1, raw_gt = Path(r["t1_source_path"]), Path(r["lesion_source_path"])
    with tempfile.TemporaryDirectory(prefix=f"qc_{cid}_") as t:
        tmp = Path(t)
        ind = tmp / "in" / "images" / "t1w"; ind.mkdir(parents=True)
        shutil.copy2(raw_t1, ind / f"{cid}.nii.gz")
        env = dict(os.environ)
        env.update(ISLES26_INPUT_DIR=str(tmp / "in"), ISLES26_OUTPUT_DIR=str(tmp / "out"),
                   ISLES26_MODEL_DIR=model_dir, ISLES26_FOLDS=str(fold),
                   PYTHONPATH=os.pathsep.join([str(REPO / "submission" / "isles26_algorithm"),
                                               os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep))
        p = subprocess.run([sys.executable, str(REPO / "submission/isles26_algorithm/inference.py")],
                           env=env, capture_output=True, text=True)
        if p.returncode != 0:
            print(f"[FAIL] {cid}\n{p.stderr[-1200:]}", flush=True); continue
        masks = sorted((tmp / "out" / "images").rglob("*.mha"))
        mp = next(m for m in masks if "probability" not in m.parent.name)
        pp = next(m for m in masks if "probability" in m.parent.name)
        pred = sitk.GetArrayFromImage(sitk.ReadImage(str(mp))).transpose(2, 1, 0).astype(bool)
        prob = sitk.GetArrayFromImage(sitk.ReadImage(str(pp))).transpose(2, 1, 0).astype(np.float32)

    t1_img = nib.load(str(raw_t1))
    gt_img = nib.load(str(raw_gt))
    t1 = np.asanyarray(t1_img.dataobj).astype(np.float32)
    gt = np.asanyarray(gt_img.dataobj) > 0.5
    if not (t1.shape == gt.shape == pred.shape == prob.shape):
        raise RuntimeError(
            f"{cid}: native-space shape mismatch: "
            f"t1={t1.shape}, gt={gt.shape}, pred={pred.shape}, prob={prob.shape}"
        )
    zoom = [float(z) for z in gt_img.header.get_zooms()[:3]]
    vml = float(np.prod(zoom)) / 1000.0
    inter = int(np.logical_and(pred, gt).sum()); den = int(pred.sum()) + int(gt.sum())
    dice = 2 * inter / den if den else 1.0
    rows.append(dict(case=cid, fold=fold, dice=round(dice, 4),
                     gt_ml=round(float(gt.sum()) * vml, 3),
                     pred_ml=round(float(pred.sum()) * vml, 3)))
    # Write separate native-space NIfTIs so each image can be opened or used as
    # an overlay in standard neuroimaging viewers. These regenerable volumes are
    # gitignored; the compact panel and summary remain reviewable Git artifacts.
    shutil.copy2(raw_t1, volume_dir / f"{cid}_t1w.nii.gz")
    shutil.copy2(raw_gt, volume_dir / f"{cid}_ground_truth.nii.gz")

    def save_like_t1(data, suffix, dtype):
        header = t1_img.header.copy()
        header.set_data_dtype(dtype)
        image = nib.Nifti1Image(data.astype(dtype, copy=False), t1_img.affine, header=header)
        nib.save(image, str(volume_dir / f"{cid}_{suffix}.nii.gz"))

    save_like_t1(pred, "prediction", np.uint8)
    save_like_t1(prob, "probability", np.float32)
    print(f"[ok] {cid} fold{fold} Dice {dice:.4f} pred {rows[-1]['pred_ml']} mL "
          f"gt {rows[-1]['gt_ml']} mL", flush=True)

(report_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print("\n[done] native-space NIfTIs written; render with eval/render_qc_predictions.py")
PYEOF

# Render the panels with the eval venv (matplotlib lives there, not in .venv-nnunet)
module purge
[[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]] && module use "$GROUP_HOME/modules"
module load devel; module load python/3.12.1
export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
"$REPO/.venv-eval/bin/python" "$REPO/eval/render_qc_predictions.py" \
  --in-dir "$VOLUME_DIR" --report-dir "$REPORT_DIR"
echo "[done] volumes: $VOLUME_DIR"
echo "[done] report:  $REPORT_DIR"
