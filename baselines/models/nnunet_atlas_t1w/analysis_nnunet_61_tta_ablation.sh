#!/bin/bash
#SBATCH --job-name=tta_ablation
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH -G 1
#
# How much does mirroring TTA actually buy?
#
# This became a live question, not a curiosity: the CPU runtime measurement
# (analysis_nnunet_60) shows the shipped 5-fold + TTA configuration takes
# 12.9 min/case on CPU against a ~10 min guideline budget. Everything else fits:
#     5 folds, no TTA   2.0 min
#     1 fold,  TTA      2.8 min
#     1 fold,  no TTA    0.5 min
# So we must drop either TTA or folds, and that is an accuracy trade.
#
# The 5-fold ensembling gain cannot be measured cleanly (every case was seen by
# 4 of the 5 members -- see work/nested_holdout/RESULT.md). The TTA gain CAN:
# predict each fold-0 validation case with fold 0 only, TTA off, and compare
# against the existing fold-0 out-of-fold numbers, which were produced with TTA on.
# Same weights, same cases, one variable.
#
# Decision rule, stated before the run: if TTA is worth less than +0.005 Dice,
# ship 5 folds without TTA (ensembling is the more reliable of the two effects
# and 5-fold-no-TTA is the cheaper configuration). Otherwise reconsider.

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
OUT="$REPO/work/official_eval/_probs_noTTA_fold0"
mkdir -p "$OUT"

PYTHONPATH="$REPO/submission/isles26_algorithm:${PYTHONPATH:-}" \
"$REPO/.venv-nnunet/bin/python" - "$MODEL_DIR" "$OUT" <<'PYEOF'
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, nibabel as nib, SimpleITK as sitk

model_dir, out = sys.argv[1], Path(sys.argv[2])
REPO = Path("/scratch/users/sciget/isles26challenge")
RAW = REPO / "work/nnunet/nnUNet_raw/Dataset507_ATLASR30_CORRECTED"
splits = json.loads((REPO / "work/nnunet/nnUNet_preprocessed/Dataset507_ATLASR30_CORRECTED/splits_final.json").read_text())
cases = sorted(splits[0]["val"])
print(f"[info] {len(cases)} fold-0 validation cases, TTA OFF, fold 0 only", flush=True)

for i, cid in enumerate(cases, 1):
    target = out / f"{cid}.nii.gz"
    if target.exists():
        continue
    with tempfile.TemporaryDirectory(prefix=f"tta_{cid}_") as t:
        tmp = Path(t)
        ind = tmp / "in" / "images" / "t1-brain-mri"; ind.mkdir(parents=True)
        shutil.copy2(RAW / "imagesTr" / f"{cid}_0000.nii.gz", ind / f"{cid}.nii.gz")
        (tmp / "in" / "inputs.json").write_text(json.dumps(
            [{"socket": {"slug": "t1-brain-mri"}}, {"socket": {"slug": "stroke-metadata"}}]))
        (tmp / "in" / "stroke-metadata.json").write_text(json.dumps(
            {"CENTER": None, "CHRONICITY": None, "DAYS_POST_STROKE": None}))
        env = dict(os.environ)
        env.update(ISLES26_INPUT_DIR=str(tmp / "in"), ISLES26_OUTPUT_DIR=str(tmp / "out"),
                   ISLES26_MODEL_DIR=model_dir, ISLES26_FOLDS="0",
                   ISLES26_DISABLE_TTA="1", ISLES26_FLATTEN_EMPTY="0",
                   PYTHONPATH=os.pathsep.join([str(REPO / "submission" / "isles26_algorithm"),
                                               os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep))
        p = subprocess.run([sys.executable, "-c",
                            "import inference; inference.run(inference.init_model())"],
                           env=env, capture_output=True, text=True,
                           cwd=str(REPO / "submission" / "isles26_algorithm"))
        if p.returncode != 0:
            print(f"[FAIL] {cid}\n{p.stderr[-1000:]}", flush=True); continue
        pp = next(m for m in (tmp / "out" / "images").rglob("*.mha") if "probability" in m.parent.name)
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(pp))).transpose(2, 1, 0).astype(np.float32)
        ref = nib.load(str(RAW / "imagesTr" / f"{cid}_0000.nii.gz"))
        nib.save(nib.Nifti1Image(arr, ref.affine, ref.header), str(target))
    if i % 25 == 0:
        print(f"[progress] {i}/{len(cases)}", flush=True)
print("[done] no-TTA probabilities written")
PYEOF

echo "[next] score with:"
echo "  sbatch --array=0-5 --export=ALL,PROB_OVERRIDE=$OUT ... eval/run_official_eval.py score"
