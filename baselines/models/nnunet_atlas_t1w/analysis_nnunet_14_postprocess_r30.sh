#!/bin/bash
#SBATCH --job-name=nnunet_postproc_r30
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH -p owners

# Determine the best postprocessing for the default-plans ATLAS R3.0 model
# (Dataset502, nnUNetTrainer_250epochs, 3d_fullres) from the 5-fold
# cross-validation predictions, and report the consolidated CV Dice. This is
# the canonical nnU-Net step that picks connected-component postprocessing only
# if it improves the cross-validation metric, so it is a safe in-distribution
# refinement. Writes postprocessing.pkl + inference_instructions.txt under the
# results directory and a crossval summary we can read back.
#
#   sbatch baselines/models/nnunet_atlas_t1w/analysis_nnunet_14_postprocess_r30.sh

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs" "$REPO/baselines/reports/nnunet_r30_crossval"

module purge
if [[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]]; then
  module use "$GROUP_HOME/modules"
fi
module load devel
module load python/3.12.1
module load py-pytorch/2.4.1_py312

export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONNOUSERSITE=1
export nnUNet_compile=f
export nnUNet_raw="$REPO/work/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/work/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"

VENV="$REPO/.venv-nnunet"
source "$VENV/bin/activate"
python "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/install_trainer_variant.py"

# Single-config postprocessing determination (no cross-config ensembling).
nnUNetv2_find_best_configuration 502 \
  -c 3d_fullres \
  -tr nnUNetTrainer_250epochs \
  -p nnUNetPlans \
  --disable_ensembling \
  -np "${SLURM_CPUS_PER_TASK:-8}"

RESULT_DIR="$nnUNet_results/Dataset502_ATLASR30/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"
CV_DIR="$nnUNet_results/Dataset502_ATLASR30/crossval_results_folds_0_1_2_3_4"
echo "=== inference_instructions.txt ==="
cat "$nnUNet_results/Dataset502_ATLASR30"/inference_instructions.txt 2>/dev/null || true
echo "=== consolidated CV (pre-postproc) summary ==="
python - "$CV_DIR" <<'PY'
import json, sys
from pathlib import Path
cv = Path(sys.argv[1])
for name in ("summary.json",):
    p = cv / name
    if p.is_file():
        d = json.loads(p.read_text())
        print(f"[{cv.name}] foreground Dice mean = {d['foreground_mean']['Dice']:.4f}")
pp = cv / "postprocessed" / "summary.json"
if pp.is_file():
    d = json.loads(pp.read_text())
    print(f"[postprocessed] foreground Dice mean = {d['foreground_mean']['Dice']:.4f}")
PY

# Copy the key artifacts into the report dir for version control.
cp -f "$nnUNet_results/Dataset502_ATLASR30"/inference_instructions.txt \
      "$REPO/baselines/reports/nnunet_r30_crossval/" 2>/dev/null || true
cp -f "$CV_DIR/summary.json" \
      "$REPO/baselines/reports/nnunet_r30_crossval/crossval_summary.json" 2>/dev/null || true
cp -f "$CV_DIR/postprocessed/summary.json" \
      "$REPO/baselines/reports/nnunet_r30_crossval/crossval_postprocessed_summary.json" 2>/dev/null || true
echo "[done] postprocessing determination complete"
