#!/bin/bash
#SBATCH --job-name=nnunet_gen_report
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH -p owners

# Fast hidden-generalization triage: join TopK10 5-fold out-of-fold metrics to
# the converted ATLAS R3.0 manifest and report performance by center,
# chronicity, and lesion-size bin. This is not leave-center-out retraining; it
# is the diagnostic report that tells us where the true holdout runs should look.

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs" "$REPO/baselines/reports/nnunet_topk10_hidden_generalization"

module purge
if [[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]]; then
  module use "$GROUP_HOME/modules"
fi
module load devel
module load python/3.12.1

export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
source "$REPO/.venv-eval/bin/activate"

SUMMARY="${SUMMARY:-$REPO/baselines/reports/nnunet_topk10_crossval/crossval_summary.json}"
MANIFEST="$REPO/work/nnunet/nnUNet_raw/Dataset502_ATLASR30/conversion_manifest.csv"
OUT_DIR="${OUT_DIR:-$REPO/baselines/reports/nnunet_topk10_hidden_generalization}"

PYTHONPATH="$REPO" python "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/report_generalization_cv.py" \
  --summary "$SUMMARY" \
  --manifest "$MANIFEST" \
  --out-dir "$OUT_DIR" \
  --min-center-cases "${MIN_CENTER_CASES:-5}"
