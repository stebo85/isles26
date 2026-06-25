#!/bin/bash
#SBATCH --job-name=nnunet_gen_splits
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH -p owners

# Build leave-center-out and leave-chronicity-out split plans from the converted
# ATLAS R3.0 manifest. These are split definitions/protocols; run dedicated
# training jobs against one split at a time for true hidden-generalization scores.

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs" "$REPO/baselines/reports/hidden_generalization_splits"

module purge
if [[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]]; then
  module use "$GROUP_HOME/modules"
fi
module load devel
module load python/3.12.1

export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
source "$REPO/.venv-eval/bin/activate"

MANIFEST="$REPO/work/nnunet/nnUNet_raw/Dataset502_ATLASR30/conversion_manifest.csv"
OUT_DIR="$REPO/baselines/reports/hidden_generalization_splits"
PYTHONPATH="$REPO" python "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/make_generalization_splits.py" \
  --manifest "$MANIFEST" \
  --out-json "$OUT_DIR/splits_generalization.json" \
  --out-summary-json "$OUT_DIR/splits_generalization_summary.json" \
  --out-md "$OUT_DIR/report.md" \
  --min-center-cases "${MIN_CENTER_CASES:-20}"
