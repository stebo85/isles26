#!/bin/bash
#SBATCH --job-name=eval_nnunet_atlas_val
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH -p owners

# Evaluate nnU-Net predictions on the held-out ATLAS R2.1 fold-0 validation set.

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs" "$REPO/baselines/reports/nnunet_atlas_val"

module purge
if [[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]]; then
  module use "$GROUP_HOME/modules"
fi
module load devel
module load python/3.12.1

export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1

source "$REPO/.venv-eval/bin/activate"

SPLITS="$REPO/work/nnunet/nnUNet_preprocessed/Dataset501_ATLASR21/splits_final.json"
PRED_DIR="$REPO/work/predictions_atlas_val"
if [[ ! -s "$SPLITS" ]]; then
  echo "[error] missing fold splits: $SPLITS" >&2
  exit 2
fi

mapfile -t VAL_CASES < <(python - "$SPLITS" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
folds = data.get("folds", data) if isinstance(data, dict) else data
for case_id in folds[0]["val"]:
    print(case_id)
PY
)
if (( ${#VAL_CASES[@]} == 0 )); then
  echo "[error] fold 0 validation list is empty in $SPLITS" >&2
  exit 2
fi

missing=()
for case_id in "${VAL_CASES[@]}"; do
  if [[ ! -s "$PRED_DIR/${case_id}.nii.gz" ]]; then
    missing+=("$case_id")
  fi
done
if (( ${#missing[@]} > 0 )); then
  echo "[error] missing ATLAS-val predictions for ${#missing[@]} case(s): ${missing[*]}" >&2
  exit 2
fi

PYTHONPATH="$REPO" python baselines/models/nnunet_atlas_t1w/nnunet_helpers/run_atlas_eval.py \
  --manifest work/nnunet/nnUNet_raw/Dataset501_ATLASR21/conversion_manifest.csv \
  --splits work/nnunet/nnUNet_preprocessed/Dataset501_ATLASR21/splits_final.json \
  --pred-dir work/predictions_atlas_val \
  --per-session-csv baselines/reports/training_data_characterization/per_session.csv \
  --out-dir baselines/reports/nnunet_atlas_val \
  --title "nnU-Net v2 3D fullres (ATLAS R2.1, fold 0, 250 ep) — ATLAS R2.1 fold-0 held-out val"
