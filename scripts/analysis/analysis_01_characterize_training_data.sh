#!/bin/bash
#SBATCH --job-name=atlas_chx
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH -p owners

set -euo pipefail

ROOT="/scratch/users/sciget/isles26challenge"
DATA_ROOT="${ROOT}/data/ATLAS_R2.1_raw/Training_Raw"
OUT_CSV="${ROOT}/baselines/reports/training_data_characterization/per_session.csv"

cd "${ROOT}"

module purge
module load python/3.12.1

# .venv-eval already has nibabel/numpy/pandas/scipy/matplotlib (see agents.md).
source .venv-eval/bin/activate

# Single-shard run. CSV will get re-created from scratch.
rm -f "${OUT_CSV}"
mkdir -p "$(dirname "${OUT_CSV}")"

python scripts/analysis/characterize_training_data.py \
    --root "${DATA_ROOT}" \
    --out-csv "${OUT_CSV}" \
    --shard-index 0 --shard-count 1

echo "DONE rows=$(wc -l < "${OUT_CSV}")"
