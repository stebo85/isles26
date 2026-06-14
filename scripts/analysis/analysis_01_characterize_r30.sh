#!/bin/bash
#SBATCH --job-name=atlas_r30_chx
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH -p owners

# Per-session characterization of ATLAS R3.0 (1453 sessions: 1284 R-site + 169
# SOOP). Drop-in reuse of the R2.1 characterizer (R3.0 is BIDS-identical). Output
# inventory CSV feeds the nnU-Net converter (-> Dataset502_ATLASR30) + stratified
# 5-fold splits.

set -euo pipefail

ROOT="/scratch/users/sciget/isles26challenge"
DATA_ROOT="${ROOT}/data/ATLAS3_Training_Raw"
OUT_CSV="${ROOT}/baselines/reports/training_data_characterization_r30/per_session.csv"

cd "${ROOT}"

module purge
module load python/3.12.1
source .venv-eval/bin/activate

rm -f "${OUT_CSV}"
mkdir -p "$(dirname "${OUT_CSV}")"

python scripts/analysis/characterize_training_data.py \
    --root "${DATA_ROOT}" \
    --out-csv "${OUT_CSV}" \
    --shard-index 0 --shard-count 1

echo "DONE rows=$(wc -l < "${OUT_CSV}")  (expect ~1454 incl header)"
