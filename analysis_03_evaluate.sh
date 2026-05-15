#!/bin/bash
#SBATCH --job-name=eval_deepisles
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH -p owners

# Run the evaluation framework against the DeepISLES predictions.

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"

module purge
module load python/3.12.1
source .venv-eval/bin/activate

PYTHONPATH="$REPO" python -m eval.run_eval \
  --data-root "$REPO/data/soop_bench" \
  --pred-dir "$REPO/work/predictions_deepisles" \
  --out-dir  "$REPO/work/reports/deepisles" \
  --title    "DeepISLES (SEALS+NVAUTO+SWAN) on soop_bench"
