#!/bin/bash
#SBATCH --job-name=reduce_official_sweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH -p owners
#
# Merge the official-metric sweep shards and aggregate them two ways:
#   - per-case MEANS, which is what the organizers' example_metrics.py prints
#   - RANK-then-aggregate across configurations, which is the ISLES ranking
#     convention and can disagree with the mean
# Both are emitted because we do not have the ranking document, and a
# configuration that wins under both is the safe thing to ship.
#
# Also publishes committed reports under baselines/reports/ so the operating
# point is reviewable rather than living in work/.
#
# Usage:
#   sbatch analysis_nnunet_49_reduce_official_sweep.sh              # d502
#   SURFACE=d507 sbatch analysis_nnunet_49_reduce_official_sweep.sh # d507

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs"

if command -v module >/dev/null 2>&1; then
  module purge
  if [[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]]; then
    module use "$GROUP_HOME/modules"
  fi
  module load devel
  module load python/3.12.1
fi

export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO"

PY="$REPO/.venv-official/bin/python"
SURFACE="${SURFACE:-d502}"

case "$SURFACE" in
  d502)
    OUT_DIR="$REPO/work/official_eval/d502_topk10"
    TITLE="ISLES'26 official metrics -- TopK10 out-of-fold, Dataset502 random folds (center-LEAKING)"
    REPORT_DIR="$REPO/baselines/reports/official_metrics_d502_topk10"
    EVAL_SET="atlas_r30_5fold_cv"
    NOTES="Official five-metric scoring of the TopK10 out-of-fold probabilities on the stratified-RANDOM 5-fold split. 52 of 55 centers appear in both the train and validation side of every fold, so this OVERSTATES generalization; use the d507 center-grouped report to choose an operating point."
    ;;
  d507)
    OUT_DIR="$REPO/work/official_eval/d507_topk10"
    TITLE="ISLES'26 official metrics -- TopK10 out-of-fold, Dataset507 center-grouped folds (HONEST)"
    REPORT_DIR="$REPO/baselines/reports/official_metrics_d507_topk10"
    EVAL_SET="atlas_r30_center_grouped_cv"
    NOTES="Official five-metric scoring on folds that hold out whole CENTERS. This is the honest generalization surface and the one the shipped operating point is selected on, because the hidden ISLES'26 test set spans 60+ centers."
    ;;
  *)
    echo "[error] unknown SURFACE=$SURFACE" >&2
    exit 2
    ;;
esac

if [[ ! -d "$OUT_DIR" ]]; then
  echo "[error] no sweep output at $OUT_DIR" >&2
  exit 2
fi

N_SHARDS=$(find "$OUT_DIR" -maxdepth 1 -name 'per_case_shard_*.json' | wc -l)
echo "[info] reducing $N_SHARDS shards from $OUT_DIR"
if [[ "$N_SHARDS" -eq 0 ]]; then
  echo "[error] no shards to reduce" >&2
  exit 2
fi

"$PY" "$REPO/eval/run_official_eval.py" reduce \
  --out-dir "$OUT_DIR" \
  --title "$TITLE" \
  --reference-threshold 0.50

mkdir -p "$REPORT_DIR"
cp -f "$OUT_DIR/report.json" "$OUT_DIR/report.md" "$REPORT_DIR/"
# per_case.csv is one row per (case x configuration): ~78k rows, worth keeping
# alongside the report so the operating-point choice can be re-derived without
# re-running the sweep.
cp -f "$OUT_DIR/per_case.csv" "$REPORT_DIR/"

"$PY" - "$REPORT_DIR" "$EVAL_SET" "$TITLE" "$NOTES" <<'PYEOF'
import json, sys
from pathlib import Path

report_dir, eval_set, title, notes = sys.argv[1:5]
report = json.loads((Path(report_dir) / "report.json").read_text())
meta = {
    "method": Path(report_dir).name,
    "display_name": title,
    "model_family": "nnunet",
    "eval_set": eval_set,
    "input_modalities": ["T1w"],
    "trained_on": "ATLAS R3.0 / ISLES'26 public training release",
    "status": "complete",
    "pipeline_dir": "baselines/models/nnunet_atlas_t1w",
    "evaluator": "isles26_official (work/isles26_upstream/utils/eval_utils.py)",
    "n_cases": report.get("n_cases"),
    "notes": notes,
}
(Path(report_dir) / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"[done] wrote {report_dir}/meta.json")
PYEOF

echo "[done] published $REPORT_DIR"
