#!/bin/bash

# Materialize the fold-0 ATLAS validation case list for the array prediction job.
#
# Usage:
#   bash baselines/models/nnunet_atlas_t1w/analysis_nnunet_05a_compute_atlas_val_array.sh
#
# The script writes work/predictions_atlas_val/val_cases.txt idempotently and
# prints the sbatch array range as:
#   array=0-N

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"

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

EVAL_PY="$REPO/.venv-eval/bin/python"
if [[ ! -x "$EVAL_PY" ]]; then
  echo "[error] missing $EVAL_PY; build the eval venv first" >&2
  exit 2
fi

SPLITS="$REPO/work/nnunet/nnUNet_preprocessed/Dataset501_ATLASR21/splits_final.json"
PRED_DIR="$REPO/work/predictions_atlas_val"
VAL_LIST="$PRED_DIR/val_cases.txt"
mkdir -p "$PRED_DIR"

if [[ ! -s "$SPLITS" ]]; then
  echo "[error] missing fold splits: $SPLITS" >&2
  exit 2
fi

TMP_LIST="$(mktemp "$VAL_LIST.tmp.XXXXXX")"
trap 'rm -f "$TMP_LIST"' EXIT

"$EVAL_PY" - "$SPLITS" "$TMP_LIST" <<'PY'
import json
import sys
from pathlib import Path

splits = Path(sys.argv[1])
out = Path(sys.argv[2])

data = json.loads(splits.read_text())
folds = data.get("folds", data) if isinstance(data, dict) else data
try:
    val_cases = list(folds[0]["val"])
except Exception as exc:
    raise SystemExit(f"could not read fold 0 val cases from {splits}: {exc}")
if not val_cases:
    raise SystemExit(f"fold 0 validation list is empty in {splits}")
if any(not isinstance(case_id, str) or not case_id for case_id in val_cases):
    raise SystemExit(f"fold 0 validation list in {splits} contains an invalid case id")
if len(set(val_cases)) != len(val_cases):
    raise SystemExit(f"fold 0 validation list in {splits} contains duplicate case ids")

out.write_text("\n".join(val_cases) + "\n")
PY

if [[ -s "$VAL_LIST" ]] && cmp -s "$TMP_LIST" "$VAL_LIST"; then
  echo "[ok] val case list already current: $VAL_LIST" >&2
  rm -f "$TMP_LIST"
else
  mv "$TMP_LIST" "$VAL_LIST"
  echo "[done] wrote val case list: $VAL_LIST" >&2
fi

N="$(wc -l < "$VAL_LIST" | tr -d '[:space:]')"
if (( N <= 0 )); then
  echo "[error] val case list is empty after write: $VAL_LIST" >&2
  exit 2
fi

echo "array=0-$((N - 1))"
