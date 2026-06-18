#!/bin/bash
#SBATCH --job-name=eval_nnunet_r30
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH -p owners

# Run the existing soop_bench evaluation framework against nnU-Net predictions.

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs" "$REPO/baselines/reports/nnunet_r30_soop"

module purge
if [[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]]; then
  module use "$GROUP_HOME/modules"
fi
module load devel
module load python/3.12.1

export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"

source "$REPO/.venv-eval/bin/activate"

MASK_ROOT="$REPO/data/soop_bench/derivatives/lesion_masks"
mapfile -t SUBJECTS < <(ls -1 "$MASK_ROOT" | grep '^sub-' | sort -V)
if (( ${#SUBJECTS[@]} != 12 )); then
  echo "[error] expected 12 soop_bench subjects under $MASK_ROOT, found ${#SUBJECTS[@]}" >&2
  exit 2
fi

missing=()
for sub in "${SUBJECTS[@]}"; do
  pred="$REPO/work/predictions_nnunet_r30/${sub}.nii.gz"
  if [[ ! -s "$pred" ]]; then
    missing+=("$sub")
  fi
done
if (( ${#missing[@]} > 0 )); then
  echo "[error] missing nnU-Net predictions for ${#missing[@]} subject(s): ${missing[*]}" >&2
  exit 2
fi

PYTHONPATH="$REPO" python -m eval.run_eval \
  --data-root "$REPO/data/soop_bench" \
  --pred-dir  "$REPO/work/predictions_nnunet_r30" \
  --out-dir   "$REPO/baselines/reports/nnunet_r30_soop" \
  --title     "nnU-Net v2 3D fullres (ATLAS R3.0, 5-fold ensemble + TTA, T1w-only) on soop_bench"

python - <<'PY'
from pathlib import Path

report = Path("/scratch/users/sciget/isles26challenge/baselines/reports/nnunet_r30_soop/report.md")
notes = """

## Method Notes

- nnU-Net v2 3D full-resolution, T1w-only, trained on ATLAS R3.0 (Dataset502, 1450 cases) for 250 epochs.
- Inference uses the full 5-fold ensemble with test-time augmentation ENABLED.
- soop_bench T1w inputs are forced to RAS canonical orientation and skull-stripped with FreeSurfer SynthStrip before inference, then rigidly warped to TRACE/DWI space.
- CAVEAT: R3.0 training included the 169 sub-soop* subjects; soop_bench (sub-1..14) may overlap, so this OOD number can be optimistically biased.
"""
text = report.read_text()
marker = "\n## Method Notes\n"
if marker in text:
    text = text.split(marker)[0].rstrip()
report.write_text(text.rstrip() + notes + "\n")
PY
