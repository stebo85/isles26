#!/bin/bash
#SBATCH --job-name=nnunet_prepare_center507
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH -p owners

# Build the corrected ATLAS R3.0 Dataset507 lineage from source. The conversion
# and preprocessing happen in an isolated nnU-Net root. Dataset507 is promoted
# only after its case set, grouped split, and preprocessing outputs are verified.

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
DATASET_ID=507
DATASET_NAME=Dataset507_ATLASR30_CORRECTED
EXPECTED_CASES=1452
EXPECTED_SKIP=ATLAS_1424
EXPECTED_INVENTORY_SHA256=d9901228bdcc2b19a7623fbbcb4d3839b5b3390c890d982be5c1efbf8dc3f7b1
EXPECTED_ASSIGNMENT_SHA256=168683caeabdb8bb7000b431972e4493f5b030e28ed223c4d839bb5d0b390492
EXPECTED_NNUNET_VERSION=2.6.2
EXPECTED_JOB_NAME=nnunet_prepare_center507

MODEL_DIR="$REPO/baselines/models/nnunet_atlas_t1w"
HELPER_DIR="$MODEL_DIR/nnunet_helpers"
REPORT_DIR="$REPO/baselines/reports/nnunet_center_grouped_splits"
INVENTORY="$REPO/baselines/reports/training_data_characterization_r30/per_session.csv"
TRACKED_SPLITS="$REPORT_DIR/splits_final.json"
TRACKED_SPLIT_SUMMARY="$REPORT_DIR/splits_summary.json"
CONVERTER="$HELPER_DIR/convert_atlas_r21_to_nnunet.py"
SPLIT_BUILDER="$HELPER_DIR/make_center_grouped_splits.py"
VERIFIER="$HELPER_DIR/verify_nnunet_dataset.py"

FINAL_RAW_ROOT="$REPO/work/nnunet/nnUNet_raw"
FINAL_PREPROCESSED_ROOT="$REPO/work/nnunet/nnUNet_preprocessed"
FINAL_RESULTS_ROOT="$REPO/work/nnunet/nnUNet_results"
FINAL_RAW_DIR="$FINAL_RAW_ROOT/$DATASET_NAME"
FINAL_PREPROCESSED_DIR="$FINAL_PREPROCESSED_ROOT/$DATASET_NAME"
FROZEN_LINEAGE="$REPORT_DIR/dataset_lineage.json"
STAGING_PARENT="$REPO/work/nnunet_center_grouped_staging"

if [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_JOB_NAME:-}" != "$EXPECTED_JOB_NAME" ]]; then
  echo "[error] preparation must run in a SLURM allocation (use analysis_nnunet_46_submit_center_topk10.sh)" >&2
  exit 2
fi

cd "$REPO"
mkdir -p "$REPO/logs" "$FINAL_RAW_ROOT" "$FINAL_PREPROCESSED_ROOT" \
  "$FINAL_RESULTS_ROOT" "$STAGING_PARENT"

# Serialize the Dataset507 existence check and promotion. This lock never
# covers Dataset502 or any other nnU-Net dataset lineage.
exec 9>"$REPO/work/nnunet/.prepare_dataset507.lock"
if ! flock -n 9; then
  echo "[error] another Dataset507 preparation owns the build lock" >&2
  exit 2
fi

assert_dataset_id_available() {
  local root
  local -a collisions
  for root in "$FINAL_RAW_ROOT" "$FINAL_PREPROCESSED_ROOT" "$FINAL_RESULTS_ROOT"; do
    mapfile -t collisions < <(find "$root" -mindepth 1 -maxdepth 1 -name 'Dataset507*' -print | sort)
    if (( ${#collisions[@]} > 0 )); then
      printf '[error] refusing to overwrite an existing dataset-ID 507 path: %s\n' "${collisions[@]}" >&2
      return 2
    fi
  done
}

assert_dataset_id_available

STAGE_ROOT="$(mktemp -d "$STAGING_PARENT/prepare_${SLURM_JOB_ID}_XXXXXXXX")"
STAGE_RAW_ROOT="$STAGE_ROOT/nnUNet_raw"
STAGE_PREPROCESSED_ROOT="$STAGE_ROOT/nnUNet_preprocessed"
STAGE_RESULTS_ROOT="$STAGE_ROOT/nnUNet_results"
STAGE_RAW_DIR="$STAGE_RAW_ROOT/$DATASET_NAME"
STAGE_PREPROCESSED_DIR="$STAGE_PREPROCESSED_ROOT/$DATASET_NAME"
STAGE_LINEAGE="$STAGE_ROOT/dataset_lineage.json"
mkdir -p "$STAGE_RAW_ROOT" "$STAGE_PREPROCESSED_ROOT" "$STAGE_RESULTS_ROOT"

RAW_PROMOTED=0
PREPROCESSED_PROMOTED=0
BUILD_COMPLETE=0
RAW_DIR_INODE=""
PREPROCESSED_DIR_INODE=""
LINEAGE_TMP=""

same_inode() {
  local path=$1
  local expected=$2
  [[ -d "$path" ]] && [[ "$(stat -Lc '%d:%i' "$path")" == "$expected" ]]
}

cleanup() {
  local status=$?
  local rollback_safe=1
  trap - EXIT
  set +e

  if (( status != 0 || BUILD_COMPLETE == 0 )); then
    # A promoted path is moved back only when it is the exact directory inode
    # created by this job. Unknown paths are left untouched for manual audit.
    if (( PREPROCESSED_PROMOTED == 1 )) && \
      same_inode "$FINAL_PREPROCESSED_DIR" "$PREPROCESSED_DIR_INODE" && \
      [[ ! -e "$STAGE_PREPROCESSED_DIR" ]]; then
      if mv -- "$FINAL_PREPROCESSED_DIR" "$STAGE_PREPROCESSED_DIR"; then
        echo "[rollback] returned this job's preprocessed Dataset507 to staging" >&2
      else
        rollback_safe=0
      fi
    elif (( PREPROCESSED_PROMOTED == 1 )); then
      rollback_safe=0
    fi
    if (( RAW_PROMOTED == 1 )) && \
      same_inode "$FINAL_RAW_DIR" "$RAW_DIR_INODE" && \
      [[ ! -e "$STAGE_RAW_DIR" ]]; then
      if mv -- "$FINAL_RAW_DIR" "$STAGE_RAW_DIR"; then
        echo "[rollback] returned this job's raw Dataset507 to staging" >&2
      else
        rollback_safe=0
      fi
    elif (( RAW_PROMOTED == 1 )); then
      rollback_safe=0
    fi
    if (( rollback_safe == 0 )); then
      echo "[error] rollback ownership check failed; retained staging for manual audit: $STAGE_ROOT" >&2
    else
      case "$STAGE_ROOT" in
        "$STAGING_PARENT"/prepare_${SLURM_JOB_ID}_*) rm -rf -- "$STAGE_ROOT" ;;
        *) rollback_safe=0 ;;
      esac
      if (( rollback_safe == 1 )); then
        echo "[cleanup] removed this failed job's staging data" >&2
      else
        echo "[error] refusing unexpected staging cleanup path: $STAGE_ROOT" >&2
      fi
    fi
  else
    case "$STAGE_ROOT" in
      "$STAGING_PARENT"/prepare_${SLURM_JOB_ID}_*) rm -rf -- "$STAGE_ROOT" ;;
      *) echo "[warn] refusing unexpected staging cleanup path: $STAGE_ROOT" >&2 ;;
    esac
  fi
  if [[ -n "$LINEAGE_TMP" && "$LINEAGE_TMP" == "$REPORT_DIR"/.dataset_lineage.json.*.tmp ]]; then
    rm -f -- "$LINEAGE_TMP"
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

module purge
if [[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]]; then
  module use "$GROUP_HOME/modules"
fi
module load devel
module load python/3.12.1
module load py-pytorch/2.4.1_py312

export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONNOUSERSITE=1
export nnUNet_raw="$STAGE_RAW_ROOT"
export nnUNet_preprocessed="$STAGE_PREPROCESSED_ROOT"
export nnUNet_results="$STAGE_RESULTS_ROOT"

VENV="$REPO/.venv-nnunet"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[error] missing $VENV; build the pinned nnU-Net environment first" >&2
  exit 2
fi
source "$VENV/bin/activate"

for required in "$INVENTORY" "$TRACKED_SPLITS" "$TRACKED_SPLIT_SUMMARY" \
  "$CONVERTER" "$SPLIT_BUILDER" "$VERIFIER"; do
  if [[ ! -s "$required" ]]; then
    echo "[error] missing required input: $required" >&2
    exit 2
  fi
done

printf '%s  %s\n' "$EXPECTED_INVENTORY_SHA256" "$INVENTORY" | sha256sum --check --status || {
  echo "[error] corrected source inventory checksum does not match: $INVENTORY" >&2
  exit 2
}

python - "$TRACKED_SPLIT_SUMMARY" "$EXPECTED_ASSIGNMENT_SHA256" "$EXPECTED_CASES" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_assignment_sha = sys.argv[2]
expected_cases = int(sys.argv[3])
if summary.get("assignment_input_sha256") != expected_assignment_sha:
    raise SystemExit("tracked split summary has the wrong assignment-input SHA-256")
if summary.get("global", {}).get("n_cases") != expected_cases:
    raise SystemExit("tracked split summary has the wrong case count")
if summary.get("validation", {}).get("passed") is not True:
    raise SystemExit("tracked split summary does not report passed invariants")
print("[check] tracked center-grouped split certificate is valid")
PY

python - "$EXPECTED_NNUNET_VERSION" <<'PY'
import importlib.metadata
import sys

expected = sys.argv[1]
actual = importlib.metadata.version("nnunetv2")
if actual != expected:
    raise SystemExit(f"nnunetv2 version mismatch: expected {expected}, found {actual}")
print(f"[info] nnunetv2={actual}")
PY

PLAN_BIN="$(command -v nnUNetv2_plan_and_preprocess || true)"
if [[ -z "$PLAN_BIN" || "$PLAN_BIN" != "$VENV/"* ]]; then
  echo "[error] nnUNetv2_plan_and_preprocess is not inside $VENV: ${PLAN_BIN:-not found}" >&2
  exit 2
fi

python "$CONVERTER" convert \
  --inventory "$INVENTORY" \
  --dataset-dir "$STAGE_RAW_DIR" \
  --repo-root "$REPO"

python - "$STAGE_RAW_DIR/conversion_manifest.csv" \
  "$STAGE_RAW_DIR/conversion_skips.csv" "$EXPECTED_CASES" "$EXPECTED_SKIP" <<'PY'
import csv
import sys
from pathlib import Path

manifest_path, skips_path = map(Path, sys.argv[1:3])
expected_cases = int(sys.argv[3])
expected_skip = sys.argv[4]
with manifest_path.open(newline="") as stream:
    rows = list(csv.DictReader(stream))
with skips_path.open(newline="") as stream:
    skips = list(csv.DictReader(stream))
ids = [row["nnunet_id"] for row in rows]
if len(rows) != expected_cases or len(set(ids)) != expected_cases:
    raise SystemExit(
        f"conversion manifest must contain {expected_cases} unique cases; "
        f"found {len(rows)} rows / {len(set(ids))} IDs"
    )
if len(skips) != 1 or skips[0].get("nnunet_id") != expected_skip:
    found = [(row.get("nnunet_id"), row.get("reason")) for row in skips]
    raise SystemExit(f"expected sole skip {expected_skip}; found {found}")
if skips[0].get("reason") != "affine_mismatch":
    raise SystemExit(
        f"expected {expected_skip} affine_mismatch; found {skips[0].get('reason')!r}"
    )
if expected_skip in ids:
    raise SystemExit(f"skipped case unexpectedly present in manifest: {expected_skip}")
print(f"[check] conversion: {len(rows)} unique cases; sole skip={expected_skip}")
PY

nnUNetv2_plan_and_preprocess \
  -d "$DATASET_ID" \
  -c 3d_fullres \
  --verify_dataset_integrity \
  --clean \
  -npfp 16 \
  -np 16

python "$SPLIT_BUILDER" \
  --manifest "$STAGE_RAW_DIR/conversion_manifest.csv" \
  --out-splits "$STAGE_PREPROCESSED_DIR/splits_final.json" \
  --out-summary-json "$STAGE_PREPROCESSED_DIR/splits_summary.json" \
  --out-summary-md "$STAGE_PREPROCESSED_DIR/splits_report.md" \
  --folds 5 \
  --seed 42 \
  --restarts 256

cmp --silent "$STAGE_PREPROCESSED_DIR/splits_final.json" "$TRACKED_SPLITS" || {
  echo "[error] staged center-grouped split differs from tracked splits_final.json" >&2
  exit 2
}

python "$VERIFIER" \
  --raw-dir "$STAGE_RAW_DIR" \
  --preprocessed-dir "$STAGE_PREPROCESSED_DIR" \
  --manifest "$STAGE_RAW_DIR/conversion_manifest.csv" \
  --splits "$STAGE_PREPROCESSED_DIR/splits_final.json" \
  --split-summary "$STAGE_PREPROCESSED_DIR/splits_summary.json" \
  --expected-dataset-name "$DATASET_NAME" \
  --expected-cases "$EXPECTED_CASES" \
  --expected-assignment-sha "$EXPECTED_ASSIGNMENT_SHA256" \
  --out-lineage "$STAGE_LINEAGE"

# Directory rename is atomic only within a filesystem. Refuse promotion if the
# configured roots do not share their devices with this job's staging paths.
if [[ "$(stat -Lc '%d' "$STAGE_RAW_DIR")" != "$(stat -Lc '%d' "$FINAL_RAW_ROOT")" ]] || \
   [[ "$(stat -Lc '%d' "$STAGE_PREPROCESSED_DIR")" != "$(stat -Lc '%d' "$FINAL_PREPROCESSED_ROOT")" ]]; then
  echo "[error] staging and final roots are on different filesystems; atomic promotion is unavailable" >&2
  exit 2
fi

RAW_DIR_INODE="$(stat -Lc '%d:%i' "$STAGE_RAW_DIR")"
PREPROCESSED_DIR_INODE="$(stat -Lc '%d:%i' "$STAGE_PREPROCESSED_DIR")"
assert_dataset_id_available
mv --no-clobber --no-target-directory -- "$STAGE_RAW_DIR" "$FINAL_RAW_DIR"
if ! same_inode "$FINAL_RAW_DIR" "$RAW_DIR_INODE" || [[ -e "$STAGE_RAW_DIR" ]]; then
  echo "[error] raw Dataset507 promotion did not complete without collision" >&2
  exit 2
fi
RAW_PROMOTED=1
mv --no-clobber --no-target-directory -- "$STAGE_PREPROCESSED_DIR" "$FINAL_PREPROCESSED_DIR"
if ! same_inode "$FINAL_PREPROCESSED_DIR" "$PREPROCESSED_DIR_INODE" || \
  [[ -e "$STAGE_PREPROCESSED_DIR" ]]; then
  echo "[error] preprocessed Dataset507 promotion did not complete without collision" >&2
  exit 2
fi
PREPROCESSED_PROMOTED=1

# The converter records absolute output paths. Re-run it against the promoted
# raw directory; source fingerprints let it refresh the manifest without
# rewriting image or label payloads.
python "$CONVERTER" convert \
  --inventory "$INVENTORY" \
  --dataset-dir "$FINAL_RAW_DIR" \
  --repo-root "$REPO"

python "$SPLIT_BUILDER" \
  --manifest "$FINAL_RAW_DIR/conversion_manifest.csv" \
  --out-splits "$FINAL_PREPROCESSED_DIR/splits_final.json" \
  --out-summary-json "$FINAL_PREPROCESSED_DIR/splits_summary.json" \
  --out-summary-md "$FINAL_PREPROCESSED_DIR/splits_report.md" \
  --folds 5 \
  --seed 42 \
  --restarts 256

cmp --silent "$FINAL_PREPROCESSED_DIR/splits_final.json" "$TRACKED_SPLITS" || {
  echo "[error] promoted center-grouped split differs from tracked splits_final.json" >&2
  exit 2
}

python "$VERIFIER" \
  --raw-dir "$FINAL_RAW_DIR" \
  --preprocessed-dir "$FINAL_PREPROCESSED_DIR" \
  --manifest "$FINAL_RAW_DIR/conversion_manifest.csv" \
  --splits "$FINAL_PREPROCESSED_DIR/splits_final.json" \
  --split-summary "$FINAL_PREPROCESSED_DIR/splits_summary.json" \
  --expected-dataset-name "$DATASET_NAME" \
  --expected-cases "$EXPECTED_CASES" \
  --expected-assignment-sha "$EXPECTED_ASSIGNMENT_SHA256" \
  --out-lineage "$FINAL_PREPROCESSED_DIR/dataset_lineage.json"

cmp --silent "$STAGE_LINEAGE" "$FINAL_PREPROCESSED_DIR/dataset_lineage.json" || {
  echo "[error] dataset lineage changed during promotion and manifest path refresh" >&2
  exit 2
}

if [[ -e "$FROZEN_LINEAGE" ]]; then
  cmp --silent "$FINAL_PREPROCESSED_DIR/dataset_lineage.json" "$FROZEN_LINEAGE" || {
    echo "[error] prepared lineage differs from frozen certificate: $FROZEN_LINEAGE" >&2
    exit 2
  }
else
  LINEAGE_TMP="$REPORT_DIR/.dataset_lineage.json.${SLURM_JOB_ID}.tmp"
  cp -- "$FINAL_PREPROCESSED_DIR/dataset_lineage.json" "$LINEAGE_TMP"
  chmod 0444 "$LINEAGE_TMP"
  mv -- "$LINEAGE_TMP" "$FROZEN_LINEAGE"
  LINEAGE_TMP=""
fi

BUILD_COMPLETE=1
echo "[done] prepared immutable $DATASET_NAME with $EXPECTED_CASES cases"
echo "[done] raw: $FINAL_RAW_DIR"
echo "[done] preprocessed: $FINAL_PREPROCESSED_DIR"
echo "[done] lineage: $FROZEN_LINEAGE"
