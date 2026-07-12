#!/bin/bash
#SBATCH --job-name=nnunet_train_center_topk10
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6|GPU_CC:8.0|GPU_CC:7.5|GPU_CC:7.0"
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --array=0-4
#SBATCH -p owners

# Train corrected ATLAS R3.0 Dataset507 with Dice+TopK10. Each validation fold
# contains complete acquisition centers, so the OOF result measures transfer to
# unseen centers rather than within-center interpolation.

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs"

child_pid=""
tmp_dataset_lineage=""
tmp_training_lineage=""

cleanup() {
  [[ -z "$tmp_dataset_lineage" ]] || rm -f "$tmp_dataset_lineage"
  [[ -z "$tmp_training_lineage" ]] || rm -f "$tmp_training_lineage"
}

stop_child() {
  if [[ -n "$child_pid" ]]; then
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
    child_pid=""
  fi
}

handle_usr1() {
  trap - USR1 TERM
  set +e
  echo "[warn] preemption signal received; stopping nnU-Net before requeue" >&2
  stop_child
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "[error] cannot requeue outside a Slurm allocation" >&2
    exit 75
  fi
  if scontrol requeue "$SLURM_JOB_ID"; then
    echo "[info] requeued Slurm job $SLURM_JOB_ID"
    exit 0
  fi
  echo "[error] Slurm requeue failed for job $SLURM_JOB_ID" >&2
  exit 75
}

handle_term() {
  trap - USR1 TERM
  set +e
  echo "[warn] termination signal received; stopping nnU-Net" >&2
  stop_child
  exit 143
}

trap cleanup EXIT
trap handle_usr1 USR1
trap handle_term TERM

module purge
if [[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]]; then
  module use "$GROUP_HOME/modules"
fi
module load devel
module load python/3.12.1
module load py-pytorch/2.4.1_py312

export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export nnUNet_n_proc_DA="${SLURM_CPUS_PER_TASK:-8}"
export nnUNet_compile=f
export PYTHONNOUSERSITE=1
export nnUNet_raw="$REPO/work/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/work/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"

VENV="$REPO/.venv-nnunet"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[error] missing $VENV" >&2
  exit 2
fi
source "$VENV/bin/activate"
if [[ ! -x "$VENV/bin/nnUNetv2_train" ]] || \
   [[ "$(readlink -f "$(command -v nnUNetv2_train)")" != "$(readlink -f "$VENV/bin/nnUNetv2_train")" ]]; then
  echo "[error] nnUNetv2_train does not resolve inside $VENV" >&2
  exit 2
fi

FOLD="${SLURM_ARRAY_TASK_ID:-0}"
if ! [[ "$FOLD" =~ ^[0-4]$ ]]; then
  echo "[error] SLURM_ARRAY_TASK_ID must be 0-4, got: $FOLD" >&2
  exit 2
fi

DATASET_ID=507
DATASET_NAME=Dataset507_ATLASR30_CORRECTED
TRAINER=nnUNetTrainerDiceTopK10Loss
PLANS=nnUNetPlans
CONFIGURATION=3d_fullres
ASSIGNMENT_SHA=168683caeabdb8bb7000b431972e4493f5b030e28ed223c4d839bb5d0b390492
RAW_DIR="$nnUNet_raw/$DATASET_NAME"
PREPROCESSED_DIR="$nnUNet_preprocessed/$DATASET_NAME"
MANIFEST="$RAW_DIR/conversion_manifest.csv"
SPLITS="$PREPROCESSED_DIR/splits_final.json"
SPLIT_SUMMARY="$PREPROCESSED_DIR/splits_summary.json"
DATASET_LINEAGE="$PREPROCESSED_DIR/dataset_lineage.json"
TRACKED_DIR="$REPO/baselines/reports/nnunet_center_grouped_splits"
TRACKED_SPLITS="$TRACKED_DIR/splits_final.json"
TRACKED_LINEAGE="$TRACKED_DIR/dataset_lineage.json"
VERIFIER="$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/verify_nnunet_dataset.py"
SCRIPT_PATH="$REPO/baselines/models/nnunet_atlas_t1w/analysis_nnunet_43_train_center_topk10.sh"

for required in \
  "$MANIFEST" \
  "$SPLITS" \
  "$SPLIT_SUMMARY" \
  "$DATASET_LINEAGE" \
  "$TRACKED_SPLITS" \
  "$TRACKED_LINEAGE" \
  "$VERIFIER" \
  "$SCRIPT_PATH"; do
  if [[ ! -s "$required" ]]; then
    echo "[error] missing required input: $required; run step 42 first" >&2
    exit 2
  fi
done

if ! cmp -s "$SPLITS" "$TRACKED_SPLITS"; then
  echo "[error] installed splits differ from the reviewed center-group split" >&2
  exit 2
fi

TMP_ROOT="${SLURM_TMPDIR:-/tmp}"
tmp_dataset_lineage="$(mktemp "$TMP_ROOT/nnunet507_lineage.XXXXXX")"
rm -f "$tmp_dataset_lineage"
python "$VERIFIER" \
  --raw-dir "$RAW_DIR" \
  --preprocessed-dir "$PREPROCESSED_DIR" \
  --manifest "$MANIFEST" \
  --splits "$SPLITS" \
  --split-summary "$SPLIT_SUMMARY" \
  --expected-dataset-name "$DATASET_NAME" \
  --expected-cases 1452 \
  --expected-assignment-sha "$ASSIGNMENT_SHA" \
  --out-lineage "$tmp_dataset_lineage"

if ! cmp -s "$tmp_dataset_lineage" "$DATASET_LINEAGE"; then
  echo "[error] Dataset507 no longer matches its immutable dataset lineage" >&2
  exit 2
fi
if ! cmp -s "$DATASET_LINEAGE" "$TRACKED_LINEAGE"; then
  echo "[error] installed and tracked Dataset507 lineages differ" >&2
  exit 2
fi

RESULT_BASE="$nnUNet_results/$DATASET_NAME/${TRAINER}__${PLANS}__${CONFIGURATION}"
FOLD_DIR="$RESULT_BASE/fold_${FOLD}"
VAL_DIR="$FOLD_DIR/validation"
TRAINING_LINEAGE="$FOLD_DIR/training_lineage.json"
CHECKPOINT_FINAL="$FOLD_DIR/checkpoint_final.pth"
CHECKPOINT_LATEST="$FOLD_DIR/checkpoint_latest.pth"
CHECKPOINT_BEST="$FOLD_DIR/checkpoint_best.pth"
LOCK_DIR="$nnUNet_results/.locks/$DATASET_NAME/${TRAINER}__${PLANS}__${CONFIGURATION}"
mkdir -p "$LOCK_DIR"
exec 9>"$LOCK_DIR/fold_${FOLD}.lock"
if ! flock -n 9; then
  echo "[error] another process holds the Dataset507 fold $FOLD training lock" >&2
  exit 73
fi

GIT_COMMIT="$(git rev-parse --verify HEAD)"
tmp_training_lineage="$(mktemp "$TMP_ROOT/nnunet507_training_lineage.XXXXXX")"
python - "$SPLITS" "$FOLD" "$DATASET_LINEAGE" "$SPLIT_SUMMARY" "$SCRIPT_PATH" \
  "$GIT_COMMIT" "$tmp_training_lineage" "$DATASET_NAME" "$TRAINER" "$PLANS" \
  "$CONFIGURATION" "$ASSIGNMENT_SHA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    splits_path,
    fold_text,
    dataset_lineage_path,
    split_summary_path,
    script_path,
    git_commit,
    output_path,
    dataset_name,
    trainer,
    plans,
    configuration,
    assignment_sha,
) = sys.argv[1:]
fold = int(fold_text)
splits = json.loads(Path(splits_path).read_text())
if not isinstance(splits, list) or len(splits) != 5:
    raise SystemExit("expected exactly five folds")
entry = splits[fold]

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

payload = {
    "schema_version": 1,
    "dataset_id": 507,
    "dataset_name": dataset_name,
    "configuration": configuration,
    "trainer": trainer,
    "plans": plans,
    "fold": fold,
    "assignment_input_sha256": assignment_sha,
    "dataset_lineage_sha256": sha256(dataset_lineage_path),
    "splits_final_sha256": sha256(splits_path),
    "splits_summary_sha256": sha256(split_summary_path),
    "training_script_sha256": sha256(script_path),
    "git_commit": git_commit,
    "train_case_ids": list(entry["train"]),
    "validation_case_ids": list(entry["val"]),
}
Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

if [[ -d "$FOLD_DIR" && ! -s "$TRAINING_LINEAGE" ]] && \
   find "$FOLD_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "[error] fold $FOLD has results but no training lineage; refusing to reuse them" >&2
  exit 2
fi
mkdir -p "$FOLD_DIR"
if [[ -s "$TRAINING_LINEAGE" ]]; then
  python - "$tmp_training_lineage" "$TRAINING_LINEAGE" "$CHECKPOINT_FINAL" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

expected_path, actual_path, checkpoint_path = map(Path, sys.argv[1:])
expected = json.loads(expected_path.read_text())
actual = json.loads(actual_path.read_text())
checkpoint_certificate = actual.pop("checkpoint_final", None)
if actual != expected:
    raise SystemExit("training lineage does not match this dataset/split/trainer/code")
if checkpoint_certificate is not None:
    if not checkpoint_path.is_file():
        raise SystemExit("training lineage certifies a final checkpoint that is missing")
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    certificate = {
        "sha256": digest.hexdigest(),
        "size_bytes": checkpoint_path.stat().st_size,
    }
    if checkpoint_certificate != certificate:
        raise SystemExit("final checkpoint does not match its training lineage certificate")
print("[ok] existing fold training lineage verified")
PY
else
  mv "$tmp_training_lineage" "$TRAINING_LINEAGE"
  tmp_training_lineage=""
fi

finalize_training_lineage() {
  python - "$TRAINING_LINEAGE" "$CHECKPOINT_FINAL" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

lineage_path, checkpoint_path = map(Path, sys.argv[1:])
if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
    raise SystemExit(f"cannot certify missing or empty checkpoint: {checkpoint_path}")
payload = json.loads(lineage_path.read_text())
digest = hashlib.sha256()
with checkpoint_path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
certificate = {
    "sha256": digest.hexdigest(),
    "size_bytes": checkpoint_path.stat().st_size,
}
existing = payload.get("checkpoint_final")
if existing is not None and existing != certificate:
    raise SystemExit("refusing to replace a mismatched checkpoint certificate")
payload["checkpoint_final"] = certificate
temporary = lineage_path.with_name(lineage_path.name + f".{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, lineage_path)
print(f"[ok] final checkpoint certified in {lineage_path}")
PY
}

validation_is_exact() {
  python - "$SPLITS" "$FOLD" "$VAL_DIR" <<'PY'
import json
import sys
from pathlib import Path

splits_path, fold_text, val_dir_text = sys.argv[1:]
expected = set(json.loads(Path(splits_path).read_text())[int(fold_text)]["val"])
val_dir = Path(val_dir_text)
nifti = {path.name[:-7] for path in val_dir.glob("*.nii.gz")} if val_dir.is_dir() else set()
npz = {path.stem for path in val_dir.glob("*.npz")} if val_dir.is_dir() else set()
pkl = {path.stem for path in val_dir.glob("*.pkl")} if val_dir.is_dir() else set()
ok = True
for label, actual in (("NIfTI", nifti), ("NPZ", npz), ("properties", pkl)):
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        print(
            f"[state] fold {fold_text} {label} set mismatch: "
            f"actual={len(actual)} expected={len(expected)} "
            f"missing={missing[:10]} extra={extra[:10]}",
            file=sys.stderr,
        )
        ok = False
if not ok:
    raise SystemExit(1)
print(f"[ok] fold {fold_text} validation IDs are exact ({len(expected)} cases, NIfTI+NPZ+properties)")
PY
}

if [[ -s "$CHECKPOINT_FINAL" ]] && validation_is_exact; then
  finalize_training_lineage
  echo "[done] Dataset507 fold $FOLD is already complete"
  exit 0
fi

python - <<'PY'
import torch
print(f"[info] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available on this Slurm allocation")
PY

TRAIN_ARGS=(nnUNetv2_train "$DATASET_ID" "$CONFIGURATION" "$FOLD" -tr "$TRAINER" -p "$PLANS" --npz)
if [[ -s "$CHECKPOINT_FINAL" ]]; then
  TRAIN_ARGS+=(--val)
  echo "[state] fold $FOLD training is final; regenerating incomplete validation outputs"
elif [[ -s "$CHECKPOINT_LATEST" ]]; then
  TRAIN_ARGS+=(--c)
  echo "[state] fold $FOLD has a resumable checkpoint; continuing training"
elif [[ -s "$CHECKPOINT_BEST" ]]; then
  echo "[error] fold $FOLD has checkpoint_best.pth but no latest/final checkpoint; refusing an ambiguous resume" >&2
  exit 2
elif find "$FOLD_DIR" -maxdepth 1 -type f -name 'checkpoint_*.pth' -print -quit 2>/dev/null | grep -q .; then
  echo "[error] fold $FOLD has an empty or unrecognized checkpoint; refusing to start over it" >&2
  exit 2
else
  echo "[state] fold $FOLD has no checkpoint; starting fresh"
fi

"${TRAIN_ARGS[@]}" &
child_pid=$!
set +e
wait "$child_pid"
train_rc=$?
set -e
child_pid=""
if (( train_rc != 0 )); then
  echo "[error] nnU-Net fold $FOLD exited with status $train_rc" >&2
  exit "$train_rc"
fi

if [[ ! -s "$CHECKPOINT_FINAL" ]]; then
  echo "[error] nnU-Net returned success without a final fold $FOLD checkpoint" >&2
  exit 1
fi
validation_is_exact
finalize_training_lineage
echo "[done] Dataset507 center-group fold $FOLD Dice+TopK10 training and validation complete"
