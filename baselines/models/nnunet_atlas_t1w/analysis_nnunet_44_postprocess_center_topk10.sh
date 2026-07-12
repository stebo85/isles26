#!/bin/bash
#SBATCH --job-name=nnunet_postproc_center_topk10
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH -p owners

# Consolidate the five held-center validation folds and select connected-
# component postprocessing only when it improves the full corrected OOF set.

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
REPORT_DIR="$REPO/baselines/reports/nnunet_center_grouped_topk10_crossval"
mkdir -p "$REPO/logs" "$REPORT_DIR"

module purge
if [[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]]; then
  module use "$GROUP_HOME/modules"
fi
module load devel
module load python/3.12.1
module load py-pytorch/2.4.1_py312

export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONNOUSERSITE=1
export nnUNet_compile=f
export nnUNet_raw="$REPO/work/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/work/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"

VENV="$REPO/.venv-nnunet"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[error] missing $VENV" >&2
  exit 2
fi
source "$VENV/bin/activate"
if [[ ! -x "$VENV/bin/nnUNetv2_find_best_configuration" ]] || \
   [[ "$(readlink -f "$(command -v nnUNetv2_find_best_configuration)")" != "$(readlink -f "$VENV/bin/nnUNetv2_find_best_configuration")" ]]; then
  echo "[error] nnUNetv2_find_best_configuration does not resolve inside $VENV" >&2
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
TRAIN_SCRIPT="$REPO/baselines/models/nnunet_atlas_t1w/analysis_nnunet_43_train_center_topk10.sh"
RESULT_DIR="$nnUNet_results/$DATASET_NAME/${TRAINER}__${PLANS}__${CONFIGURATION}"
CV_DIR="$RESULT_DIR/crossval_results_folds_0_1_2_3_4"

for required in \
  "$MANIFEST" \
  "$SPLITS" \
  "$SPLIT_SUMMARY" \
  "$DATASET_LINEAGE" \
  "$TRACKED_SPLITS" \
  "$TRACKED_LINEAGE" \
  "$VERIFIER" \
  "$TRAIN_SCRIPT"; do
  if [[ ! -s "$required" ]]; then
    echo "[error] missing required input: $required" >&2
    exit 2
  fi
done
if ! cmp -s "$SPLITS" "$TRACKED_SPLITS"; then
  echo "[error] installed splits differ from the reviewed center-group split" >&2
  exit 2
fi

tmp_lineage="$(mktemp "${SLURM_TMPDIR:-/tmp}/nnunet507_lineage.XXXXXX")"
trap 'rm -f "$tmp_lineage"' EXIT
rm -f "$tmp_lineage"
python "$VERIFIER" \
  --raw-dir "$RAW_DIR" \
  --preprocessed-dir "$PREPROCESSED_DIR" \
  --manifest "$MANIFEST" \
  --splits "$SPLITS" \
  --split-summary "$SPLIT_SUMMARY" \
  --expected-dataset-name "$DATASET_NAME" \
  --expected-cases 1452 \
  --expected-assignment-sha "$ASSIGNMENT_SHA" \
  --out-lineage "$tmp_lineage"
if ! cmp -s "$tmp_lineage" "$DATASET_LINEAGE" || ! cmp -s "$DATASET_LINEAGE" "$TRACKED_LINEAGE"; then
  echo "[error] Dataset507 does not match its immutable lineage" >&2
  exit 2
fi

python - "$SPLITS" "$DATASET_LINEAGE" "$SPLIT_SUMMARY" "$TRAIN_SCRIPT" "$RESULT_DIR" \
  "$DATASET_NAME" "$TRAINER" "$PLANS" "$CONFIGURATION" "$ASSIGNMENT_SHA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    splits_path,
    dataset_lineage_path,
    split_summary_path,
    train_script_path,
    result_dir_text,
    dataset_name,
    trainer,
    plans,
    configuration,
    assignment_sha,
) = sys.argv[1:]

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

splits = json.loads(Path(splits_path).read_text())
if not isinstance(splits, list) or len(splits) != 5:
    raise SystemExit("expected exactly five folds")
result_dir = Path(result_dir_text)
fixed = {
    "dataset_id": 507,
    "dataset_name": dataset_name,
    "configuration": configuration,
    "trainer": trainer,
    "plans": plans,
    "assignment_input_sha256": assignment_sha,
    "dataset_lineage_sha256": sha256(dataset_lineage_path),
    "splits_final_sha256": sha256(splits_path),
    "splits_summary_sha256": sha256(split_summary_path),
    "training_script_sha256": sha256(train_script_path),
}
failures = []
for fold, split in enumerate(splits):
    fold_dir = result_dir / f"fold_{fold}"
    checkpoint = fold_dir / "checkpoint_final.pth"
    lineage_path = fold_dir / "training_lineage.json"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        failures.append(f"fold {fold}: missing final checkpoint")
        continue
    if not lineage_path.is_file():
        failures.append(f"fold {fold}: missing training lineage")
        continue
    lineage = json.loads(lineage_path.read_text())
    for key, expected in fixed.items():
        if lineage.get(key) != expected:
            failures.append(f"fold {fold}: lineage {key} mismatch")
    if lineage.get("fold") != fold:
        failures.append(f"fold {fold}: lineage fold mismatch")
    if lineage.get("train_case_ids") != split["train"]:
        failures.append(f"fold {fold}: lineage train IDs mismatch")
    if lineage.get("validation_case_ids") != split["val"]:
        failures.append(f"fold {fold}: lineage validation IDs mismatch")
    certificate = lineage.get("checkpoint_final")
    if not isinstance(certificate, dict):
        failures.append(f"fold {fold}: missing final checkpoint certificate")
    else:
        expected_certificate = {
            "sha256": sha256(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
        }
        if certificate != expected_certificate:
            failures.append(f"fold {fold}: final checkpoint certificate mismatch")

    expected_ids = set(split["val"])
    val_dir = fold_dir / "validation"
    nifti_ids = {p.name[:-7] for p in val_dir.glob("*.nii.gz")}
    npz_ids = {p.stem for p in val_dir.glob("*.npz")}
    pkl_ids = {p.stem for p in val_dir.glob("*.pkl")}
    for label, actual in (("NIfTI", nifti_ids), ("NPZ", npz_ids), ("properties", pkl_ids)):
        if actual != expected_ids:
            failures.append(
                f"fold {fold}: {label} IDs mismatch "
                f"(actual={len(actual)}, expected={len(expected_ids)}, "
                f"missing={sorted(expected_ids - actual)[:5]}, extra={sorted(actual - expected_ids)[:5]})"
            )
if failures:
    raise SystemExit("training result verification failed:\n  " + "\n  ".join(failures))
print("[ok] all five final checkpoints, training lineages, and validation case sets verified")
PY

nnUNetv2_find_best_configuration "$DATASET_ID" \
  -c "$CONFIGURATION" \
  -tr "$TRAINER" \
  -p "$PLANS" \
  --disable_ensembling \
  -np "${SLURM_CPUS_PER_TASK:-8}"

python - "$SPLITS" "$CV_DIR" <<'PY'
import json
import sys
from pathlib import Path

splits_path, cv_dir_text = sys.argv[1:]
expected = set()
for split in json.loads(Path(splits_path).read_text()):
    overlap = expected.intersection(split["val"])
    if overlap:
        raise SystemExit(f"duplicate OOF validation IDs: {sorted(overlap)[:10]}")
    expected.update(split["val"])
if len(expected) != 1452:
    raise SystemExit(f"expected 1452 unique OOF IDs, found {len(expected)}")
for label, directory in (
    ("raw", Path(cv_dir_text)),
    ("postprocessed", Path(cv_dir_text) / "postprocessed"),
):
    actual = {path.name[:-7] for path in directory.glob("*.nii.gz")} if directory.is_dir() else set()
    if actual != expected:
        raise SystemExit(
            f"{label} OOF IDs mismatch: actual={len(actual)} expected={len(expected)} "
            f"missing={sorted(expected - actual)[:10]} extra={sorted(actual - expected)[:10]}"
        )
    summary = directory / "summary.json"
    if not summary.is_file():
        raise SystemExit(f"missing {label} summary: {summary}")
    score = json.loads(summary.read_text())["foreground_mean"]["Dice"]
    print(f"[ok] {label} OOF IDs exact ({len(actual)} cases); foreground Dice={score:.4f}")
PY

cp -f "$CV_DIR/summary.json" "$REPORT_DIR/crossval_summary.json"
cp -f "$CV_DIR/postprocessed/summary.json" "$REPORT_DIR/crossval_postprocessed_summary.json"
cp -f "$CV_DIR/postprocessing.json" "$REPORT_DIR/postprocessing.json"
cp -f "$CV_DIR/postprocessing.pkl" "$REPORT_DIR/postprocessing.pkl"
if [[ -s "$nnUNet_results/$DATASET_NAME/inference_instructions.txt" ]]; then
  cp -f "$nnUNet_results/$DATASET_NAME/inference_instructions.txt" "$REPORT_DIR/inference_instructions.txt"
fi
echo "[done] Dataset507 center-group TopK cross-validation consolidation complete"
