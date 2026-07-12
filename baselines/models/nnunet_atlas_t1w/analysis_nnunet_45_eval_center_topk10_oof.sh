#!/bin/bash
#SBATCH --job-name=nnunet_eval_center_topk10
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH -p owners

# Run the repository's rich segmentation evaluation on both raw and selected-
# postprocessing OOF masks from the five held-center folds.

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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONNOUSERSITE=1
export nnUNet_raw="$REPO/work/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/work/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"
EVAL_WORKERS="${EVAL_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

EVAL_PY="$REPO/.venv-eval/bin/python"
if [[ ! -x "$EVAL_PY" ]]; then
  echo "[error] missing .venv-eval" >&2
  exit 2
fi

DATASET_NAME=Dataset507_ATLASR30_CORRECTED
TRAINER=nnUNetTrainerDiceTopK10Loss
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
PER_SESSION="$REPO/baselines/reports/training_data_characterization_r30/per_session.csv"
RESULT_DIR="$nnUNet_results/$DATASET_NAME/${TRAINER}__nnUNetPlans__3d_fullres"
CV_DIR="$RESULT_DIR/crossval_results_folds_0_1_2_3_4"
POSTPROCESSED_DIR="$CV_DIR/postprocessed"
REPORT_DIR="$REPO/baselines/reports/nnunet_center_grouped_topk10_crossval"
RAW_REPORT_DIR="$REPO/baselines/reports/nnunet_center_grouped_topk10_crossval_raw"

for required in \
  "$MANIFEST" \
  "$SPLITS" \
  "$SPLIT_SUMMARY" \
  "$DATASET_LINEAGE" \
  "$TRACKED_SPLITS" \
  "$TRACKED_LINEAGE" \
  "$VERIFIER" \
  "$PER_SESSION" \
  "$CV_DIR/summary.json" \
  "$POSTPROCESSED_DIR/summary.json" \
  "$CV_DIR/postprocessing.json"; do
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
"$EVAL_PY" "$VERIFIER" \
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

"$EVAL_PY" - "$SPLITS" "$CV_DIR" "$POSTPROCESSED_DIR" <<'PY'
import json
import sys
from pathlib import Path

splits_path, raw_dir_text, post_dir_text = sys.argv[1:]
expected = set()
folds = json.loads(Path(splits_path).read_text())
if not isinstance(folds, list) or len(folds) != 5:
    raise SystemExit("expected exactly five folds")
for fold in folds:
    overlap = expected.intersection(fold["val"])
    if overlap:
        raise SystemExit(f"duplicate validation IDs: {sorted(overlap)[:10]}")
    expected.update(fold["val"])
if len(expected) != 1452:
    raise SystemExit(f"expected 1452 validation IDs, found {len(expected)}")
for label, directory in (("raw", Path(raw_dir_text)), ("postprocessed", Path(post_dir_text))):
    actual = {path.name[:-7] for path in directory.glob("*.nii.gz")} if directory.is_dir() else set()
    if actual != expected:
        raise SystemExit(
            f"{label} prediction IDs mismatch: actual={len(actual)} expected={len(expected)} "
            f"missing={sorted(expected - actual)[:10]} extra={sorted(actual - expected)[:10]}"
        )
    print(f"[ok] {label} OOF prediction IDs are exact ({len(actual)} cases)")
PY

run_eval() {
  local pred_dir="$1"
  local out_dir="$2"
  local title="$3"
  mkdir -p "$out_dir"
  PYTHONPATH="$REPO" "$EVAL_PY" \
    "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/run_atlas_eval.py" \
    --manifest "$MANIFEST" \
    --splits "$SPLITS" \
    --folds all \
    --pred-dir "$pred_dir" \
    --per-session-csv "$PER_SESSION" \
    --out-dir "$out_dir" \
    --title "$title" \
    --workers "$EVAL_WORKERS"
}

run_eval \
  "$CV_DIR" \
  "$RAW_REPORT_DIR" \
  "nnU-Net corrected R3.0 Dice+TopK10: held-center 5-fold OOF (raw)"
run_eval \
  "$POSTPROCESSED_DIR" \
  "$REPORT_DIR" \
  "nnU-Net corrected R3.0 Dice+TopK10: held-center 5-fold OOF (postprocessed)"

"$EVAL_PY" - "$SPLITS" "$RAW_REPORT_DIR/report.json" "$REPORT_DIR/report.json" <<'PY'
import json
import sys
from pathlib import Path

splits_path, raw_report_path, post_report_path = sys.argv[1:]
expected = {case_id for fold in json.loads(Path(splits_path).read_text()) for case_id in fold["val"]}
for label, report_path in (("raw", raw_report_path), ("postprocessed", post_report_path)):
    report = json.loads(Path(report_path).read_text())
    actual = report.get("atlas_val_cases", [])
    if len(actual) != len(expected) or set(actual) != expected:
        raise SystemExit(f"{label} report case IDs do not match the five validation folds")
    if report.get("n_cases") != len(expected):
        raise SystemExit(f"{label} report n_cases is not {len(expected)}")
    print(f"[ok] {label} report covers the exact {len(expected)}-case OOF set")
PY

"$EVAL_PY" - "$SPLITS" "$RAW_REPORT_DIR" "$REPORT_DIR" <<'PY'
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

splits_path = Path(sys.argv[1])
report_directories = (("raw", Path(sys.argv[2])), ("postprocessed", Path(sys.argv[3])))
splits = json.loads(splits_path.read_text())
case_to_fold = {}
for fold, split in enumerate(splits):
    for case_id in split["val"]:
        if case_id in case_to_fold:
            raise SystemExit(f"case occurs in multiple validation folds: {case_id}")
        case_to_fold[case_id] = fold

metrics = (
    "dice",
    "lesion_f1",
    "surface_dice_3mm",
    "hd95_mm",
    "abs_volume_diff_ml",
)


def finite_values(rows, metric):
    values = []
    for row in rows:
        value = row.get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            values.append(float(value))
    return values


def distribution(values):
    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


for label, report_directory in report_directories:
    report = json.loads((report_directory / "report.json").read_text())
    rows = report.get("per_case")
    if not isinstance(rows, list) or len(rows) != len(case_to_fold):
        raise SystemExit(f"{label} report has an invalid per_case payload")

    rows_by_fold = defaultdict(list)
    rows_by_center = defaultdict(list)
    for row in rows:
        case_id = row.get("subject_id")
        center = row.get("center")
        if case_id not in case_to_fold or not isinstance(center, str) or not center:
            raise SystemExit(f"{label} report row lacks a certified case/fold/center mapping")
        rows_by_fold[case_to_fold[case_id]].append(row)
        rows_by_center[center].append(row)
    if set(rows_by_fold) != set(range(5)) or len(rows_by_center) != 55:
        raise SystemExit(
            f"{label} report grouping mismatch: folds={sorted(rows_by_fold)}, "
            f"centers={len(rows_by_center)}"
        )

    metric_summary = {}
    for metric in metrics:
        center_means = []
        for center_rows in rows_by_center.values():
            values = finite_values(center_rows, metric)
            if values:
                center_means.append(statistics.fmean(values))
        fold_means = []
        for fold in range(5):
            values = finite_values(rows_by_fold[fold], metric)
            if values:
                fold_means.append(statistics.fmean(values))
        metric_summary[metric] = {
            "case_pooled": distribution(finite_values(rows, metric)),
            "center_macro": distribution(center_means),
            "fold_mean_distribution": distribution(fold_means),
            "means_by_fold": fold_means,
        }

    fold_summary = []
    for fold in range(5):
        fold_rows = rows_by_fold[fold]
        fold_summary.append(
            {
                "fold": fold,
                "n_cases": len(fold_rows),
                "n_centers": len({row["center"] for row in fold_rows}),
                "metrics": {
                    metric: distribution(finite_values(fold_rows, metric))
                    for metric in metrics
                },
            }
        )

    summary = {
        "schema_version": 1,
        "evaluation": label,
        "n_cases": len(rows),
        "n_centers": len(rows_by_center),
        "n_folds": len(rows_by_fold),
        "metrics": metric_summary,
        "folds": fold_summary,
    }
    (report_directory / "generalization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        f"# Held-center generalization summary ({label})",
        "",
        "| metric | case-pooled mean | center-macro mean | fold means (mean +/- SD) | fold range |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in metrics:
        values = metric_summary[metric]
        pooled = values["case_pooled"]["mean"]
        center_macro = values["center_macro"]["mean"]
        fold_distribution = values["fold_mean_distribution"]
        lines.append(
            f"| {metric} | {pooled:.4f} | {center_macro:.4f} | "
            f"{fold_distribution['mean']:.4f} +/- {fold_distribution['std']:.4f} | "
            f"{fold_distribution['min']:.4f} to {fold_distribution['max']:.4f} |"
        )
    lines.extend(
        [
            "",
            "| fold | cases | centers | Dice mean | lesion F1 mean | surface Dice 3mm mean |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for fold in fold_summary:
        lines.append(
            f"| {fold['fold']} | {fold['n_cases']} | {fold['n_centers']} | "
            f"{fold['metrics']['dice']['mean']:.4f} | "
            f"{fold['metrics']['lesion_f1']['mean']:.4f} | "
            f"{fold['metrics']['surface_dice_3mm']['mean']:.4f} |"
        )
    (report_directory / "generalization_summary.md").write_text("\n".join(lines) + "\n")
    print(
        f"[ok] {label} center-macro/fold summary: "
        f"Dice={metric_summary['dice']['center_macro']['mean']:.4f}, "
        f"centers={len(rows_by_center)}"
    )
PY

"$EVAL_PY" - "$RAW_REPORT_DIR/meta.json" "$REPORT_DIR/meta.json" <<'PY'
import json
import sys
from pathlib import Path

common = {
    "model_family": "nnunet",
    "eval_set": "atlas_r30_center_group_5fold_cv",
    "input_modalities": ["T1w"],
    "trained_on": "Corrected ATLAS R3.0 / ISLES'26 public training release, Dataset507_ATLASR30_CORRECTED",
    "status": "complete",
    "pipeline_dir": "baselines/models/nnunet_atlas_t1w",
}
raw = dict(common)
raw.update({
    "method": "nnunet_r30_corrected_center_group_topk10_cv_raw",
    "display_name": "nnU-Net corrected R3.0 TopK10, held-center CV (raw)",
    "notes": "Five folds hold out complete centers; raw nnU-Net validation masks.",
})
post = dict(common)
post.update({
    "method": "nnunet_r30_corrected_center_group_topk10_cv",
    "display_name": "nnU-Net corrected R3.0 TopK10, held-center CV",
    "notes": "Five folds hold out complete centers; nnU-Net-selected postprocessing.",
})
for path, payload in ((sys.argv[1], raw), (sys.argv[2], post)):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

cp -f "$CV_DIR/summary.json" "$REPORT_DIR/crossval_summary.json"
cp -f "$POSTPROCESSED_DIR/summary.json" "$REPORT_DIR/crossval_postprocessed_summary.json"
cp -f "$CV_DIR/postprocessing.json" "$REPORT_DIR/postprocessing.json"
cp -f "$CV_DIR/summary.json" "$RAW_REPORT_DIR/crossval_summary.json"
if [[ -s "$nnUNet_results/$DATASET_NAME/inference_instructions.txt" ]]; then
  cp -f "$nnUNet_results/$DATASET_NAME/inference_instructions.txt" "$REPORT_DIR/inference_instructions.txt"
fi

PYTHONPATH="$REPO" "$EVAL_PY" "$REPO/baselines/compare/compare_models.py"

"$EVAL_PY" - "$RAW_REPORT_DIR/report.json" "$REPORT_DIR/report.json" <<'PY'
import json
import sys

for label, path in (("raw", sys.argv[1]), ("postprocessed", sys.argv[2])):
    report = json.loads(open(path).read())
    metrics = {row["metric"]: row for row in report["overall"]}
    print(f"[done] {label} held-center OOF rich evaluation")
    for metric in ("dice", "lesion_f1", "surface_dice_3mm", "hd95_mm", "abs_volume_diff_ml"):
        row = metrics.get(metric)
        if row:
            print(f"{metric}: mean={row.get('mean'):.4f} median={row.get('median'):.4f}")
PY
