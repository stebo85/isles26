#!/bin/bash
#SBATCH --job-name=nnunet_predict
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6|GPU_CC:8.0|GPU_CC:7.5|GPU_CC:7.0"
#SBATCH --array=0-11
#SBATCH -p owners

# Skull-strip soop_bench T1w with SynthStrip, run nnU-Net fold-0 inference,
# rigidly warp the T1w-space prediction into TRACE/DWI space, and write one
# prediction per subject under work/predictions_nnunet/.
#
# Test-time augmentation is disabled to keep this reduced baseline fast.

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs"

PREDICTION_VERSION="v2"
MASK_ROOT="$REPO/data/soop_bench/derivatives/lesion_masks"
mapfile -t SUBJECTS < <(ls -1 "$MASK_ROOT" | grep '^sub-' | sort -V)
ARRAY_MAX="${SLURM_ARRAY_TASK_MAX:-11}"
EXPECTED_SUBJECTS=$((ARRAY_MAX + 1))
if (( ${#SUBJECTS[@]} != EXPECTED_SUBJECTS )); then
  echo "[error] found ${#SUBJECTS[@]} soop_bench subjects under $MASK_ROOT, expected $EXPECTED_SUBJECTS from SLURM_ARRAY_TASK_MAX=$ARRAY_MAX" >&2
  exit 2
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#SUBJECTS[@]} )); then
  echo "[error] SLURM_ARRAY_TASK_ID=$TASK_ID outside valid range 0-$ARRAY_MAX" >&2
  exit 2
fi
SUB="${SUBJECTS[$TASK_ID]}"

PREDS="$REPO/work/predictions_nnunet"
TMP_ROOT="$REPO/work/predictions_nnunet_tmp"
BRAIN_DIR="$REPO/work/soop_bench_t1w_brain"
QC_DIR="$REPO/work/qc_t1_to_trace"
RUN_DIR="$TMP_ROOT/${SUB}_run"
REG_DIR="$TMP_ROOT/$SUB"
mkdir -p "$PREDS" "$TMP_ROOT" "$BRAIN_DIR" "$QC_DIR" "$REG_DIR"

module purge
if [[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]]; then
  module use "$GROUP_HOME/modules"
fi
module load devel
module load python/3.12.1
module load py-pytorch/2.4.1_py312
module load freesurfer/8.1.0
module load ants/2.6.0

export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONNOUSERSITE=1
export nnUNet_raw="$REPO/work/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/work/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"

EVAL_PY="$REPO/.venv-eval/bin/python"
if [[ ! -x "$EVAL_PY" ]]; then
  echo "[error] missing $EVAL_PY; build the eval venv first" >&2
  exit 2
fi

VENV="$REPO/.venv-nnunet"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[error] missing $VENV; run analysis_nnunet_01_prepare_dataset.sh first" >&2
  exit 2
fi
source "$VENV/bin/activate"
python "$REPO/baselines/analyses/nnunet_helpers/install_trainer_variant.py"

RESULT_FOLD="$nnUNet_results/Dataset501_ATLASR21/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_0"
if [[ -s "$RESULT_FOLD/checkpoint_final.pth" ]]; then
  CHECKPOINT="$RESULT_FOLD/checkpoint_final.pth"
elif [[ -s "$RESULT_FOLD/checkpoint_latest.pth" ]]; then
  CHECKPOINT="$RESULT_FOLD/checkpoint_latest.pth"
else
  echo "[error] no fold-0 checkpoint_final.pth or checkpoint_latest.pth found under $RESULT_FOLD" >&2
  exit 2
fi
MODEL_SHA="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"

FINAL_PRED="$PREDS/${SUB}.nii.gz"
SIDECAR="$PREDS/${SUB}.json"
AFFINE_SENTINEL="$REG_DIR/0GenericAffine.mat"

sidecar_matches() {
  "$EVAL_PY" - "$1" "$2" "$PREDICTION_VERSION" <<'PY'
import json
import sys
from pathlib import Path

sidecar = Path(sys.argv[1])
expected_sha = sys.argv[2]
expected_version = sys.argv[3]
try:
    payload = json.loads(sidecar.read_text())
except Exception:
    raise SystemExit(1)
ok = (
    payload.get("model_checkpoint_sha256") == expected_sha
    and payload.get("prediction_version") == expected_version
)
raise SystemExit(0 if ok else 1)
PY
}

validate_integer_nifti() {
  "$EVAL_PY" - "$1" <<'PY'
import sys
import numpy as np
import nibabel as nib

path = sys.argv[1]
img = nib.load(path)
shape = img.shape
dtype = np.dtype(img.get_data_dtype())
if len(shape) != 3:
    raise SystemExit(f"{path} is not 3-D: shape={shape}")
if dtype.kind not in {"i", "u", "b"}:
    raise SystemExit(f"{path} does not have integer dtype: {dtype}")
print(f"[ok] validated integer NIfTI: {path} shape={shape} dtype={dtype}")
PY
}

SRC_T1="$REPO/data/soop_bench/$SUB/anat/${SUB}_T1w.nii.gz"
TRACE_FIXED="$REPO/data/soop_bench/$SUB/dwi/${SUB}_rec-TRACE_dwi.nii.gz"
TRACE_REF3D="$REG_DIR/${SUB}_TRACE_ref3d.nii.gz"
CANON_T1="$BRAIN_DIR/${SUB}_T1w_RAS.nii.gz"
BRAIN_T1="$BRAIN_DIR/${SUB}_T1w_brain.nii.gz"

if [[ ! -s "$SRC_T1" ]]; then
  echo "[error] missing source T1w: $SRC_T1" >&2
  exit 2
fi
if [[ ! -s "$TRACE_FIXED" ]]; then
  echo "[error] missing TRACE fixed image: $TRACE_FIXED" >&2
  exit 2
fi

"$EVAL_PY" - "$TRACE_FIXED" "$TRACE_REF3D" <<'PY'
from pathlib import Path
import os
import sys

import nibabel as nib
import numpy as np

src = Path(sys.argv[1])
dst = Path(sys.argv[2])


def existing_is_current(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        img = nib.load(str(path))
    except Exception:
        return False
    return len(img.shape) == 3 and nib.aff2axcodes(img.affine) == ("R", "A", "S")


if existing_is_current(dst):
    img = nib.load(str(dst))
    print(f"[ok] 3-D RAS TRACE reference exists: {dst} shape={img.shape}")
    raise SystemExit(0)

img = nib.as_closest_canonical(nib.load(str(src)))
data = np.asanyarray(img.dataobj)
while data.ndim > 3 and data.shape[-1] == 1:
    data = data[..., 0]
if data.ndim != 3:
    raise SystemExit(f"{src} is not 3-D after trailing singleton squeeze: shape={data.shape}")

out = nib.Nifti1Image(data, img.affine, img.header.copy())
out.set_data_dtype(data.dtype)
tmp = dst.with_name(dst.name + ".tmp.nii.gz")
nib.save(out, str(tmp))
if not tmp.exists() or tmp.stat().st_size == 0:
    raise SystemExit(f"failed to write non-empty 3-D TRACE reference: {tmp}")
os.replace(tmp, dst)
print(f"[done] wrote 3-D RAS TRACE reference: {dst} shape={out.shape}")
PY

if [[ -s "$FINAL_PRED" && -s "$SIDECAR" && -s "$AFFINE_SENTINEL" ]] && sidecar_matches "$SIDECAR" "$MODEL_SHA"; then
  echo "[ok] $SUB already has current TRACE-space prediction and sidecar: $FINAL_PRED"
  exit 0
fi

if [[ -e "$FINAL_PRED" || -e "$SIDECAR" ]]; then
  echo "[stale] removing old prediction/sidecar for $SUB before rerun"
  rm -f "$FINAL_PRED" "$SIDECAR"
fi

if [[ ! -s "$CANON_T1" ]]; then
  "$EVAL_PY" - "$SRC_T1" "$CANON_T1" <<'PY'
from pathlib import Path
import os
import sys

import nibabel as nib
import numpy as np

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
img = nib.as_closest_canonical(nib.load(str(src)))
data = np.asanyarray(img.dataobj)
if data.ndim == 4 and data.shape[-1] == 1:
    data = data[..., 0]
    img = nib.Nifti1Image(data, img.affine, img.header)
tmp = dst.with_name(dst.name + ".tmp.nii.gz")
nib.save(img, str(tmp))
if not tmp.exists() or tmp.stat().st_size == 0:
    raise SystemExit(f"failed to write non-empty canonical T1w: {tmp}")
os.replace(tmp, dst)
print(f"[done] canonical T1w: {dst} shape={img.shape}")
PY
fi

if [[ ! -s "$BRAIN_T1" ]]; then
  TMP_BRAIN="$BRAIN_T1.tmp.nii.gz"
  rm -f "$TMP_BRAIN"
  mri_synthstrip -i "$CANON_T1" -o "$TMP_BRAIN"
  if [[ ! -s "$TMP_BRAIN" ]]; then
    echo "[error] SynthStrip did not create a non-empty output for $SUB" >&2
    exit 2
  fi
  mv "$TMP_BRAIN" "$BRAIN_T1"
  echo "[done] skull-stripped T1w: $BRAIN_T1"
else
  echo "[ok] skull-stripped T1w exists: $BRAIN_T1"
fi

rm -rf "$RUN_DIR"
INPUT_DIR="$RUN_DIR/input"
OUTPUT_DIR="$RUN_DIR/output"
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"
STAGED_INPUT="$INPUT_DIR/${SUB}_0000.nii.gz"
cp "$BRAIN_T1" "$STAGED_INPUT.tmp"
mv "$STAGED_INPUT.tmp" "$STAGED_INPUT"

nnUNetv2_predict \
  -i "$INPUT_DIR" \
  -o "$OUTPUT_DIR" \
  -d 501 \
  -c 3d_fullres \
  -f 0 \
  -tr nnUNetTrainer_250epochs \
  --disable_tta

NNUNET_OUT="$OUTPUT_DIR/${SUB}.nii.gz"
if [[ ! -s "$NNUNET_OUT" ]]; then
  echo "[error] expected nnU-Net output not found: $NNUNET_OUT" >&2
  exit 2
fi
validate_integer_nifti "$NNUNET_OUT"

PRED_T1SPACE="$REG_DIR/${SUB}_pred_t1space.nii.gz"
cp "$NNUNET_OUT" "$PRED_T1SPACE.tmp"
mv "$PRED_T1SPACE.tmp" "$PRED_T1SPACE"

REG_PREFIX="$REG_DIR/${SUB}_t1_to_trace_ref3d_"
WARPED_T1="$REG_DIR/${SUB}_t1_in_trace.nii.gz"
AFFINE_MAT="${REG_PREFIX}0GenericAffine.mat"
if [[ ! -s "$AFFINE_MAT" || ! -s "$WARPED_T1" ]]; then
  antsRegistration --dimensionality 3 --float 1 --collapse-output-transforms 1 \
    --output "[$REG_PREFIX,$WARPED_T1]" \
    --interpolation Linear \
    --initial-moving-transform "[$TRACE_REF3D,$BRAIN_T1,1]" \
    --transform "Rigid[0.1]" \
    --metric "MI[$TRACE_REF3D,$BRAIN_T1,1,32,Regular,0.25]" \
    --convergence "[1000x500x250x100,1e-6,10]" \
    --shrink-factors 8x4x2x1 \
    --smoothing-sigmas 3x2x1x0vox
else
  echo "[ok] reusing existing T1-to-TRACE registration: $AFFINE_MAT"
fi
if [[ ! -s "$AFFINE_MAT" ]]; then
  echo "[error] expected ANTs affine not found: $AFFINE_MAT" >&2
  exit 2
fi
cp "$AFFINE_MAT" "$AFFINE_SENTINEL.tmp"
mv "$AFFINE_SENTINEL.tmp" "$AFFINE_SENTINEL"

WARPED_PRED_TMP="$REG_DIR/${SUB}_pred_trace_tmp.nii.gz"
rm -f "$WARPED_PRED_TMP"
antsApplyTransforms -d 3 \
  -i "$PRED_T1SPACE" \
  -r "$TRACE_REF3D" \
  -o "$WARPED_PRED_TMP" \
  -n GenericLabel \
  -t "$AFFINE_MAT"
validate_integer_nifti "$WARPED_PRED_TMP"

QC_PNG="$QC_DIR/${SUB}.png"
"$EVAL_PY" - "$TRACE_REF3D" "$WARPED_T1" "$QC_PNG" "$SUB" <<'PY'
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def squeeze3(img_path: str) -> np.ndarray:
    data = np.asanyarray(nib.load(img_path).dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise SystemExit(f"expected 3-D image for {img_path}, got shape {data.shape}")
    return data


def norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, (1, 99))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def take_slice(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    sl = [slice(None)] * 3
    sl[axis] = idx
    return vol[tuple(sl)]


trace_path, warped_t1_path, out_png, subject = sys.argv[1:5]
trace = squeeze3(trace_path)
warped_t1 = squeeze3(warped_t1_path)
if trace.shape != warped_t1.shape:
    raise SystemExit(f"TRACE/T1 shape mismatch after registration: {trace.shape} vs {warped_t1.shape}")

views = [("sagittal", 0), ("coronal", 1), ("axial", 2)]
fig, axes = plt.subplots(1, 3, figsize=(11, 4))
for ax, (name, axis) in zip(axes, views):
    idx = trace.shape[axis] // 2
    bg = np.rot90(norm01(take_slice(trace, axis, idx)))
    fg = np.rot90(norm01(take_slice(warped_t1, axis, idx)))
    ax.imshow(bg, cmap="gray", interpolation="nearest")
    ax.imshow(fg, cmap="magma", alpha=0.35, interpolation="nearest")
    ax.set_title(f"{name} {idx}", fontsize=9)
    ax.set_axis_off()
fig.suptitle(f"{subject}: warped T1w on TRACE", fontsize=11)
fig.tight_layout()
Path(out_png).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_png, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"[done] registration QC: {out_png}")
PY

TMP_FINAL="$PREDS/${SUB}.tmp.nii.gz"
cp "$WARPED_PRED_TMP" "$TMP_FINAL"
mv "$TMP_FINAL" "$FINAL_PRED"

SYNTHSTRIP_INFO="$(mri_synthstrip --version 2>&1 || true)"
SYNTHSTRIP_INFO="${SYNTHSTRIP_INFO%%$'\n'*}"
if [[ -z "${SYNTHSTRIP_INFO//[[:space:]]/}" ]]; then
  SYNTHSTRIP_INFO="$(command -v mri_synthstrip || true)"
fi
CREATED_AT_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
"$EVAL_PY" - "$SIDECAR" "$MODEL_SHA" "$SYNTHSTRIP_INFO" "$PREDICTION_VERSION" "$CREATED_AT_UTC" "$CHECKPOINT" <<'PY'
import json
import os
import sys
from pathlib import Path

sidecar = Path(sys.argv[1])
payload = {
    "model_checkpoint_sha256": sys.argv[2],
    "synthstrip_model_info": sys.argv[3],
    "prediction_version": sys.argv[4],
    "created_at_utc": sys.argv[5],
    "model_checkpoint_path": sys.argv[6],
}
tmp = sidecar.with_name(sidecar.name + ".tmp")
tmp.write_text(json.dumps(payload, indent=2) + "\n")
os.replace(tmp, sidecar)
PY

echo "[done] $SUB -> $FINAL_PRED"
