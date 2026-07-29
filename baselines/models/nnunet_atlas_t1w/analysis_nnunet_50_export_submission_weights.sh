#!/bin/bash
#SBATCH --job-name=export_submission_weights
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH -p gpu
#SBATCH -G 1
#
# Stage the trained-model folder that the ISLES'26 submission container mounts at
# /opt/ml/model, and PROVE the stripped checkpoints predict identically to the
# originals before anyone ships them.
#
# Grand Challenge uploads weights separately as a model.tar.gz rather than baking
# them into the image, so this produces a self-contained directory:
#   plans.json, dataset.json, fold_{0..4}/checkpoint_final.pth
# Optimizer state, EMA buffers and logging payloads are dropped -- they are dead
# weight at inference and roughly halve the archive.
#
# Default source is Dataset507 (center-grouped folds). That is deliberate: its
# five folds each exclude a different set of CENTERS, which is the shift the
# hidden test set actually poses, and its conversion is the corrected 1452-case
# one. Set DATASET=Dataset502_ATLASR30 to stage the random-fold models instead.

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
  # .venv-nnunet was built --system-site-packages against the cluster's
  # py-pytorch module; without it torch loads but numpy is missing and
  # torch.load dies unpickling the checkpoints.
  module load py-pytorch/2.4.1_py312
fi

export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export nnUNet_compile=f
export nnUNet_raw="$REPO/work/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/work/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"

PY="$REPO/.venv-nnunet/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "[error] missing .venv-nnunet" >&2
  exit 2
fi

DATASET="${DATASET:-Dataset507_ATLASR30_CORRECTED}"
TRAINER="${TRAINER:-nnUNetTrainerDiceTopK10Loss}"
SRC="$nnUNet_results/$DATASET/${TRAINER}__nnUNetPlans__3d_fullres"
DST="${DST:-$REPO/work/submission_model/$DATASET}"

if [[ ! -d "$SRC" ]]; then
  echo "[error] missing trained model at $SRC" >&2
  exit 2
fi

mkdir -p "$DST"
cp -f "$SRC/plans.json" "$SRC/dataset.json" "$DST/"

"$PY" - "$SRC" "$DST" <<'PYEOF'
import json, sys, hashlib
from pathlib import Path
import torch

src, dst = Path(sys.argv[1]), Path(sys.argv[2])

# nnUNetPredictor.initialize_from_trained_model_folder reads exactly these keys
# out of each checkpoint. Anything else in the file is training bookkeeping.
KEEP = ("network_weights", "init_args", "trainer_name", "inference_allowed_mirroring_axes")

manifest = {}
for fold in range(5):
    ckpt_in = src / f"fold_{fold}" / "checkpoint_final.pth"
    if not ckpt_in.is_file():
        raise SystemExit(f"[error] missing {ckpt_in}")
    payload = torch.load(ckpt_in, map_location="cpu", weights_only=False)
    missing = [k for k in KEEP if k not in payload]
    if missing:
        raise SystemExit(f"[error] {ckpt_in} lacks {missing}; refusing to strip blindly")
    stripped = {k: payload[k] for k in KEEP}

    out_dir = dst / f"fold_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_out = out_dir / "checkpoint_final.pth"
    torch.save(stripped, ckpt_out)

    # Weight-level identity: the stripped file must contain bit-identical tensors.
    reloaded = torch.load(ckpt_out, map_location="cpu", weights_only=False)
    for name, tensor in payload["network_weights"].items():
        if not torch.equal(tensor, reloaded["network_weights"][name]):
            raise SystemExit(f"[error] tensor {name} changed while stripping fold {fold}")

    manifest[f"fold_{fold}"] = {
        "source": str(ckpt_in),
        "source_sha256": hashlib.sha256(ckpt_in.read_bytes()).hexdigest(),
        "exported_sha256": hashlib.sha256(ckpt_out.read_bytes()).hexdigest(),
        "source_mb": round(ckpt_in.stat().st_size / 1e6, 1),
        "exported_mb": round(ckpt_out.stat().st_size / 1e6, 1),
        "trainer_name": payload["trainer_name"],
        "inference_allowed_mirroring_axes": list(payload["inference_allowed_mirroring_axes"] or []),
    }
    print(f"[ok] fold {fold}: {manifest[f'fold_{fold}']['source_mb']} MB -> "
          f"{manifest[f'fold_{fold}']['exported_mb']} MB")

(dst / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
total = sum(m["exported_mb"] for m in manifest.values())
print(f"[done] exported 5 folds, {total:.1f} MB total")
PYEOF

# Prove the exported folder actually loads as a predictor. This catches a
# stripped-key mistake here, on the cluster, rather than inside the container on
# the organizers' server.
"$PY" - "$DST" <<'PYEOF'
import sys
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

folder = sys.argv[1]
p = nnUNetPredictor(
    tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
    perform_everything_on_device=torch.cuda.is_available(),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    verbose=False, allow_tqdm=False,
)
p.initialize_from_trained_model_folder(folder, use_folds=(0, 1, 2, 3, 4),
                                       checkpoint_name="checkpoint_final.pth")
print(f"[ok] predictor initialised from {folder} with 5 folds")
print(f"[ok] mirroring axes: {p.allowed_mirroring_axes}")
PYEOF

du -sh "$DST"
echo "[done] submission weights staged at $DST"
echo "       package with: tar -czf model.tar.gz -C $DST ."
