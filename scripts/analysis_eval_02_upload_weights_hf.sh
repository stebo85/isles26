#!/bin/bash
#SBATCH --job-name=upload_weights_hf
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH -p owners
#
# Publish the submission weights to a PUBLIC Hugging Face model repo, so the
# GitHub Actions container build can pull them with no credentials at all.
#
# HOW TO RUN (the token must never appear on a command line or in a committed
# file -- on a shared cluster, sbatch arguments are visible in job metadata):
#
#     read -rs HF_TOKEN && export HF_TOKEN
#     <paste the token, press enter>
#     sbatch --export=ALL,HF_TOKEN scripts/analysis_eval_02_upload_weights_hf.sh
#
# `--export=ALL,HF_TOKEN` passes the variable by NAME, not by value, so the
# secret is not recorded in the job record. Rotate the token afterwards if it has
# ever been pasted anywhere it could be logged.
#
# What is uploaded: the stripped inference-only checkpoints from
# analysis_nnunet_50_export_submission_weights.sh -- network weights and init
# args only, no optimizer state (596 MB for 5 folds rather than 1.25 GB), tensor
# identity already verified against the originals at export time. A manifest of
# per-file sha256 sums is generated and uploaded alongside; CI and the Dockerfile
# both re-verify against it, because a truncated download would otherwise yield a
# container that segments plausible nonsense.

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"; mkdir -p "$REPO/logs"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[error] HF_TOKEN is not set in the job environment." >&2
  echo "        read -rs HF_TOKEN && export HF_TOKEN" >&2
  echo "        sbatch --export=ALL,HF_TOKEN $0" >&2
  exit 2
fi

module purge
[[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]] && module use "$GROUP_HOME/modules"
module load devel
module load python/3.12.1

export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="${SCRATCH:-$REPO/work}/.pip-cache"
export HF_HOME="${SCRATCH:-$REPO/work}/.hf-home"   # keep the HF cache off $HOME (15 GB quota)
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME"

VENV="$REPO/.venv-hf"
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  "$VENV/bin/python" -m pip install --quiet "huggingface_hub>=0.34"
fi

DATASET="${DATASET:-Dataset507_ATLASR30_CORRECTED}"
SRC="$REPO/work/submission_model/$DATASET"
HF_REPO="${HF_REPO:-sbollmann/isles26-nnunet-d507-topk10}"

if [[ ! -d "$SRC" ]]; then
  echo "[error] weights not staged at $SRC" >&2
  echo "        run analysis_nnunet_50_export_submission_weights.sh first" >&2
  exit 2
fi

"$VENV/bin/python" - "$SRC" "$HF_REPO" "$DATASET" <<'PYEOF'
import hashlib, json, os, sys
from pathlib import Path
from huggingface_hub import HfApi

src, repo_id, dataset = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
api = HfApi(token=os.environ["HF_TOKEN"])

manifest = {"dataset": dataset, "files": {}}
for p in sorted(src.rglob("*")):
    if p.is_file() and p.name not in ("hf_manifest.json", "README.md"):
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        manifest["files"][str(p.relative_to(src))] = {
            "sha256": h.hexdigest(), "bytes": p.stat().st_size}
manifest["total_bytes"] = sum(v["bytes"] for v in manifest["files"].values())
(src / "hf_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"[manifest] {len(manifest['files'])} files, {manifest['total_bytes']/1e6:.1f} MB")

(src / "README.md").write_text(f"""---
license: apache-2.0
tags: [medical-imaging, image-segmentation, stroke, nnunet, isles]
library_name: nnunetv2
---

# ISLES'26 stroke lesion segmentation - nnU-Net v2 (Dataset507, Dice+TopK10)

Inference-only weights for the ISLES'26 submission container. 5-fold 3d_fullres
nnU-Net v2 trained on the ISLES'26 public release (ATLAS R3.0 derived, 1452
cases) with a **center-grouped** split: every fold holds out whole acquisition
centers, which is the shift the hidden test set actually poses.

## Contents

    plans.json, dataset.json
    fold_0..fold_4/checkpoint_final.pth
    hf_manifest.json    # sha256 of every file; re-verified in CI and at image build

Checkpoints are stripped to `network_weights`, `init_args`, `trainer_name` and
`inference_allowed_mirroring_axes`; optimizer state removed, tensor identity
verified against the originals at export.

## Measured performance

Out-of-fold on the center-held-out split (n=1452), scored with the ISLES'26
organizers' own evaluation code, at the shipped operating point (threshold 0.35,
min connected component 20 voxels, 26-connectivity):

| metric | value |
|---|---:|
| Dice | 0.6283 |
| absolute volume difference | 6.37 mL |
| absolute lesion count difference | 1.87 |
| lesion-wise F1 (panoptica RQ, one-to-one IoU>=0.25) | 0.5791 |
| PR-AUC (soft map) | 0.7417 |

Caveats worth carrying: on the center-LEAKING random-fold split the same model
scores Dice 0.6558, so 0.628 is the honest figure. The volume difference above is
~0.4 mL optimistic because it was measured on stored validation softmax rather
than the deployment path.

## Usage

Consumed by the ISLES'26 submission container, which handles native-space
orientation round-tripping and emits both a binary mask and a float32
probability map.
""")

api.upload_folder(folder_path=str(src), repo_id=repo_id, repo_type="model",
                  commit_message=f"Publish inference weights for {dataset}")
print(f"[done] https://huggingface.co/{repo_id}")
PYEOF

echo "[done] weights published; the container build can now pull them with no credentials"
