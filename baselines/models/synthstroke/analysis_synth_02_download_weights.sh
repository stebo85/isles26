#!/bin/bash
#SBATCH --job-name=synth_download
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH -p owners

# Phase A / Step 2: download pretrained SynthStroke model weights from
# HuggingFace Hub into work/synthstroke/weights/.

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs" "$REPO/work/synthstroke/weights"

module purge
if [[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]]; then
  module use "$GROUP_HOME/modules"
fi
module load devel
module load python/3.12.1
module load py-pytorch/2.4.1_py312

export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONNOUSERSITE=1

VENV="$REPO/.venv-synthstroke"
source "$VENV/bin/activate"

export WANDB_MODE=offline
export HF_HUB_ENABLE_HF_TRANSFER=0

python - <<'PY'
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from safetensors import safe_open


REPOS = [
    "liamchalcroft/synthstroke-baseline",
    "liamchalcroft/synthstroke-synth",
    "liamchalcroft/synthstroke-synth-pseudo",
    "liamchalcroft/synthstroke-synth-plus",
    "liamchalcroft/synthstroke-qatlas",
    "liamchalcroft/synthstroke-qsynth",
]

weights_root = Path("work/synthstroke/weights")
weights_root.mkdir(parents=True, exist_ok=True)
api = HfApi()
manifest = {"models": {}}

for repo_id in REPOS:
    name = repo_id.rsplit("/", 1)[1]
    model_dir = weights_root / name
    model_dir.mkdir(parents=True, exist_ok=True)

    revision = api.model_info(repo_id).sha
    required = [model_dir / "model.safetensors", model_dir / "config.json"]
    had_required = all(path.exists() and path.stat().st_size > 0 for path in required)

    print(f"[sync] {repo_id} @ {revision} -> {model_dir}")
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(model_dir),
    )
    print(f"[{'cached' if had_required else 'downloaded'}] {name}: local files synced")

    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"{name} is missing required files: {', '.join(missing)}")

    try:
        with (model_dir / "config.json").open() as f:
            json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} has corrupt config.json: {exc}") from exc

    try:
        with safe_open(model_dir / "model.safetensors", framework="numpy") as f:
            tensor_keys = list(f.keys())
    except Exception as exc:
        raise ValueError(f"{name} has unreadable model.safetensors: {exc}") from exc
    if not tensor_keys:
        raise ValueError(f"{name} has model.safetensors with an empty tensor header")

    files = sorted(
        str(path.relative_to(model_dir))
        for path in model_dir.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(model_dir).parts
    )
    manifest["models"][name] = {
        "repo_id": repo_id,
        "commit_hash": revision,
        "files": files,
    }

manifest_path = weights_root / "manifest.json"
tmp_path = weights_root / "manifest.json.tmp"
with tmp_path.open("w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
    f.write("\n")
tmp_path.replace(manifest_path)
print(f"[done] wrote {manifest_path}")
PY
