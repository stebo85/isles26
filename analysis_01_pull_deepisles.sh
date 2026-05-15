#!/bin/bash
#SBATCH --job-name=pull_deepisles
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH -p owners

# Pull the DeepISLES Docker image and convert to Apptainer SIF.
# Idempotent: skips if the SIF already exists.

set -euo pipefail

SIF=/scratch/users/sciget/isles26challenge/work/containers/deepisles.sif
mkdir -p "$(dirname "$SIF")"

if [[ -f "$SIF" ]]; then
  echo "[ok] SIF already exists: $SIF"
  exit 0
fi

cd /scratch/users/sciget/isles26challenge/work/containers
# Use a scratch tmp directory for layer extraction (default /tmp is small).
export APPTAINER_TMPDIR=/scratch/users/sciget/isles26challenge/work/containers/tmp
export APPTAINER_CACHEDIR=/scratch/users/sciget/isles26challenge/work/containers/cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

echo "[info] pulling docker://isleschallenge/deepisles ..."
# Retry up to 5 times — apptainer pull from Docker Hub regularly fails with
# transient stream errors on large images. Cached layers are reused on retry.
attempts=0
max_attempts=5
until apptainer pull --force "$SIF" docker://isleschallenge/deepisles:latest; do
  attempts=$((attempts + 1))
  if (( attempts >= max_attempts )); then
    echo "[error] pull failed after $attempts attempts" >&2
    exit 1
  fi
  echo "[warn] pull attempt $attempts failed; retrying in 30s ..."
  sleep 30
done

echo "[done] $SIF -> $(stat -c%s "$SIF") bytes"
