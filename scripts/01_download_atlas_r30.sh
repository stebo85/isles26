#!/bin/bash
#SBATCH --job-name=atlas_r30_download
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH -p normal

# Download + extract ATLAS R3.0 training-raw (the public INDI release, ~8.2 GB)
# onto $SCRATCH. Runs on a compute node (cluster policy: no large transfers on
# the login node). Public S3, no auth required.
#
#   sbatch scripts/01_download_atlas_r30.sh
#
# ATLAS R2.1 (Batch 1, 955 scans) already lives at data/ATLAS_R2.1_raw/. R3.0 is
# the larger expansion; once extracted, inspect its layout before wiring it into
# the characterizer / nnU-Net converter (see docs/isles26_data_status.md).

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs" "$REPO/data"

URL="https://fcp-indi.s3.us-east-1.amazonaws.com/data/Projects/INDI/ATLAS/R3.0/atlas3_training_raw.tar.gz"
TARBALL="$REPO/data/atlas3_training_raw.tar.gz"
EXPECTED_BYTES=8244197870
EXTRACT_DIR="$REPO/data"

echo "[info] downloading ATLAS R3.0 training-raw -> $TARBALL"
# -c resumes a partial download; retry transient network errors.
wget -c --tries=5 --timeout=60 -O "$TARBALL" "$URL"

ACTUAL_BYTES="$(stat -c %s "$TARBALL")"
echo "[info] downloaded bytes=$ACTUAL_BYTES (expected $EXPECTED_BYTES)"
if [[ "$ACTUAL_BYTES" -ne "$EXPECTED_BYTES" ]]; then
  echo "[error] size mismatch; download incomplete/corrupt" >&2
  exit 2
fi

echo "[info] verifying gzip integrity"
gzip -t "$TARBALL"

echo "[info] extracting into $EXTRACT_DIR"
tar -xzf "$TARBALL" -C "$EXTRACT_DIR"

echo "[info] top-level entries created under $EXTRACT_DIR:"
tar -tzf "$TARBALL" | awk -F/ '{print $1}' | sort -u | head

echo "[done] ATLAS R3.0 downloaded + extracted. Inspect the new dir, then wire it"
echo "       into the characterizer + nnU-Net converter (docs/isles26_data_status.md)."
