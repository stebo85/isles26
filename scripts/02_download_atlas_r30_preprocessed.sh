#!/bin/bash
#SBATCH --job-name=atlas_r30_prep_dl
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH -p normal

# Download + decrypt + extract ATLAS R3.0 *preprocessed* training set (~14.8 GB
# encrypted) onto $SCRATCH, for a PREPROCESSING-QUALITY COMPARISON ONLY -- we do
# NOT train on it (training stays on raw native T1w so the model works on the
# raw, unprocessed inputs provided at challenge inference time; nnU-Net
# self-preprocesses internally). Runs on a compute node (no large transfers on
# the login node). Same AES-256-CBC key as the raw set (data/key.txt).
#
#   sbatch scripts/02_download_atlas_r30_preprocessed.sh

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs" "$REPO/data"

URL="https://fcp-indi.s3.us-east-1.amazonaws.com/data/Projects/INDI/ATLAS/R3.0/atlas3_training_preprocessed.tar.gz"
ENC="$REPO/data/atlas3_training_preprocessed.tar.gz"     # AES-256-CBC base64 (NOT gzip)
DEC="$REPO/data/ATLAS_R3.0_preprocessed.tar.gz"          # decrypted real gzip tarball
KEY="$REPO/data/key.txt"                                 # ATLAS key (gitignored secret)
EXPECTED_BYTES=14823858360
EXTRACT_DIR="$REPO/data"

echo "[info] downloading ATLAS R3.0 preprocessed (encrypted) -> $ENC"
wget -c --tries=5 --timeout=60 -O "$ENC" "$URL"

ACTUAL_BYTES="$(stat -c %s "$ENC")"
echo "[info] encrypted bytes=$ACTUAL_BYTES (expected $EXPECTED_BYTES)"
if [[ "$ACTUAL_BYTES" -ne "$EXPECTED_BYTES" ]]; then
  echo "[error] size mismatch; download incomplete/corrupt" >&2
  exit 2
fi

if [[ ! -s "$KEY" ]]; then
  echo "[error] missing decryption key: $KEY" >&2
  exit 2
fi
echo "[info] decrypting -> $DEC"
openssl aes-256-cbc -md sha256 -d -a -in "$ENC" -out "$DEC" -pass "file:$KEY"

echo "[info] verifying gzip integrity of decrypted tarball"
gzip -t "$DEC"

echo "[info] extracting into $EXTRACT_DIR"
tar -xzf "$DEC" -C "$EXTRACT_DIR"

echo "[info] top-level entries created under $EXTRACT_DIR:"
tar -tzf "$DEC" | awk -F/ '{print $1}' | sort -u | head

echo "[done] ATLAS R3.0 preprocessed downloaded + decrypted + extracted (comparison only)."
