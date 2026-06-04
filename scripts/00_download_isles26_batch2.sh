#!/bin/bash
# ISLES'26 Batch 2 — staging helper (NOT an automated download).
#
# Batch 2 download is AUTH-GATED: it requires a *verified Grand Challenge
# account* that has joined the ISLES'26 challenge. That cannot be scripted from
# the cluster (interactive browser auth). A human must download it via browser,
# then stage it here. This script only (a) documents the procedure and
# (b) validates + summarizes whatever has been staged, so the ingestion pipeline
# can pick it up. See docs/isles26_data_status.md.
#
# Manual steps (human, in a browser):
#   1. Verify your account at https://grand-challenge.org/ and JOIN
#      https://isles-26.grand-challenge.org/  (request participation).
#   2. Download the Batch 2 archive(s) from the challenge "dataset" page.
#   3. Copy/unpack them onto /scratch under:  data/isles26_batch2/
#         scp/rsync the archive to the cluster, then unpack here.
#   4. Re-run this script to validate + summarize the staged layout.
#
# Batch 1 (ATLAS v2.0, 955 scans) is ALREADY staged at
#   data/ATLAS_R2.1_raw/Training_Raw/   (and converted to nnU-Net Dataset501).
# Do NOT re-download Batch 1.

set -euo pipefail
REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"

DEST="data/isles26_batch2"

if [[ ! -d "$DEST" ]]; then
  cat <<EOF
[info] $DEST does not exist yet.
       Batch 2 has not been staged. Follow the manual steps in the header of
       this script (verified Grand Challenge account required), then place the
       unpacked data under: $REPO/$DEST
       and re-run: bash scripts/00_download_isles26_batch2.sh
EOF
  exit 0
fi

echo "[info] Inspecting staged Batch 2 under $DEST"
echo "=== top-level entries ==="
ls -1 "$DEST" | head -50
echo "=== file-type counts ==="
echo "  NIfTI (.nii/.nii.gz): $(find "$DEST" -type f \( -name '*.nii' -o -name '*.nii.gz' \) | wc -l)"
echo "  JSON (.json):         $(find "$DEST" -type f -name '*.json' | wc -l)"
echo "  CSV/TSV:              $(find "$DEST" -type f \( -name '*.csv' -o -name '*.tsv' \) | wc -l)"
echo "=== a few example NIfTI paths (to read off naming convention) ==="
find "$DEST" -type f \( -name '*.nii' -o -name '*.nii.gz' \) | head -8
echo "=== a few example JSON paths ==="
find "$DEST" -type f -name '*.json' | head -5
echo ""
echo "[next] Once the layout is known, wire it into the (pattern/config-driven)"
echo "       characterizer + nnU-Net converter and build a combined Batch1+Batch2"
echo "       dataset with stratified splits. See docs/isles26_data_status.md."
