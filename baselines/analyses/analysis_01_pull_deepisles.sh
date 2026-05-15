#!/bin/bash
#SBATCH --job-name=pull_deepisles
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --tmp=80G
#SBATCH -p owners

# Build a directory-based (sandbox) Apptainer image for DeepISLES. The
# squashfs/SIF path on this cluster's apptainer 1.5.0 + rootless proot fails
# silently and produces a 41 KB stub SIF, even though the rootfs is correctly
# populated. The sandbox path skips squashfs entirely and was verified to be
# the issue (multiple SIF rebuilds against a clean cache all produced 41 KB).

set -euo pipefail

SANDBOX=/scratch/users/sciget/isles26challenge/work/containers/deepisles_sandbox
mkdir -p "$(dirname "$SANDBOX")"

# A valid DeepISLES rootfs is several GB. Anything smaller is a failed build.
MIN_BYTES=$((500 * 1024 * 1024))
if [[ -d "$SANDBOX" ]] && (( $(du -sb "$SANDBOX" | awk '{print $1}') > MIN_BYTES )); then
  echo "[ok] sandbox already exists: $SANDBOX ($(du -sh "$SANDBOX" | awk '{print $1}'))"
  exit 0
fi
if [[ -d "$SANDBOX" ]]; then
  echo "[warn] removing undersized sandbox: $SANDBOX"
  rm -rf "$SANDBOX"
fi

cd /scratch/users/sciget/isles26challenge/work/containers
export APPTAINER_TMPDIR="${TMPDIR:-/tmp}/apptainer"
export APPTAINER_CACHEDIR=/scratch/users/sciget/isles26challenge/work/containers/cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# Drop the broken stub SIFs cached from previous attempts so apptainer doesn't
# refuse to rebuild.
find "$APPTAINER_CACHEDIR" -path '*/oci-tmp/*' -size -100k -type f -print -delete || true

df -h "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# Pin to an immutable Docker manifest digest so future re-pulls produce the
# same image even if `:latest` is retagged. The digest below was captured on
# 2026-05-15 from `apptainer --debug pull docker://isleschallenge/deepisles`.
# To bump, run `apptainer inspect --json <sandbox>` after pulling the new tag
# and update both `IMAGE_REF` and `IMAGE_BASE_DIGEST` here.
IMAGE_REF=docker://isleschallenge/deepisles@sha256:52f7ec71c60d50b94710f04b5e4558966060a68068fed463f665880664a783a8
IMAGE_BASE_DIGEST=sha256:848c9eceb67dbc585bcb37f093389d142caeaa98878bd31039af04ef297a5af4
echo "[info] building sandbox from $IMAGE_REF ..."
attempts=0
max_attempts=5
until apptainer --debug build --sandbox --force "$SANDBOX" "$IMAGE_REF"; do
  attempts=$((attempts + 1))
  if (( attempts >= max_attempts )); then
    echo "[error] build failed after $attempts attempts" >&2
    exit 1
  fi
  echo "[warn] build attempt $attempts failed; retrying in 30s ..."
  sleep 30
done

actual=$(du -sb "$SANDBOX" | awk '{print $1}')
if (( actual <= MIN_BYTES )); then
  echo "[error] sandbox too small (${actual} bytes) — build likely failed silently" >&2
  exit 2
fi
echo "[done] $SANDBOX -> $(du -sh "$SANDBOX" | awk '{print $1}')"

# Record image metadata next to the sandbox so the benchmark report can cite
# the exact image that produced its numbers.
META="$(dirname "$SANDBOX")/deepisles_image_metadata.json"
apptainer inspect --json "$SANDBOX" > "$META.tmp"
python3 - "$META.tmp" "$META" "$IMAGE_REF" "$IMAGE_BASE_DIGEST" <<'PY'
import json, sys, datetime
inspect_path, out_path, image_ref, base_digest = sys.argv[1:5]
with open(inspect_path) as f:
    inspect = json.load(f)
out = {
    "captured_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "image_ref_pinned": image_ref,
    "image_base_digest": base_digest,
    "apptainer_inspect": inspect,
}
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
PY
rm -f "$META.tmp"
echo "[info] image metadata: $META"
echo "[info] inventory:"; ls "$SANDBOX" | head -20
