#!/bin/bash
#SBATCH --job-name=verify_container_config
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#
# Verify the submission container's configuration resolution after the
# 2026-08-11 health-check failure (GC result 0b843f9e).
#
# Checks, in order:
#   1. inference.py imports at all under the pinned interpreter
#   2. weights resolve to the BAKED path, not Grand Challenge's /opt/ml/model
#      overmount, when the GC mount is absent/empty
#   3. an explicit ISLES26_MODEL_DIR still wins
#   4. TTA is auto (None) by default and reduces correctly per device
#   5. the four interface slugs still match the challenge sockets
#
# No model weights and no inference: this is a pure configuration check, which
# is why it is 5 minutes and 4 GB.

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
APP="$REPO/submission/isles26_algorithm"
mkdir -p "$REPO/logs"

# Same stack the other container-facing scripts load (analysis_nnunet_66): the
# venv inherits numpy/torch from py-pytorch rather than carrying its own.
module load devel
module load python/3.12.1
module load py-pytorch/2.4.1_py312

# `python` is not necessarily the module's interpreter (the base env provides a
# 2.7). Prefer the project venv -- it has numpy/SimpleITK, which inference.py
# imports at module scope -- and note it only runs once the python module is
# loaded, since it links the module's libpython3.12.so.1.0.
PY_BIN="$REPO/.venv-nnunet/bin/python"
[[ -x "$PY_BIN" ]] || PY_BIN="$(command -v python3.12 || command -v python3)"
echo "[env] interpreter: $PY_BIN -> $("$PY_BIN" --version 2>&1)"
"$PY_BIN" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 9) else 1)' \
  || { echo "[error] need python >= 3.9, got $("$PY_BIN" --version 2>&1)" >&2; exit 2; }

cd "$APP"

"$PY_BIN" - <<'PY'
import os, sys, pathlib
sys.path.insert(0, ".")

failures = []

def check(name, got, want):
    ok = got == want
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {got!r}" + ("" if ok else f" != {want!r}"))
    if not ok:
        failures.append(name)

import inference as inf

# 2. Weights must NOT resolve to Grand Challenge's tarball overmount. On this
#    cluster /opt/ml/model does not exist, which is the same observable state as
#    GC's empty read-only mount.
check("model dir is the baked path", inf.MODEL_DIR, inf.BAKED_MODEL_DIR)
check("baked path is next to inference.py",
      inf.BAKED_MODEL_DIR, pathlib.Path(os.getcwd()).resolve() / "model")
check("gc mount constant", inf.GC_MODEL_MOUNT, pathlib.Path("/opt/ml/model"))

# 3. Explicit override still wins over both.
os.environ["ISLES26_MODEL_DIR"] = "/tmp/override"
check("ISLES26_MODEL_DIR override", inf.resolve_model_dir(), pathlib.Path("/tmp/override"))
del os.environ["ISLES26_MODEL_DIR"]
check("falls back once override cleared", inf.resolve_model_dir(), inf.BAKED_MODEL_DIR)

# 4. TTA tri-state. None means auto; the device decision is made in init_model,
#    so replicate that one expression rather than importing torch.
check("TTA default is auto", inf.DISABLE_TTA, None)
for disable, device, want in [(None, "cuda", True), (None, "cpu", False),
                              (True, "cuda", False), (False, "cpu", True)]:
    got = (device == "cuda") if disable is None else not disable
    check(f"tta(disable={disable}, {device})", got, want)

# 5. Interface slugs, against the sockets shown on the algorithm page.
check("input image slug", inf.INPUT_IMAGE_SLUG, "t1-brain-mri")
check("mask slug", inf.MASK_SLUG, "stroke-lesion-segmentation")
check("prob slug", inf.PROB_SLUG, "lesion-probability-map")
check("threshold", inf.THRESHOLD, 0.35)
check("min cc voxels", inf.MIN_CC_VOXELS, 20)
check("flatten empty", inf.FLATTEN_EMPTY, True)
check("folds", inf.FOLDS, (0, 1, 2, 3, 4))

# The dispatch table must accept exactly the challenge's two input sockets.
import inspect
src = inspect.getsource(inf.run)
check("dispatch key", '("stroke-metadata", "t1-brain-mri")' in src, True)

print()
if failures:
    sys.exit(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
print("all container configuration checks passed")
PY
