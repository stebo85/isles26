#!/bin/bash
#SBATCH --job-name=container_preflight
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH -p normal,owners
#
# Pre-flight for the submission container, run on the cluster because the
# cluster cannot build Docker images and CI is the only place the real image
# gets exercised. This runs every gate CI runs that does not need Docker, so a
# regression is caught here in minutes instead of after a multi-GB image build.
#
# Gates, in the order CI applies them:
#   1. test_geometry.py -- all 48 axis-aligned orientations + NIfTI round-trip.
#      This is the build-blocking test; a silent mirror flip emits a well-formed
#      mask of plausible size and costs ~94% of Dice.
#   2. The app.py HTTP lifecycle: boot the server, poll GET /health until 200,
#      POST /invoke, require 201. This is the contract Grand Challenge uses --
#      a plain function call does not exercise it.
#   3. A synthetic PSR-oriented volume, the same one CI injects: an axis
#      permutation the network never saw in training.
#   4. A real ATLAS case at the SHIPPED defaults, asserting the emitted config
#      is actually folds 0-4 / thr 0.35 / minCC 20 / TTA off, and that both
#      required output sockets land on the caller's exact grid.
#
# The weights are read from work/submission_model, which is what was published
# to Hugging Face and what CI bakes into the image.

set -euo pipefail
REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"; mkdir -p "$REPO/logs"

module purge
[[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]] && module use "$GROUP_HOME/modules"
module load devel; module load python/3.12.1; module load py-pytorch/2.4.1_py312
export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1 nnUNet_compile=f
export CUDA_VISIBLE_DEVICES=""          # the budget that governs is the CPU one
export nnUNet_raw="$REPO/work/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/work/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"

APP="$REPO/submission/isles26_algorithm"
MODEL_DIR="$REPO/work/submission_model/Dataset507_ATLASR30_CORRECTED"
PY="$REPO/.venv-nnunet/bin/python"
WORK="$REPO/work/container_preflight"
rm -rf "$WORK"; mkdir -p "$WORK"

[[ -d "$MODEL_DIR" ]] || { echo "[error] missing $MODEL_DIR" >&2; exit 2; }

fail=0

# fastapi/uvicorn are in the image's requirements.txt but not in .venv-nnunet, and
# without them the HTTP lifecycle -- the actual Grand Challenge contract -- cannot
# be exercised here at all. Install them into a throwaway directory on PYTHONPATH
# rather than into any venv: .venv-nnunet reproduces training and inference
# numbers and must not acquire new transitive pins two weeks before a deadline.
SERVER_LIBS="$WORK/pylibs"
if ! "$PY" -c "import fastapi, uvicorn" 2>/dev/null; then
  echo "[setup] installing fastapi+uvicorn into $SERVER_LIBS (throwaway, no venv touched)"
  "$PY" -m pip install --quiet --no-cache-dir --target "$SERVER_LIBS" fastapi uvicorn \
    || echo "[warn] install failed; the lifecycle test will fall back to a direct call"
fi
# APPEND, never replace: numpy/torch reach this interpreter through the
# py-pytorch module's own PYTHONPATH, so overwriting it hides the stdlib deps
# and makes every gate below fail for a reason that has nothing to do with the
# container. (That is exactly what the first run of this script did.)
export PYTHONPATH="$SERVER_LIBS:$APP:${PYTHONPATH:-}"

echo "===================== 0. dependency inventory ====================="
"$PY" - <<'PY'
import importlib
need = ["numpy", "scipy", "nibabel", "SimpleITK", "torch", "nnunetv2"]
server = ["fastapi", "uvicorn", "httpx"]
for group, mods in (("required", need), ("server", server)):
    for m in mods:
        try:
            mod = importlib.import_module(m)
            print(f"  [{group}] {m:12} OK   {getattr(mod, '__version__', '?')}")
        except Exception as exc:
            print(f"  [{group}] {m:12} MISSING ({type(exc).__name__})")
PY

echo
echo "===================== 1. geometry gate ====================="
# Exactly the command the Dockerfile and the CI job run.
if "$PY" "$APP/test_geometry.py" --per-session-csv /nonexistent; then
  echo "[pass] geometry round-trip over all 48 orientations"
else
  echo "[FAIL] geometry gate"; fail=1
fi

echo
echo "===================== 2-4. lifecycle + real case ====================="
APP="$APP" MODEL_DIR="$MODEL_DIR" WORK="$WORK" REPO="$REPO" "$PY" - <<'PYEOF'
import json, os, shutil, socket, subprocess, sys, time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

APP = Path(os.environ["APP"]); WORK = Path(os.environ["WORK"])
REPO = Path(os.environ["REPO"]); MODEL_DIR = os.environ["MODEL_DIR"]
RAW = REPO / "work/nnunet/nnUNet_raw/Dataset507_ATLASR30_CORRECTED/imagesTr"

failures = []


def make_case(dst: Path, kind: str):
    """Provision one /input directory in the challenge's own layout."""
    img_dir = dst / "images" / "t1-brain-mri"; img_dir.mkdir(parents=True)
    if kind == "psr_synthetic":
        # Byte-for-byte the volume CI injects: a storage orientation the network
        # never saw, so a geometry regression fails here rather than on the
        # leaderboard.
        rng = np.random.default_rng(0)
        img = sitk.GetImageFromArray((rng.random((64, 72, 80)).astype(np.float32) * 100))
        img.SetSpacing((1.0, 1.0, 1.0)); img.SetOrigin((10.0, -5.0, 2.0))
        img.SetDirection((0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0))
        sitk.WriteImage(img, str(img_dir / "case.mha"), useCompression=True)
    else:
        shutil.copy2(RAW / "ATLAS_0001_0000.nii.gz", img_dir / "case.nii.gz")
    (dst / "inputs.json").write_text(json.dumps(
        [{"socket": {"slug": "t1-brain-mri"}}, {"socket": {"slug": "stroke-metadata"}}]))
    (dst / "stroke-metadata.json").write_text(json.dumps(
        {"CENTER": None, "CHRONICITY": None, "DAYS_POST_STROKE": None}))
    return next(img_dir.iterdir())


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def check_outputs(out_dir: Path, src_path: Path, tag: str):
    """Both sockets must exist and sit on the caller's exact grid."""
    src = sitk.ReadImage(str(src_path))
    for slug in ("stroke-lesion-segmentation", "lesion-probability-map"):
        hits = list((out_dir / "images" / slug).glob("*.mha"))
        if not hits:
            failures.append(f"{tag}: missing output socket {slug}"); continue
        d = sitk.ReadImage(str(hits[0]))
        for attr in ("GetSize", "GetSpacing", "GetDirection", "GetOrigin"):
            a, b = getattr(d, attr)(), getattr(src, attr)()
            if attr in ("GetSpacing", "GetDirection", "GetOrigin"):
                ok = np.allclose(a, b, atol=1e-5)
            else:
                ok = a == b
            if not ok:
                failures.append(f"{tag}/{slug}: {attr} {a} != {b}")
        arr = sitk.GetArrayFromImage(d)
        print(f"  [{tag}] {slug}: size={d.GetSize()} dtype={arr.dtype} "
              f"range=({float(arr.min()):.3f}, {float(arr.max()):.3f})")
        if slug == "stroke-lesion-segmentation" and not np.isin(arr, (0, 1)).all():
            failures.append(f"{tag}: mask is not binary")


# --- the HTTP lifecycle, which is the contract GC actually uses ---------------
try:
    import httpx  # noqa: F401
    import fastapi  # noqa: F401
    import uvicorn  # noqa: F401
    have_server = True
except ImportError as exc:
    have_server = False
    print(f"[warn] server deps unavailable here ({exc.name}); testing the handler "
          f"directly instead. CI still exercises the full HTTP lifecycle.")

CASES = [("psr_synthetic", "PSR synthetic (CI's geometry canary)"),
         ("real_atlas", "ATLAS_0001, shipped defaults")]

for kind, label in CASES:
    print(f"\n--- {label} ---", flush=True)
    case_dir = WORK / kind
    in_dir, out_dir = case_dir / "in", case_dir / "out"
    src = make_case(in_dir, kind)
    out_dir.mkdir(parents=True)

    env = dict(os.environ)
    env.update(ISLES26_INPUT_DIR=str(in_dir), ISLES26_OUTPUT_DIR=str(out_dir),
               ISLES26_MODEL_DIR=MODEL_DIR,
               PYTHONPATH=os.pathsep.join([str(APP), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep))
    # NOTE: no ISLES26_FOLDS / THRESHOLD / DISABLE_TTA here on purpose -- the
    # point is to prove the SHIPPED DEFAULTS are what run.

    started = time.time()
    if have_server:
        port = free_port()
        proc = subprocess.Popen([sys.executable, "-c",
                                 f"import uvicorn, app; uvicorn.run(app.app, host='127.0.0.1', port={port}, log_level='warning')"],
                                env=env, cwd=str(APP), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        import httpx
        ready, deadline = False, time.time() + 600
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=5).status_code == 200:
                    ready = True; break
            except Exception:
                pass
            time.sleep(2)
        if not ready:
            proc.kill()
            failures.append(f"{kind}: /health never returned 200")
            print(f"[FAIL] {kind}: server never became healthy")
            print((proc.stdout.read() or "")[-3000:])
            continue
        print(f"  [{kind}] /health 200 after {time.time() - started:.1f}s", flush=True)
        t0 = time.time()
        r = httpx.post(f"http://127.0.0.1:{port}/invoke", timeout=3600)
        invoke_s = time.time() - t0
        if r.status_code != 201:
            failures.append(f"{kind}: /invoke returned {r.status_code}, expected 201")
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log = proc.stdout.read() or ""
    else:
        t0 = time.time()
        p = subprocess.run([sys.executable, "-c",
                            "import inference; raise SystemExit(inference.run(inference.init_model()))"],
                           env=env, cwd=str(APP), capture_output=True, text=True)
        invoke_s = time.time() - t0
        log = p.stdout + p.stderr
        if p.returncode != 0:
            failures.append(f"{kind}: handler exited {p.returncode}")
            print(log[-3000:])

    print(f"  [{kind}] invoke took {invoke_s:.1f}s", flush=True)
    for line in log.splitlines():
        if line.startswith("[init]") or line.startswith("[case]") or "FAILED" in line:
            print(f"    {line}")

    # The shipped defaults must be what actually ran.
    want = ["folds=(0, 1, 2, 3, 4)", "tta=off", "threshold=0.35", "min_cc_voxels=20",
            "flatten_empty=True"]
    for token in want:
        if token not in log:
            failures.append(f"{kind}: expected default '{token}' not in the init log")

    check_outputs(out_dir, src, kind)

print("\n===================== verdict =====================")
if failures:
    print("PREFLIGHT FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("PREFLIGHT PASSED: geometry gate, health/invoke lifecycle, shipped defaults, "
      "and both output sockets on the caller's grid.")
PYEOF
rc=$?

if [[ $rc -ne 0 || $fail -ne 0 ]]; then
  echo "[RESULT] preflight FAILED"; exit 1
fi
echo "[RESULT] preflight PASSED -- container is ready to build"
