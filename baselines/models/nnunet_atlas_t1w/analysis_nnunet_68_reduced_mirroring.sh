#!/bin/bash
#SBATCH --job-name=reduced_mirroring
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --array=0-3
#
# Can a CHEAPER mirroring TTA buy back what the container gave up?
#
# The shipped container runs 5 folds with mirroring TTA OFF, because 5 folds +
# full TTA costs 12.9 min/case on CPU against a ~10 min budget. analysis_nnunet_61
# measured what that costs: Dice -0.0141, lesion-F1 -0.0205, PR-AUC -0.0191.
#
# Full mirroring flips all three axes and therefore costs 2^3 = 8 forward passes.
# A SUBSET of axes costs 2^k. At 5 folds that is:
#     0 axes (shipped)   5 passes    ~2.0 min/case
#     1 axis            10 passes    ~4 min/case      <- fits the budget
#     2 axes            20 passes    ~8 min/case      <- marginal
#     3 axes (full)     40 passes   ~12.9 min/case    <- over budget
# So if most of TTA's gain lives in one or two axes, both effects fit and the
# container gets the accuracy back for free.
#
# Design is identical to analysis_nnunet_61 so the arms are directly comparable:
# fold 0 of Dataset507 (center-grouped), its 281 validation cases, fold 0 ONLY,
# run through the container's own inference.py. The controls already exist --
# no-TTA in work/official_eval/_probs_noTTA_fold0 and full TTA in
# .../_probs_d507/fold_0 -- so mirroring axes are the single variable.
#
# Axis convention: volumes are RAS-canonical and the plans record
# transpose_forward [0,1,2], so axis 0 is left-right (the anatomically
# meaningful flip for a brain), 1 is posterior-anterior, 2 is inferior-superior.
# All three singletons are run rather than assumed, plus the best-guess pair.
#
# One array task per axis subset.

set -euo pipefail
REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"; mkdir -p "$REPO/logs"

module purge
[[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]] && module use "$GROUP_HOME/modules"
module load devel; module load python/3.12.1; module load py-pytorch/2.4.1_py312
export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1 nnUNet_compile=f
export nnUNet_raw="$REPO/work/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/work/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"

DATASET=Dataset507_ATLASR30_CORRECTED
MODEL_DIR="$REPO/work/submission_model/$DATASET"

AXES_LIST=("0" "1" "2" "0,1")
TAGS=("ax0" "ax1" "ax2" "ax01")
IDX="${SLURM_ARRAY_TASK_ID:-0}"
AXES="${AXES_LIST[$IDX]}"
TAG="${TAGS[$IDX]}"
OUT="$REPO/work/official_eval/_probs_mirror_${TAG}_fold0"
mkdir -p "$OUT"

echo "[info] array task $IDX: mirror axes ($AXES) -> $OUT"

MODEL_DIR="$MODEL_DIR" OUT="$OUT" AXES="$AXES" REPO="$REPO" \
"$REPO/.venv-nnunet/bin/python" - <<'PYEOF'
import json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

REPO = Path(os.environ["REPO"])
model_dir, out, axes = os.environ["MODEL_DIR"], Path(os.environ["OUT"]), os.environ["AXES"]
RAW = REPO / "work/nnunet/nnUNet_raw/Dataset507_ATLASR30_CORRECTED"

splits = json.loads((REPO / "work/nnunet/nnUNet_preprocessed"
                     / "Dataset507_ATLASR30_CORRECTED/splits_final.json").read_text())
cases = sorted(splits[0]["val"])
print(f"[info] {len(cases)} fold-0 validation cases, fold 0 only, mirror axes=({axes})",
      flush=True)

elapsed = []
for i, cid in enumerate(cases, 1):
    target = out / f"{cid}.nii.gz"
    if target.exists():
        continue
    with tempfile.TemporaryDirectory(prefix=f"mir_{cid}_") as t:
        tmp = Path(t)
        ind = tmp / "in" / "images" / "t1-brain-mri"; ind.mkdir(parents=True)
        shutil.copy2(RAW / "imagesTr" / f"{cid}_0000.nii.gz", ind / f"{cid}.nii.gz")
        (tmp / "in" / "inputs.json").write_text(json.dumps(
            [{"socket": {"slug": "t1-brain-mri"}}, {"socket": {"slug": "stroke-metadata"}}]))
        (tmp / "in" / "stroke-metadata.json").write_text(json.dumps(
            {"CENTER": None, "CHRONICITY": None, "DAYS_POST_STROKE": None}))
        env = dict(os.environ)
        env.update(
            ISLES26_INPUT_DIR=str(tmp / "in"), ISLES26_OUTPUT_DIR=str(tmp / "out"),
            ISLES26_MODEL_DIR=model_dir, ISLES26_FOLDS="0",
            ISLES26_DISABLE_TTA="0",          # mirroring ON, but only on AXES
            ISLES26_MIRROR_AXES=axes,
            ISLES26_FLATTEN_EMPTY="0",        # raw maps: the scorer applies the rule
            PYTHONPATH=os.pathsep.join([str(REPO / "submission" / "isles26_algorithm"),
                                        os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep))
        t0 = time.time()
        p = subprocess.run([sys.executable, "-c",
                            "import inference; inference.run(inference.init_model())"],
                           env=env, capture_output=True, text=True,
                           cwd=str(REPO / "submission" / "isles26_algorithm"))
        elapsed.append(time.time() - t0)
        if p.returncode != 0:
            print(f"[FAIL] {cid}\n{p.stderr[-1500:]}", flush=True)
            continue
        if i == 1:
            # Prove the override actually took effect rather than silently
            # falling back to all three axes.
            for line in p.stdout.splitlines():
                if line.startswith("[init]"):
                    print(f"  {line}", flush=True)
            # Exact tuple repr, not a prefix: "mirror_axes=(0" is also a
            # prefix of "(0, 1, 2)", so a prefix test would pass silently when
            # the override did NOT apply -- the one thing this check exists for.
            want = "mirror_axes=" + repr(tuple(int(a) for a in axes.split(",")))
            if want not in p.stdout:
                sys.exit(f"[error] expected '{want}...' in the init log; "
                         "the mirroring override did not apply")
        pp = next(m for m in (tmp / "out" / "images").rglob("*.mha")
                  if "probability" in m.parent.name)
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(pp))).transpose(2, 1, 0).astype(np.float32)
        ref = nib.load(str(RAW / "imagesTr" / f"{cid}_0000.nii.gz"))
        nib.save(nib.Nifti1Image(arr, ref.affine, ref.header), str(target))
    if i % 25 == 0:
        print(f"[progress] {i}/{len(cases)}  mean {np.mean(elapsed):.1f}s/case", flush=True)

n = len(list(out.glob("*.nii.gz")))
print(f"[done] axes=({axes}): {n} probability maps"
      + (f", mean {np.mean(elapsed):.1f}s/case on GPU" if elapsed else ""), flush=True)
if n != len(cases):
    sys.exit(f"[error] expected {len(cases)} maps, wrote {n}")
PYEOF

echo "[next] score with analysis_nnunet_69_score_reduced_mirroring.sh"
