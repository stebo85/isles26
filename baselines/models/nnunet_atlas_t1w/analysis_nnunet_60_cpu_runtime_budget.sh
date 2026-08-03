#!/bin/bash
#SBATCH --job-name=cpu_runtime_budget
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH -p normal
#
# Measure CPU inference time per case for each fold/TTA configuration, because
# the challenge guidelines quote a ~10 MINUTE CPU BUDGET and the official
# template only says the GPU is used "when enabled" -- i.e. a GPU is not
# guaranteed on the evaluation instances.
#
# Our shipped configuration is a 5-fold ensemble with mirroring TTA, measured at
# ~30 s/case on a GPU of which only ~8 s is GPU compute. On CPU, 5 folds x 8
# mirror configurations is a plausible way to blow that budget, and the fold
# count is a real accuracy trade -- so it must be measured, not assumed.
#
# Deliberately runs on CPU only (no --gres), with 8 cores, which is a plausible
# stand-in for a Grand Challenge CPU instance.

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
export CUDA_VISIBLE_DEVICES=""          # force CPU even if a GPU is visible

DATASET=Dataset507_ATLASR30_CORRECTED
MODEL_DIR="$REPO/work/submission_model/$DATASET"
OUT="$REPO/baselines/reports/cpu_runtime_budget"
mkdir -p "$OUT"

PYTHONPATH="$REPO/submission/isles26_algorithm:${PYTHONPATH:-}" \
"$REPO/.venv-nnunet/bin/python" - "$MODEL_DIR" "$OUT" <<'PYEOF'
import csv, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

model_dir, out_dir = sys.argv[1], Path(sys.argv[2])
REPO = Path("/scratch/users/sciget/isles26challenge")
RAW = REPO / "work/nnunet/nnUNet_raw/Dataset507_ATLASR30_CORRECTED"
man = {r["nnunet_id"]: r for r in csv.DictReader(open(RAW / "conversion_manifest.csv"))}

# Time a median-sized case and a large one: runtime scales with brain volume,
# and a budget that only holds for the median is not a budget.
cands = sorted(man, key=lambda c: float(man[c]["lesion_volume_ml"] or 0))
cases = [cands[len(cands) // 2], cands[-1]]

CONFIGS = [
    ("5fold_tta",   "0,1,2,3,4", "0"),
    ("5fold_noTTA", "0,1,2,3,4", "1"),
    ("1fold_tta",   "0",         "0"),
    ("1fold_noTTA", "0",         "1"),
]

rows = []
for cid in cases:
    raw_t1 = Path(man[cid]["t1_source_path"])
    for name, folds, no_tta in CONFIGS:
        with tempfile.TemporaryDirectory(prefix=f"cpu_{cid}_") as t:
            tmp = Path(t)
            ind = tmp / "in" / "images" / "t1-brain-mri"; ind.mkdir(parents=True)
            shutil.copy2(raw_t1, ind / f"{cid}.mha".replace(".mha", ".nii.gz"))
            # The handler dispatches on inputs.json, so provide the real fixtures.
            (tmp / "in" / "inputs.json").write_text(json.dumps([
                {"socket": {"slug": "t1-brain-mri"}},
                {"socket": {"slug": "stroke-metadata"}}]))
            (tmp / "in" / "stroke-metadata.json").write_text(json.dumps(
                {"CENTER": "R001", "CHRONICITY": None, "DAYS_POST_STROKE": None}))
            env = dict(os.environ)
            env.update(ISLES26_INPUT_DIR=str(tmp / "in"), ISLES26_OUTPUT_DIR=str(tmp / "out"),
                       ISLES26_MODEL_DIR=model_dir, ISLES26_FOLDS=folds,
                       ISLES26_DISABLE_TTA=no_tta, CUDA_VISIBLE_DEVICES="",
                       PYTHONPATH=os.pathsep.join([str(REPO / "submission" / "isles26_algorithm"),
                                                   os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep))
            t0 = time.time()
            p = subprocess.run([sys.executable, "-c",
                                "import inference,sys; m=inference.init_model(); "
                                "t=__import__('time').time(); inference.run(m); "
                                "print(f'INVOKE_SECONDS={__import__(\"time\").time()-t:.1f}')"],
                               env=env, capture_output=True, text=True, cwd=str(REPO / "submission" / "isles26_algorithm"))
            total = time.time() - t0
        invoke = None
        for line in p.stdout.splitlines():
            if line.startswith("INVOKE_SECONDS="):
                invoke = float(line.split("=")[1])
        ok = p.returncode == 0
        rows.append(dict(case=cid, config=name, ok=ok,
                         total_s=round(total, 1), invoke_s=invoke))
        print(f"[{'ok' if ok else 'FAIL'}] {cid} {name:12s} total {total:6.1f}s "
              f"invoke {invoke if invoke is not None else float('nan'):6.1f}s", flush=True)
        if not ok:
            print(p.stderr[-1500:], flush=True)

(out_dir / "timings.json").write_text(json.dumps(rows, indent=2) + "\n")
print("\n=== per-invoke seconds (the number the 10-minute budget applies to) ===")
for name, _, _ in CONFIGS:
    v = [r["invoke_s"] for r in rows if r["config"] == name and r["invoke_s"] is not None]
    if v:
        print(f"  {name:12s} max {max(v):7.1f}s  ({max(v)/60:.1f} min)"
              f"   {'OK' if max(v) < 540 else 'OVER BUDGET'}")
PYEOF
echo "[done] $OUT"
