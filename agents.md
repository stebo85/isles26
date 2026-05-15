## general guidelines
- no credentials in the repo
- commit and push after new features are implemented
- when learning new insights make sure to update agents.md with a short description and longer documentation goes in docs (linked from agents.md)
- literature review for ISLES'26 preparation lives in [docs/Literature_review_summary.md](docs/Literature_review_summary.md); key current lessons are to start with nnU-Net/MAPPING-style baselines, build metrics first, stratify validation by center/chronicity/lesion size, and treat small-lesion detection as the main risk, and compare any attention/transformer/SAM branch against nnU-Net under identical splits
- evaluation framework lives in [eval/](eval/) with docs in [docs/evaluation_framework.md](docs/evaluation_framework.md); sensitive metrics include lesion-wise F1, HD95/ASSD, surface Dice @ 3 mm, and size-binned recall (pooled across cases) plus per-chronicity/per-center stratification; runner is `eval/run_eval.py`, unit tests with `PYTHONPATH=. python eval/tests/test_metrics.py` (19/19)
- python venv for eval is at `.venv-eval/` (built with `python/3.12.1` module); scipy/numpy/etc. need `--only-binary=:all:` and `numpy<2.1 scipy<1.14` on this cluster (no OpenBLAS dev headers available for source builds)
- soop_bench reference benchmark uses DeepISLES (`isleschallenge/deepisles`, SEALS+NVAUTO+SWAN ensemble) because GT masks are in DWI/TRACE space; pipeline is `analysis_01_pull_deepisles.sh` → `analysis_02_run_deepisles.sh` (SLURM array) → `analysis_03_evaluate.sh`
- don't run large analyses directly - write and sumbit slurm job files! Make sure to make them robust to interruption

```bash
#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -p owners


module purge
module use $GROUP_HOME/modules/
module load ants/2.6.0
ants.... $1
```

and submit them to the owners que:
```bash
sbatch -p owners submit.sbatch
```
