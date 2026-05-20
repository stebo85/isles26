## general guidelines
- when learning new insights make sure to update agents.md with a short description and longer documentation goes in docs (linked from agents.md)
- always use codex to implement code and use codex adversarial review to check generated code and assumptions
- literature review for ISLES'26 preparation lives in [docs/Literature_review_summary.md](docs/Literature_review_summary.md); key current lessons are to start with nnU-Net/MAPPING-style baselines, build metrics first, stratify validation by center/chronicity/lesion size, and treat small-lesion detection as the main risk, and compare any attention/transformer/SAM branch against nnU-Net under identical splits
- evaluation framework lives in [eval/](eval/) with docs in [docs/evaluation_framework.md](docs/evaluation_framework.md); sensitive metrics include lesion-wise F1, HD95/ASSD, surface Dice @ 3 mm, and size-binned recall (pooled across cases) plus per-chronicity/per-center stratification; runner is `eval/run_eval.py`, unit tests with `PYTHONPATH=. python eval/tests/test_metrics.py` (19/19)
- ATLAS R2.1 raw training data is decrypted to `data/ATLAS_R2.1_raw/Training_Raw/` (gitignored). Characterization: 955 sessions / 33 sites, all readable, file/affine/shape sanity OK; key lessons = 82 % 1 mm iso (R015/R027/R049 are non-iso outliers), orientations split LAS/RAS (force canonical), T1w intensity scale varies by 5 orders of magnitude (per-image percentile + z-score required; `t1_brain_p50`/`t1_brain_p99` columns added as normalization anchors), 126 sessions have no `DAYS_POST_STROKE` and 24 R049/R050 sessions encode it as `0`/`-4` (treated as `unknown` → `chronicity_counts = {ge180d: 540, lt180d: 264, unknown: 151}`), R039 stores masks as float64 `0.9999…` (re-binarize on load is step 1 of the preprocessing block — mandatory), lesion volumes log-skewed (median 4.3 mL, p95 136 mL, p5 0.14 mL — small lesions dominate lesion-wise F1), 26-connectivity CC distribution median 2 / p75 3 / p95 8 / max 38 (matches `eval/metrics.py`), severe per-site chronicity bias (R031 ≈3 % chronic vs. R047/R048 = 100 %). Full report + per-site CSV + QC PNGs in [docs/atlas_r21_training_data_characterization.md](docs/atlas_r21_training_data_characterization.md) and [baselines/reports/training_data_characterization/](baselines/reports/training_data_characterization/) (includes `adversarial_review.md` with defects 1–7 + fixes); pipeline scripts in [scripts/analysis/](scripts/analysis/) (SLURM job `analysis_01_characterize_training_data.sh`).
- python venv for eval is at `.venv-eval/` (built with `python/3.12.1` module); scipy/numpy/etc. need `--only-binary=:all:` and `numpy<2.1 scipy<1.14` on this cluster (no OpenBLAS dev headers available for source builds)
- nnU-Net v2 3D fullres T1w baseline on ATLAS R2.1 (`Dataset501_ATLASR21`, fold 0, 250 epochs) lives in [baselines/analyses/analysis_nnunet_*](baselines/analyses/) (six SLURM scripts: 01 prepare → 02 train → 03 soop predict → 04 soop eval; 05a compute val array → 05 ATLAS val predict (SLURM array) → 06 ATLAS val eval). Two reports are produced: [baselines/reports/nnunet/](baselines/reports/nnunet/) (soop_bench OOD) and [baselines/reports/nnunet_atlas_val/](baselines/reports/nnunet_atlas_val/) (ATLAS fold-0 held-out, in-distribution). Key engineering: stratified splits by site × chronicity × lesion-size bin (seed 42, 5 folds) written via `convert_atlas_r21_to_nnunet.py splits`; trainer subclass `nnUNetTrainer_250epochs` (num_epochs=250, save_every=10) installed into the venv's `nnunetv2/training/.../variants/training_length/`; soop_bench inference uses SynthStrip skull-strip + ANTs rigid T1w→TRACE registration + `antsApplyTransforms -n GenericLabel` to warp predictions into TRACE space (TRACE files are 4-D singleton (X,Y,Z,1) — squeeze to 3-D before any ANTs call); preemption-safe via `--requeue` + USR1 trap calling `scontrol requeue`; checkpoint-SHA sidecars (`<sub>.json` next to each `<sub>.nii.gz`) gate idempotent re-runs. nnU-Net venv at `.venv-nnunet/` (separate from `.venv-eval/`) with `--system-site-packages` to pick up cluster's `py-pytorch/2.4.1_py312` + CUDA 12.6; pinned `nnunetv2==2.6.2` + `scikit-learn==1.5.2`, all deps `--only-binary=:all:`. Use `PYTHONNOUSERSITE=1` and `LD_LIBRARY_PATH=/share/software/user/open/python/3.12.1/lib:...` for every python invocation. Codex was used to write all scripts and the adversarial-review pass found 1 blocker (ANTs `-d 3` vs 4-D singleton TRACE) + 2 majors (registration tail level, array vs batched val) — see `/tmp/adv_review_2.md` lineage.
- **nnU-Net-on-this-cluster pip/runtime gotchas (hard-won, all fixed in `analysis_nnunet_01`/`02`):** (1) `nnunetv2`, `acvl-utils`, `dynamic-network-architectures` ship **sdist-only** (pure-Python) — do NOT use `--only-binary=:all:` for them or pip errors "no matching distribution"; install nnunetv2 with `--no-deps` then its deps without the binary-only flag (pip still prefers wheels for numpy/scipy/etc.). (2) nnunetv2 2.6.2 requires `acvl-utils>=0.2.3,<0.3`, `dynamic-network-architectures>=0.4.1,<0.5`, `seaborn`, `nibabel` — all must be explicit since we install `--no-deps`. (3) Drop `dicom2nifti` (pulls `python-gdcm` whose sdist has a broken CMake build) and `imagecodecs` (sdist needs `lcms2.h`; manylinux_2_28 wheel needs newer GLIBC) — nnunetv2 imports neither at runtime. (4) The cluster torch is a **custom dev build `2.4.0a0+gitunknown`**: no published `torchvision` wheel matches its ABI, so nnU-Net's `find_class_by_name` trainer-discovery (which imports *every* `variants/` module and does NOT catch ImportError) dies on `primus_trainers.py` → `timm` → `torchvision`. Fix: neutralize `…/nnUNetTrainer/primus/primus_trainers.py` to a stub after install (done in `analysis_nnunet_01`). (5) Same dev-build torch has a broken `_inductor`/triton, so `torch.compile` crashes with `TypeError: must be called with a dataclass type` — set `export nnUNet_compile=f` in every train/predict script. (6) Custom trainer subclasses MUST replicate the parent `__init__(self, plans, configuration, fold, dataset_json, device=...)` signature exactly — nnU-Net captures init kwargs via `locals()[k]` by name, so `*args, **kwargs` raises `KeyError: 'args'`. (7) nnU-Net saves plans as `nnUNetPlans.json`, not `plans.json` — preflight check accordingly.
- don't run large analyses directly - write and sumbit slurm job files! Make sure to make them robust to interruption

```bash
#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=work/logs/%x_%j.out
#SBATCH --error=work/logs/%x_%j.err
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

## project layout

Rule of thumb: anything reviewable (markdown reports, summary JSONs, QC snapshots) belongs in this repo (e.g. `baselines/` for analysing existing state of the art baseline models); anything temporary and regenerable from a script belongs in `work/` including logs. 


```
isles26challenge/
├── agents.md                          # this file — agent guidelines + repo map
├── scripts/                           # repo-wide utility scripts (not tied to a specific baseline)
│   └── download_data.sh               # datalad pull of soop_bench
├── eval/                              # evaluation framework (loader, metrics, qc_viz, report, run_eval)
│   └── tests/                         # unit tests (`PYTHONPATH=. python eval/tests/test_metrics.py`)
├── docs/                              # literature review, evaluation framework, challenge notes
├── baselines/                         # KEPT artifacts (committed)
│   ├── DeepIsles/                     # upstream DeepISLES repo clone (gitignored)
│   ├── analyses/                      # SLURM-ready baseline pipelines
│   │   ├── analysis_01_pull_deepisles.sh  # apptainer pull → sandbox under work/containers/
│   │   ├── analysis_02_run_deepisles.sh   # SLURM array over subjects → work/predictions_deepisles/
│   │   └── analysis_03_evaluate.sh        # eval framework → baselines/reports/deepisles/
│   └── reports/                       # benchmark reports per method
│       ├── deepisles/                 # report.md/.json, per_case.json, qc/*.png, benchmark_summary.md, adversarial_review.md
│       ├── empty/                     # null-prediction baseline
│       └── oracle/                    # GT-as-prediction sanity check
├── work/                              # TEMPORARY outputs (gitignored, regenerable)
│   ├── containers/                    # apptainer sandbox + cache (~44 GB)
│   ├── deepisles_runs/                # per-subject DeepISLES intermediate run dirs
│   ├── predictions_deepisles/         # NIfTI predictions consumed by analysis_03
│   └── predictions_oracle/            # GT copied as predictions for oracle baseline
├── data/                              # soop_bench BIDS data (gitignored, datalad-managed)
├── logs/                              # SLURM stdout/stderr (gitignored)
└── .venv-eval/                        # python venv for eval (gitignored)
```
