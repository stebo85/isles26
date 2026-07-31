# Baselines & model-comparison framework

This directory holds **everything reviewable** about each segmentation model we
run on this data: the pipeline scripts, the per-method evaluation reports, and a
leaderboard that compares them. Heavy/regenerable artifacts (trained weights,
predictions, containers, logs) live under `work/` and are gitignored.

```
baselines/
├── models/                 # one self-contained pipeline per model
│   ├── deepisles/          # pretrained ISLES'22 ensemble (DWI/ADC/FLAIR)
│   └── nnunet_atlas_t1w/   # nnU-Net v2 trained by us on ATLAS R2.1 (T1w)
├── reports/                # one dir per (model × eval-set) — the comparison substrate
│   ├── <name>/report.{md,json}   # produced by eval/run_eval.py (or run_atlas_eval.py)
│   ├── <name>/per_case.json
│   └── <name>/meta.json          # method/eval-set/modality tags (see schema below)
├── compare/
│   ├── compare_models.py   # scans reports/*/ → leaderboard.{md,csv}
│   ├── leaderboard.md      # GENERATED — do not hand-edit
│   └── leaderboard.csv     # GENERATED
└── DeepIsles/              # upstream DeepISLES clone (gitignored)
```

## The leaderboard

`reports/<name>/report.json` + `reports/<name>/meta.json` together make one
leaderboard row. Regenerate after any new eval:

```bash
module load devel python/3.12.1
export LD_LIBRARY_PATH=/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}
source .venv-eval/bin/activate
python baselines/compare/compare_models.py        # writes compare/leaderboard.{md,csv}
```

Rows are grouped by **eval set** because models are only comparable on the same
target. We currently have two eval sets:

- **`soop_bench`** — 12 subjects, GT defined on **DWI/TRACE** (acute infarcts).
  This is the out-of-distribution challenge benchmark. Note: a T1w-only model is
  handicapped here because the lesions are DWI-defined (see the nnU-Net method
  card and `docs/lessons/problems.md` Problem 3).
- **`atlas_r21_fold0_val`** — 194 held-out ATLAS R2.1 cases, GT defined on
  **T1w** (chronic). In-distribution for any ATLAS-trained T1w model.
- **`atlas_r30_5fold_cv`** — 1450 usable ATLAS R3.0 / ISLES'26 public-training
  cases evaluated by nnU-Net out-of-fold cross-validation. This is the current
  main in-distribution model-selection track.

`oracle` (GT-as-prediction, Dice≈1) and `empty` (null prediction, Dice 0) bound
each eval set top and bottom as sanity anchors.

## How to add a new model

1. **Create the pipeline dir** `baselines/models/<your_model>/` with numbered
   SLURM scripts following the repo convention `analysis_<NN>_<desc>.sh`
   (see `claude.md`). Put any model-specific helper scripts in a subdir there.
   Add a `README.md` method card (copy the structure of an existing one).
2. **Write predictions** to `work/predictions_<your_model>/` as
   `<subject>.nii.gz`, in the **same physical space as the eval-set GT**
   (resample/register if your model works in another space — see
   `nnunet_atlas_t1w` for the T1w→TRACE example).
3. **Evaluate** with the shared framework so the report schema matches:
   - soop_bench: `python -m eval.run_eval --data-root data/soop_bench
     --pred-dir work/predictions_<m> --out-dir baselines/reports/<m>
     --title "..."`
   - ATLAS val: use `nnunet_helpers/run_atlas_eval.py` as the template.
4. **Tag it** — drop a `meta.json` in the report dir (schema below).
5. **Regenerate** the leaderboard (command above) and commit
   `models/<m>/`, `reports/<m>/{report.*,meta.json}`, and the updated
   `compare/leaderboard.*`.

### `meta.json` schema

```json
{
  "method": "short_id",
  "display_name": "Human readable name (config summary)",
  "model_family": "nnunet | ensemble | transformer | sanity | ...",
  "eval_set": "soop_bench | atlas_r21_fold0_val | ...",
  "input_modalities": ["T1w"],
  "trained_on": "what data / phase, or null",
  "status": "complete | running | planned",
  "pipeline_dir": "baselines/models/<name>  (or null for sanity baselines)",
  "notes": "caveats that matter when reading the number"
}
```

A report dir without `meta.json` is skipped by the leaderboard (that is how
`training_data_characterization/`, which is not a model, stays out).

## Fair-comparison rules (learned the hard way)

- **Only compare within an eval set.** A T1w model's low soop_bench score is a
  modality/phase mismatch, not necessarily a worse model — see the nnU-Net
  method card.
- Keep `oracle` and `empty` present for every eval set as anchors.
- Don't select models on Dice alone; report lesion-F1 and size-binned recall too
  (`docs/lessons/problems.md` Problem 1).
- Validate splits by site × chronicity × lesion-size; never random k-fold
  (leaks center/chronicity structure).
