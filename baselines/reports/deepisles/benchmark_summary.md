# DeepISLES on soop_bench — Benchmark Summary

**Model**: DeepISLES (de la Rosa et al. 2025) — ensemble of SEALS (nnU-Net),
NVAUTO (MONAI Auto3DSeg), and SWAN (Factorizer), with HD-BET skull stripping
and majority-voting fusion.

**Image pinned by digest** (Codex review #6):
`docker://isleschallenge/deepisles@sha256:52f7ec71c60d50b94710f04b5e4558966060a68068fed463f665880664a783a8`
(`:latest` at pull time, 2026-05-15). Apptainer-recorded base digest
`sha256:848c9eceb67dbc585bcb37f093389d142caeaa98878bd31039af04ef297a5af4`.
Full `apptainer inspect` output in
[`work/containers/deepisles_image_metadata.json`](../../../work/containers/deepisles_image_metadata.json).

**Inputs**: per-subject DWI (TRACE), ADC (resampled to TRACE grid with linear
interpolation), FLAIR. Container ran with `--skull_strip`.

**Ground truth**: `derivatives/lesion_masks/<sub>/dwi/<sub>_space-TRACE_desc-lesion_mask.nii.gz`
(union of acute + chronic lesions where both are annotated).

**Evaluation framework**: `eval/run_eval.py` (see [docs/evaluation_framework.md](../../../docs/evaluation_framework.md)).

## Headline numbers (n=12)

| Metric                  | n_valid | Mean   | Median |
|-------------------------|--------:|-------:|-------:|
| Dice                    | 12      | 0.540  | 0.704  |
| Lesion-wise F1          | 12      | 0.582  | 0.608  |
| Lesion recall           | 12      | 0.584  | 0.536  |
| Lesion precision        | 12      | 0.803  | 1.000  |
| HD95 (mm)               | **10**  | 18.9   |  6.9   |
| ASSD (mm)               | **10**  |  6.6   |  1.8   |
| Surface Dice @ 3 mm     | **10**  | 0.763  | 0.835  |
| AVD (mL)                | 12      | 18.9   |  3.1   |
| |Δ lesion count|        | 12      |  2.25  |  1.0   |

The wide mean–median gap (Dice mean 0.54 vs median 0.70) is driven by **two
empty-prediction cases (sub-8 and sub-11)** that drop the mean — see below.

**Boundary metrics caveat**: HD95, ASSD, and surface Dice are undefined when
either mask is empty, so the two empty-prediction failures (sub-8, sub-11)
are excluded from boundary aggregates (n_valid = 10/12). Reading the boundary
numbers in isolation will understate the true error — Dice / lesion-F1 are
the honest headline metrics here.

## Where the model fails

The benchmark surfaces the failure modes the literature warned us about:

1. **Catastrophic empty-mask predictions on 2/12 subjects.** sub-8 (acute,
   80 mL, 9 components) and sub-11 (chronic, 5 mL) get **Dice = 0.000** —
   the model predicted no lesion at all. sub-8 in particular has obvious
   diffuse stroke on TRACE; this is the single most damaging case to overall
   Dice.
2. **Tiny lesions are systematically missed.** Pooled lesion recall:

   | size bin (voxels) | n_GT | n_detected | recall |
   |---|---:|---:|---:|
   | tiny (<10)        |  5 | **0** | **0.00** |
   | small (10–100)    | 19 | 11 | 0.58 |
   | medium (100–1k)   | 11 |  3 | 0.27 |
   | large (1k–10k)    |  9 |  4 | 0.44 |
   | huge (≥10k)       |  4 |  3 | 0.75 |

   By volume in mL, the picture is the same:

   | size bin (mL) | n_GT | n_detected | recall |
   |---|---:|---:|---:|
   | <0.1 mL       | 11 |  2 | **0.18** |
   | 0.1–1 mL      | 18 | 11 | 0.61 |
   | 1–10 mL       | 11 |  3 | 0.27 |
   | 10–50 mL      |  4 |  3 | 0.75 |
   | ≥50 mL        |  4 |  2 | 0.50 |

   Tiny (<10 voxel) GT components — 5 of them, spread across 4 subjects — were
   100% missed. This is exactly the documented ATLAS / ISLES'22 failure mode.

3. **One chronic subject was the only chronic case, and DeepISLES missed it
   entirely.** DeepISLES was trained for acute DWI-bright stroke, so a chronic
   case with no DWI hyperintensity is out of distribution. n=1 makes this
   anecdotal, but worth flagging.

4. **Per-scanner-model numbers are descriptive only — not evidence of a
   scanner effect.** Dice ranges from 0.21 (Philips Intera, n=2) to 0.87
   (Philips Achieva, n=2). Each center group has 2 subjects, so this is
   indistinguishable from per-subject noise. The framework supports
   per-center reporting, which will be meaningful on the full ISLES'26
   training set; on soop_bench it is a smoke test of the column, not a
   finding.

## Where the model does well

- 8/12 subjects have Dice > 0.5; 6/12 have Dice > 0.70.
- **Lesion precision is high (median 1.00).** When DeepISLES predicts a
  component, it's usually real — overdetection is rare.
- **Best case sub-3 reaches Dice 0.91**, F1 0.60, with a 154 mL lesion that
  the model captures faithfully on the right hemisphere; the small left-side
  lesion is missed.
- Surface Dice @ 3 mm is 0.76 mean / 0.84 median — boundary fidelity is
  reasonable on the cases that aren't missed entirely.

## Per-case table (worst → best by Dice)

| subject  | chronicity | center                       | Dice  | F1    | Recall | HD95 mm | Pred mL | GT mL  | GT cc | Pred cc |
|----------|------------|------------------------------|------:|------:|-------:|--------:|--------:|-------:|------:|--------:|
| sub-11   | chronic    | Philips Ingenia Ambition X   | 0.000 | 0.000 |  0.00  |   n/a   |   0.00  |   5.17 |    2  |    0    |
| sub-8    | acute      | Philips Ingenia Ambition X   | 0.000 | 0.000 |  0.00  |   n/a   |   0.00  |  79.74 |    9  |    0    |
| sub-1    | mixed      | Philips Intera               | 0.165 | 0.615 |  0.50  |   77.1  |  11.98  | 110.44 |    4  |    5    |
| sub-13   | acute      | Philips Intera               | 0.261 | 0.400 |  0.33  |   36.0  |   0.54  |   1.08 |    3  |    2    |
| sub-4    | acute      | Philips Ingenia Ambition X   | 0.518 | 1.000 |  1.00  |    7.6  |   1.14  |   3.23 |    1  |    2    |
| sub-14   | acute      | Philips Ingenia Ambition X   | 0.674 | 1.000 |  1.00  |    6.1  |   0.23  |   0.45 |    1  |    1    |
| sub-7    | acute      | Philips Ingenia Ambition X   | 0.734 | 0.667 |  0.50  |    6.0  |   7.83  |   8.53 |   12  |    8    |
| sub-10   | acute      | Philips Ingenia Ambition X   | 0.747 | 0.571 |  1.00  |    5.5  |   8.68  |  12.19 |    1  |    5    |
| sub-2    | acute      | Philips Ingenia Elition X    | 0.756 | 0.727 |  0.57  |   19.3  |  54.53  |  82.90 |    7  |    8    |
| sub-5    | acute      | Philips Achieva              | 0.829 | 1.000 |  1.00  |    0.9  |   0.59  |   0.83 |    1  |    1    |
| sub-9    | acute      | Philips Ingenia Elition X    | 0.884 | 0.400 |  0.50  |    2.4  |  31.68  |  34.33 |    2  |    6    |
| sub-3    | mixed      | Philips Achieva              | 0.906 | 0.600 |  0.60  |   28.4  | 154.95  | 159.99 |    5  |    5    |

QC overlays (TRACE background, **green** = FN, **red** = FP, **yellow** = TP)
are in `work/reports/deepisles/qc/`.

## How these numbers compare to DeepISLES's published results

The DeepISLES Nature Communications paper reports strong performance on the
in-distribution ISLES'22 test cohort (see [de la Rosa et al. 2025](https://www.nature.com/articles/s41467-025-62373-x) Table 1 / Fig. 2
for the headline numbers — verify before quoting in a publication; our
out-of-distribution comparison should not paraphrase those figures from
memory).

What we can say from this run:

- **Median** Dice 0.70 and lesion F1 0.61 on soop_bench fall well below
  DeepISLES's reported ISLES'22 figures, even before mean–median divergence
  is considered.
- The **mean** is dominated by two empty-prediction failures (sub-8, sub-11),
  pulling mean Dice down to 0.54 and mean F1 down to 0.58.
- soop_bench is small (n=12), all Philips, and includes one chronic case that
  the model — which is trained on acute DWI-bright lesions — was never
  designed to find. None of these are in-distribution for DeepISLES, so a
  numerical gap is expected; the interesting questions are *which* cases fail
  and *why*, not the headline mean.

Do not treat this benchmark as a like-for-like reproduction of DeepISLES's
ISLES'22 numbers. n=12 with no confidence intervals is not enough to claim
"close to" or "below" published performance.

## Reproducibility

```
analysis_01_pull_deepisles.sh   # builds 27 GB Apptainer sandbox image
analysis_02_run_deepisles.sh    # SLURM array, 1 GPU per subject, ~3 min each
analysis_03_evaluate.sh         # runs eval/run_eval.py on predictions
PYTHONPATH=. python eval/qc_viz.py ...   # generates PNG overlays
```

Pinned constraint `GPU_CC:8.6|GPU_CC:8.0|GPU_CC:7.5|GPU_CC:7.0` excludes
Hopper / Ada GPUs (PyTorch 1.11 in the DeepISLES container does not support
them).
