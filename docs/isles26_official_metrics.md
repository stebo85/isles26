# ISLES'26 official metrics — the definitive reference

**This document is the arbiter.** Where it disagrees with any other doc in this
repo, or with any number in `baselines/reports/`, this document is right.

Source of truth: the organizers' own code, cloned verbatim to
`work/isles26_upstream/utils/eval_utils.py`
([ezequieldlrosa/isles26](https://github.com/ezequieldlrosa/isles26)).
We import it rather than reimplement it — see `eval/official_metrics.py`.

## Provenance and a correction

The file was **rewritten on 2026-07-24** (commit `c430f15`), nine days after
first publication. Three properties people had reasoned about from the
2026-07-15 version are no longer true:

| observed on the 2026-07-15 version | status at HEAD |
|---|---|
| ROC-AUC via `roc_auc_score` | **changed** — now PR-AUC via `precision_recall_curve` + `auc(recall, precision)` |
| true positive = 1-voxel overlap of a GT component | **changed** — now panoptica one-to-one matching at **IoU >= 0.25** |
| `np.bool` / `np.float` (removed in NumPy 1.24) | **fixed** — now `np.bool_` / `np.float32` |
| `empty_value=1.0` | **unchanged** — empty-GT cases are scored and a correct empty prediction is rewarded |

## The five ranked metrics

| # | metric | implementation | direction |
|---|---|---|---|
| 1 | Dice | `panoptica result.global_bin_dsc` (plain binary Dice) | higher better |
| 2 | Absolute volume difference (mL) | voxel counts x voxel volume | **lower** better |
| 3 | Absolute lesion count difference | `abs(num_ref_instances - num_pred_instances)` | **lower** better |
| 4 | Lesion-wise F1 | `panoptica result.rq` = `TP / (TP + 0.5*FP + 0.5*FN)` | higher better |
| 5 | PR-AUC | `auc(recall, precision)` over the **soft** map | higher better |

Instance handling for #3 and #4:
`ConnectedComponentsInstanceApproximator` + `NaiveThresholdMatching(matching_threshold=0.25)`.

## Semantics we MEASURED rather than assumed

`eval/probe_panoptica_semantics.py` runs synthetic phantoms with known answers
through the organizers' code. All confirmed:

- **Connectivity is 26.** A corner-adjacent voxel pair is ONE instance. This
  matches `eval/metrics.py`; had it been 6, every component count in the repo
  would have been wrong.
- **The 0.25 threshold is on IoU, not Dice.** A pair with IoU 0.176 and Dice
  0.300 does NOT match (`rq = 0.0`).
- **Matching is one-to-one** (`allow_many_to_one=False` in panoptica).
- `rq == TP/(TP + 0.5FP + 0.5FN)`; `global_bin_dsc` == plain binary Dice.
- **PR-AUC is invariant under any strictly monotone transform.** Temperature
  scaling, Platt calibration and sigmoid rescaling therefore cannot move it.
  Do not spend time on calibration.
- Volume difference is in mL; the wrapper takes voxel volume already in mL.

## The two edge cases that will surprise you

**Empty ground truth, PR-AUC.** `compute_pr_auc` returns `empty_value=1.0` on a
lesion-free ground truth **only if the soft map is exactly constant**, and `0.0`
otherwise:

```python
if total_positives == 0:
    if np.all(pred_flat == pred_flat[0]):
        return empty_value      # 1.0
    else:
        return 0.0
```

A real softmax is never exactly constant, so a model that correctly finds no
lesion scores **0.0**, while a model emitting a hard constant scores 1.0. Dice
and lesion-F1 on that same case both return 1.0. This asymmetry is why the
submission container emits an all-zero map when its binary mask is empty
(`ISLES26_FLATTEN_EMPTY`, on by default). It is raised as an open question with
the organizers in `submission/isles26_algorithm/FORUM_QUESTIONS.md`.

**Empty on both sides, Dice and F1.** `empty_value=1.0`. Empty cases are scored,
not excluded. In ATLAS R3.0 out-of-fold, 5/1450 cases have empty GT and 26/1450
get an empty prediction.

## How this differs from `eval/metrics.py` — and why we keep both

`eval/metrics.py` matches lesions at **1-voxel any-overlap, many-to-many**. That
is far more lenient than one-to-one IoU >= 0.25: a single-voxel clip of a lesion
counts as a detection, and one blob bridging two lesions gets credit for both.

The two harnesses have different jobs:

- `eval/official_metrics.py` **decides**. Model selection, operating-point
  selection, and anything that goes in a submission.
- `eval/metrics.py` **explains**. Size-binned recall, per-component overlap,
  HD95/ASSD/surface Dice, per-center and per-chronicity stratification — none of
  which the official scorer provides, and all of which are needed to understand
  *why* a number moved.

Agreement between them, and the intentional divergences, are pinned by
`eval/tests/test_official_agreement.py`.

**Consequence:** every `lesion_f1` recorded in `baselines/reports/` before
2026-07-28 uses the lenient criterion and is NOT comparable to the leaderboard.
Those columns have been relabelled rather than deleted.

## Running it

```bash
# one-off venv (panoptica + scikit-learn; kept separate from .venv-eval)
sbatch scripts/analysis_eval_01_setup_official_venv.sh

# score + sweep threshold x min-component-size on saved out-of-fold probabilities
sbatch baselines/models/nnunet_atlas_t1w/analysis_nnunet_47_official_metric_sweep.sh
SURFACE=d507 sbatch baselines/models/nnunet_atlas_t1w/analysis_nnunet_47_official_metric_sweep.sh
sbatch baselines/models/nnunet_atlas_t1w/analysis_nnunet_49_reduce_official_sweep.sh
```

The reduce step reports **two** aggregates: per-case means (what the organizers'
`example_metrics.py` prints) and rank-then-aggregate across configurations (the
ISLES ranking convention). We do not have the ISLES'26 ranking document — the
README links Zenodo 10991145, which is the ISLES'24 document — so prefer a
configuration that is non-losing under both.
