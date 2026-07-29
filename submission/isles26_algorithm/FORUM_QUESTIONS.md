# Questions for the ISLES'26 organizers

Post to https://isles-26.grand-challenge.org/forum/topics/. These gate the
container design, so they are on the critical path — the sanity phase opens
2026-07-30 and submission closes 2026-08-15.

## 1. Output interface: is a soft probability map a required second output?

The challenge page lists the algorithm output as "Binary infarct segmentation
mask" only. But `utils/eval_utils.py:compute_pr_auc` scores a *continuous* map,
PR-AUC is listed as one of the five ranking metrics, and
`utils/example_metrics.py` passes `soft_prediction_map` and
`binary_segmentation_mask` as two independent arrays while its T1 glob excludes
files containing `lesion_pred`.

Please confirm the exact input and output socket slugs, file formats and dtypes,
and whether a soft map is expected as a second output.

## 2. PR-AUC on lesion-free cases

`compute_pr_auc` returns `empty_value=1.0` when the ground truth is empty **only
if** `np.all(pred_flat == pred_flat[0])`, and `0.0` otherwise. A real network
softmax is never exactly constant, so an algorithm that correctly finds no lesion
scores 0.0, while an algorithm that emits a hard constant scores 1.0. Dice and
lesion-F1 on the same case both return 1.0.

Is emitting an exactly-constant map when the algorithm predicts no lesion the
intended behaviour? If not, would you consider `average_precision_score` or an
explicit empty-case rule? We would rather implement the intended behaviour than
optimise an artefact.

Related: does the hidden test set contain lesion-free cases, and roughly what
fraction?

## 3. PR-AUC definition

The code computes `auc(recall, precision)` (trapezoidal), not
`average_precision_score`. At the very low lesion prevalence of this task the two
differ measurably. Please confirm the trapezoidal form is what the leaderboard
uses.

## 4. Metadata delivery

The task description says algorithms receive "lesion chronicity labels (tabular
values)". How is this delivered to the container — which socket, which filename,
which schema? Note the training CSVs use a `SITE` column while the challenge text
refers to `CENTER`, and `DAYS_POST_STROKE` is missing or non-positive for ~25% of
the public training cases.

## 5. Ranking computation

The README links Zenodo record 10991145 for the ranking, but that record is the
ISLES'24 document. Could you point to the ISLES'26 ranking scheme — in
particular whether it is rank-then-aggregate per metric, and how the two
unbounded error metrics (absolute volume difference, absolute lesion count
difference) are handled.

## 6. Per-job runtime and hardware limits

What GPU and what per-job wall-clock limit apply in the test phase? Our current
configuration measures ~30 s/case for a 5-fold ensemble with test-time
augmentation.
