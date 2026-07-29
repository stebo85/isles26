# Adaptive per-case operating point: NEGATIVE. Do not pursue.

## Result

Nested on the center-held-out folds (fit on 4, apply to the 5th), n=1452:

| policy | Dice | AVD mL | abs count diff | lesion-F1 (RQ) |
|---|---:|---:|---:|---:|
| fixed baseline (thr 0.35, k=20) | 0.6283 | 6.3732 | 1.8685 | 0.5791 |
| per-case threshold from volume regression | 0.6273 | 6.5408 | 1.8836 | 0.5800 |
| + per-case min-CC rule | 0.6271 | 6.5403 | 1.8850 | 0.5805 |
| learned classifier over configurations | 0.6272 | 6.5497 | 1.9883 | 0.5652 |
| oracle (cheats) | 0.6433 | 4.6773 | 1.5172 | 0.6118 |

**Every policy is worse than the fixed baseline on the two metrics it was built to
improve.** The learned classifier is much worse on lesion count and RQ.

## Why — the oracle headroom was not real headroom

The motivating number was an oracle per-case minimum over 54 configurations. That
is a **selection maximum over 54 noisy options**, and it is upward-biased for
exactly the same reason an in-sample grid search is. Measured:

- Moving the threshold from 0.20 to 0.60 changes the predicted volume by a median
  of only **0.739 mL**, while the median volume error is **1.532 mL** — the lever
  is half the size of the error it is supposed to correct.
- The true volume lies inside the achievable range on only **17.5%** of cases. On
  the other 82.5% no threshold in the grid can reach it, so the inversion
  saturates at a grid endpoint. That is visible in the policy's own behaviour:
  602 cases pinned to thr 0.20 and 562 to thr 0.60, the two extremes.
- So the oracle's AVD of 4.51 mL is obtained by picking whichever endpoint
  happens to land nearer the truth per case. It is fitting noise, not structure.

The volume estimate itself is not the bottleneck and cannot be improved much: the
threshold-free softmax mass already correlates with true volume at Spearman
**+0.916**, essentially the same as the volume at a fixed threshold (+0.914).
Both have a mean absolute error of ~6.3 mL. The information needed to place the
operating point per case is not in the soft map.

## The one real (tiny) scrap

Choosing the threshold so the predicted volume matches the **softmax mass**
(a fixed rule, no fitting at all) gives AVD **6.306** vs **6.373** mL — a genuine
−0.067 mL, about 1% of AVD. Not worth the added moving part in a submission
container; recorded here so nobody re-derives it.

## Lesson

An oracle bound computed as a per-case min/max over many configurations is not an
achievability bound. Before chasing it, check that the knob has more leverage than
the error it is meant to correct. Here one line — the median achievable volume
span versus the median volume error — would have killed the idea before any model
was fitted.
