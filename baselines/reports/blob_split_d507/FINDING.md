# Blob splitting for the instance metrics: NEGATIVE. Do not pursue.

Center-held-out surface (Dataset507, n=1452), on top of the shipped operating
point (thr 0.35, min-CC 20). `h` is the h-maxima depth: how deep a probability
valley inside a predicted component must be before it is cut in two.

| h | Dice | AVD mL | abs count diff | lesion-F1 (RQ) | cases changed |
|---|---:|---:|---:|---:|---:|
| 0.00 (baseline) | 0.6283 | 6.3732 | 1.8685 | 0.5791 | — |
| 0.10 | 0.6252 | 6.3664 | 1.8664 | 0.5782 | 12 |
| 0.20 | 0.6254 | 6.3686 | 1.8657 | 0.5787 | 8 |
| 0.30 | 0.6253 | 6.3681 | 1.8657 | 0.5787 | 6 |
| 0.40 | 0.6254 | 6.3690 | 1.8657 | 0.5787 | 5 |

Every setting is worse on Dice and on RQ. Paired tests on the two metrics it was
meant to help: RQ -0.0009 (t=-2.28) at h=0.10, count difference -0.0021
(t=-0.69) -- i.e. a significant small loss on RQ and nothing on the count metric.

## Why — the intervention almost never fires

**It changes 5 to 13 cases out of 1452.** The h-maxima seeding finds two
sufficiently deep probability peaks inside a predicted component almost never.
Where the ground truth is fragmented into several instances, the model does not
produce a multi-peaked probability map over those instances -- it produces one
smooth, confident region. There is no valley to cut at.

The small Dice loss is mechanical: `watershed_line=True` deletes the boundary
voxels between fragments, which costs overlap for no instance-level gain.

## What this shares with the adaptive-operating-point result

Both experiments tried to recover instance structure that the soft map does not
contain. The per-case operating point failed because the threshold has less
leverage than the volume error; blob splitting fails because the probability
surface has no internal structure to split on. Same conclusion, reached twice
independently:

**The information needed to improve the instance-based and volume-based official
metrics is not present in the predicted probability map. These metrics are not
reachable by post-processing; they require a different model.**

That is the finding that should drive any remaining training effort.
