# QC on RAW native-space cases — what the container actually produces

9 cases sampled evenly across the ground-truth lesion-size range (0.0 to 388 mL),
each predicted with **only the fold that held it out**, run through the
submission container's own `inference.py` on the RAW files in
`data/ATLAS3_Training_Raw/` (not the canonicalised training copies). Panel:
`qc_panel.png`.

| case | fold | GT mL | pred mL | Dice |
|---|---:|---:|---:|---:|
| ATLAS_0809 | 4 | 0.00 | 0.02 | **0.000** |
| ATLAS_0310 | 1 | 0.45 | 0.43 | 0.804 |
| ATLAS_0712 | 1 | 1.17 | 0.81 | 0.561 |
| ATLAS_0800 | 0 | 2.42 | 2.82 | 0.806 |
| ATLAS_1247 | 3 | 5.39 | 3.77 | 0.409 |
| ATLAS_0635 | 1 | 14.81 | 12.49 | 0.884 |
| ATLAS_1138 | 2 | 34.78 | 33.01 | 0.857 |
| ATLAS_1372 | 4 | 71.63 | 42.96 | 0.617 |
| ATLAS_0104 | 3 | 387.63 | 383.72 | 0.851 |

## What is working

Geometry is correct on real native-space data: contours sit on the anatomy, and
predicted volumes track ground truth closely on medium and large lesions
(33.0 vs 34.8, 383.7 vs 387.6 mL). Mid-size lesions reach Dice 0.85-0.88.

## Three weaknesses, all visible in the panel

**1. Confidently wrong on lesion-free scans — the most expensive failure.**
ATLAS_0809 has no lesion; the container emits a 0.02 mL (21-voxel) blob, just
above the 20-voxel filter. Under the official empty-value convention this is not
a small error, it is a cliff:

    predict empty   -> Dice 1.0, lesion-F1 1.0, PR-AUC 1.0
    predict anything -> Dice 0.0, lesion-F1 0.0, PR-AUC 0.0

**One tiny false positive swings three of the five ranked metrics from 1.0 to
0.0.** Across the center-held-out set only **2 of 5** lesion-free cases are
correctly predicted empty at the shipped operating point (3 of 5 at min-CC 35).

*It is not fixable with a confidence gate.* On lesion-free scans the model is as
confident as it is on real lesions (median max-probability 0.9938 vs 1.0000 for
true sub-1 mL lesions). Every threshold that blanks even 1 of the 5 lesion-free
cases destroys 11-22 of the 317 genuine sub-1 mL lesions — a plainly bad trade.
Raising min-CC is the only lever, and the sweep already prices it: k=35 is
rank-neutral overall.

**Risk, stated plainly:** at ATLAS's 0.34% lesion-free rate this costs almost
nothing in aggregate. If the hidden test set contains materially more lesion-free
scans -- plausible, since ISLES'26 adds acute cases from 60+ centers -- the cost
scales linearly on *three* metrics at once. This is the single largest unhedged
risk in the current configuration.

**2. Large lesions under-segment.** ATLAS_1372: 43.0 predicted against 71.6 mL,
Dice 0.617, and the soft map over the affected cortical ribbon is diffuse and
low-probability rather than confident. Consistent with the separately measured
deployment-path result (-4.1% predicted volume on the 15 largest cases) and with
the population-level under-segmentation bias.

**3. Small lesions are erratic, not uniformly poor.** 0.804, 0.561, 0.806, 0.409
across four sub-6 mL cases. The failures are localisation-partial rather than
total misses -- the prediction overlaps part of the ground truth and misses the
rest (ATLAS_1247 clearly shows one of two ground-truth components recovered).

## Honest note on sampling

Cases were selected by evenly spanning the GT volume range, before any result was
seen. A panel of large confluent lesions would have shown Dice 0.85-0.88
throughout and been misleading; the sub-6 mL rows are where the real variance is.
