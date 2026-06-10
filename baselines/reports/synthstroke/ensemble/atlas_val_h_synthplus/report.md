# Ensemble Complementarity

All candidates are tuned on the same validation cases over the requested threshold/min-CC grid. Treat these numbers as a relative comparison, not as an unbiased deployment estimate.
Probabilities are the raw saved maps and are evaluated without extra brain masking; this is internally consistent across candidates.

Usable cases: 194
Skipped cases: 0

| candidate | mean_dice | lesion_f1 | best(thr,cc) |
|---|---:|---:|---|
| h | 0.5661 | 0.4502 | (0.30,10) |
| ens_mean_all | 0.5560 | 0.4987 | (0.40,10) |
| synthplus | 0.4589 | 0.5655 | (0.30,10) |

## Oracle

Oracle best-per-case Dice: 0.5833 (oracle - best single h: 0.0172; oracle - ens_mean_all: 0.0272; ens_mean_all - best single: -0.0101).

## Pairwise Complementarity

| model A | model B | frac A dice > B | mean abs dice diff | Pearson r |
|---|---|---:|---:|---:|
| h | synthplus | 0.7629 | 0.1416 | 0.8281 |
