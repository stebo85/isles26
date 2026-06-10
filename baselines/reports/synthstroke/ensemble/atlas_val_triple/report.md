# Ensemble Complementarity

All candidates are tuned on the same validation cases over the requested threshold/min-CC grid. Treat these numbers as a relative comparison, not as an unbiased deployment estimate.
Probabilities are the raw saved maps and are evaluated without extra brain masking; this is internally consistent across candidates.

Usable cases: 183
Skipped cases: 11

| candidate | mean_dice | lesion_f1 | best(thr,cc) |
|---|---:|---:|---|
| nnunet | 0.6370 | 0.6787 | (0.40,10) |
| ens_mean_h+nnunet | 0.6259 | 0.6280 | (0.50,10) |
| ens_mean_synthplus+nnunet | 0.6116 | 0.6484 | (0.40,10) |
| ens_mean_all | 0.6105 | 0.6485 | (0.40,10) |
| h | 0.5630 | 0.4597 | (0.40,10) |
| ens_mean_h+synthplus | 0.5544 | 0.5984 | (0.50,10) |
| synthplus | 0.4566 | 0.5626 | (0.30,10) |

## Oracle

Oracle best-per-case Dice: 0.6573 (oracle - best single nnunet: 0.0202; oracle - ens_mean_all: 0.0467; ens_mean_all - best single: -0.0265).

## Pairwise Complementarity

| model A | model B | frac A dice > B | mean abs dice diff | Pearson r |
|---|---|---:|---:|---:|
| h | synthplus | 0.7596 | 0.1420 | 0.8260 |
| h | nnunet | 0.2077 | 0.1070 | 0.8545 |
| synthplus | nnunet | 0.0984 | 0.1939 | 0.7770 |

Skipped case reasons are recorded in `summary.json`.
