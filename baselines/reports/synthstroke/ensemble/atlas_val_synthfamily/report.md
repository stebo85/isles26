# Ensemble Complementarity

All candidates are tuned on the same validation cases over the requested threshold/min-CC grid. Treat these numbers as a relative comparison, not as an unbiased deployment estimate.
Probabilities are the raw saved maps and are evaluated without extra brain masking; this is internally consistent across candidates.

Usable cases: 194
Skipped cases: 0

| candidate | mean_dice | lesion_f1 | best(thr,cc) |
|---|---:|---:|---|
| ens_mean_all | 0.5718 | 0.5138 | (0.40,10) |
| ens_mean_h+i | 0.5699 | 0.4723 | (0.30,10) |
| ens_mean_h+g | 0.5693 | 0.5177 | (0.50,10) |
| ens_mean_h+e | 0.5689 | 0.5195 | (0.50,10) |
| h | 0.5661 | 0.4502 | (0.30,10) |
| ens_mean_i+g | 0.5627 | 0.4713 | (0.40,10) |
| ens_mean_i+e | 0.5579 | 0.4577 | (0.40,10) |
| i | 0.5568 | 0.4904 | (0.30,10) |
| ens_mean_g+e | 0.5251 | 0.4494 | (0.50,10) |
| g | 0.5174 | 0.4018 | (0.60,10) |
| e | 0.4992 | 0.4138 | (0.60,10) |

## Oracle

Oracle best-per-case Dice: 0.5955 (oracle - best single h: 0.0293; oracle - ens_mean_all: 0.0237; ens_mean_all - best single: 0.0057).

## Pairwise Complementarity

| model A | model B | frac A dice > B | mean abs dice diff | Pearson r |
|---|---|---:|---:|---:|
| h | i | 0.5309 | 0.0451 | 0.9747 |
| h | g | 0.7165 | 0.0722 | 0.9342 |
| h | e | 0.7010 | 0.0839 | 0.9198 |
| i | g | 0.6495 | 0.0834 | 0.9168 |
| i | e | 0.6186 | 0.0972 | 0.8907 |
| g | e | 0.5515 | 0.0453 | 0.9671 |
