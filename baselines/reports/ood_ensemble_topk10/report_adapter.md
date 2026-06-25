# OOD blend R3.0 TopK10 nnU-Net + synth-plus on T1w

Cases: 12

| selected model | w_nn | threshold | mean Dice | median Dice |
|---|---:|---:|---:|---:|
| blend w_nn=0.6 | 0.6 | 0.30 | 0.3006 | 0.2566 |

Caveat: blend weight and threshold were selected on this same soop_bench sweep.

Exploratory soop_bench blend; challenge-realistic T1w input, but threshold/weight selected on the same 12 cases.
