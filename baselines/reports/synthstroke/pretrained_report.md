# SynthStroke Pretrained Benchmark

Missing report.json files are shown as n/a.
Reference rows use the fixed headline values specified for this benchmark.

| model | eval set | n | Dice mean | Dice median | lesion-F1 mean | lesion-F1 median | source |
|---|---|---:|---:|---:|---:|---:|---|
| nnU-Net baseline | atlas_val | n/a | 0.640 | n/a | 0.654 | n/a | given reference |
| nnU-Net baseline | soop_t1w | n/a | 0.188 | n/a | 0.201 | n/a | given reference |
| DeepISLES reference | soop_trace | n/a | 0.540 | n/a | 0.582 | n/a | given reference |
| synthstroke-baseline | atlas_val | 194 | 0.334 | 0.285 | 0.240 | 0.159 | [report](synthstroke-baseline/atlas_val/report.md) |
| synthstroke-baseline | soop_trace | 12 | 0.044 | 0.000 | 0.021 | 0.000 | [report](synthstroke-baseline/soop_trace/report.md) |
| synthstroke-baseline | soop_t1w | 12 | 0.099 | 0.000 | 0.134 | 0.000 | [report](synthstroke-baseline/soop_t1w/report.md) |
| synthstroke-synth | atlas_val | 194 | 0.403 | 0.416 | 0.599 | 0.667 | [report](synthstroke-synth/atlas_val/report.md) |
| synthstroke-synth | soop_trace | 12 | 0.261 | 0.150 | 0.128 | 0.148 | [report](synthstroke-synth/soop_trace/report.md) |
| synthstroke-synth | soop_t1w | 12 | 0.145 | 0.000 | 0.222 | 0.091 | [report](synthstroke-synth/soop_t1w/report.md) |
| synthstroke-synth-pseudo | atlas_val | 194 | 0.381 | 0.373 | 0.333 | 0.267 | [report](synthstroke-synth-pseudo/atlas_val/report.md) |
| synthstroke-synth-pseudo | soop_trace | 12 | 0.262 | 0.158 | 0.167 | 0.182 | [report](synthstroke-synth-pseudo/soop_trace/report.md) |
| synthstroke-synth-pseudo | soop_t1w | 12 | 0.180 | 0.006 | 0.284 | 0.107 | [report](synthstroke-synth-pseudo/soop_t1w/report.md) |
| synthstroke-synth-plus | atlas_val | 194 | 0.458 | 0.521 | 0.515 | 0.500 | [report](synthstroke-synth-plus/atlas_val/report.md) |
| synthstroke-synth-plus | soop_trace | 12 | 0.447 | 0.491 | 0.478 | 0.472 | [report](synthstroke-synth-plus/soop_trace/report.md) |
| synthstroke-synth-plus | soop_t1w | 12 | 0.217 | 0.043 | 0.231 | 0.225 | [report](synthstroke-synth-plus/soop_t1w/report.md) |
| synthstroke-qatlas | atlas_val | 194 | 0.282 | 0.210 | 0.329 | 0.286 | [report](synthstroke-qatlas/atlas_val/report.md) |
| synthstroke-qatlas | soop_trace | 12 | 0.052 | 0.001 | 0.187 | 0.077 | [report](synthstroke-qatlas/soop_trace/report.md) |
| synthstroke-qatlas | soop_t1w | 12 | 0.115 | 0.000 | 0.178 | 0.091 | [report](synthstroke-qatlas/soop_t1w/report.md) |
| synthstroke-qsynth | atlas_val | 194 | 0.367 | 0.351 | 0.461 | 0.444 | [report](synthstroke-qsynth/atlas_val/report.md) |
| synthstroke-qsynth | soop_trace | 12 | 0.246 | 0.129 | 0.216 | 0.205 | [report](synthstroke-qsynth/soop_trace/report.md) |
| synthstroke-qsynth | soop_t1w | 12 | 0.164 | 0.012 | 0.287 | 0.122 | [report](synthstroke-qsynth/soop_t1w/report.md) |

Notes:
- nnU-Net soop is listed under soop_t1w because that baseline uses T1w inputs warped into TRACE space.
- DeepISLES soop is listed under soop_trace because its predictions are evaluated in native soop_bench/TRACE space.
