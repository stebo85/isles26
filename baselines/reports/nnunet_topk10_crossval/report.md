# nnU-Net R3.0 Dice+TopK10 1000ep 5-fold CV

Cases: 1450

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 1447 | 0.6551 | 0.7631 | 0.2784 | 0.0000 | 0.9721 |
| iou | 1447 | 0.5419 | 0.6170 | 0.2686 | 0.0000 | 0.9457 |
| sensitivity | 1445 | 0.6729 | 0.7829 | 0.2956 | 0.0000 | 1.0000 |
| precision | 1424 | 0.7359 | 0.8411 | 0.2686 | 0.0000 | 1.0000 |
| specificity | 1450 | 0.9996 | 0.9999 | 0.0008 | 0.9902 | 1.0000 |

## Notes

Current best in-distribution model: Dice+TopK10, 1000 epochs, 5-fold ensemble-ready cross-validation over Dataset502_ATLASR30.
