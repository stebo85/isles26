# nnU-Net R3.0 default 1000ep 5-fold CV

Cases: 1450

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 1448 | 0.6528 | 0.7634 | 0.2787 | 0.0000 | 0.9743 |
| iou | 1448 | 0.5390 | 0.6173 | 0.2674 | 0.0000 | 0.9498 |
| sensitivity | 1445 | 0.6769 | 0.7890 | 0.2970 | 0.0000 | 1.0000 |
| precision | 1434 | 0.7195 | 0.8274 | 0.2760 | 0.0000 | 1.0000 |
| specificity | 1450 | 0.9996 | 0.9999 | 0.0009 | 0.9866 | 1.0000 |

## Notes

nnU-Net consolidated out-of-fold validation over Dataset502_ATLASR30; 1450 usable converted cases from the 1453-session public training release.
