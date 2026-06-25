# nnU-Net R3.0 default 250ep 5-fold CV

Cases: 1450

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 1448 | 0.6372 | 0.7438 | 0.2734 | 0.0000 | 0.9745 |
| iou | 1448 | 0.5192 | 0.5921 | 0.2609 | 0.0000 | 0.9504 |
| sensitivity | 1445 | 0.6718 | 0.7728 | 0.2917 | 0.0000 | 1.0000 |
| precision | 1439 | 0.6919 | 0.7966 | 0.2830 | 0.0000 | 1.0000 |
| specificity | 1450 | 0.9995 | 0.9999 | 0.0010 | 0.9841 | 1.0000 |

## Notes

R3.0 baseline consolidated out-of-fold validation over Dataset502_ATLASR30; postprocessing chose none.
