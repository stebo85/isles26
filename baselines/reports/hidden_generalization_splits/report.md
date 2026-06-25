# Hidden-generalization split plan

Splits are leave-group-out plans for retraining/evaluating hidden-generalization risk.
The companion JSON is nnU-Net-compatible: copy one split into `splits_final.json` as index 0 for an isolated training run.

| index | split_id | kind | holdout | train n | val n | val chronicity |
|---:|---|---|---|---:|---:|---|
| 0 | center_SOOP | center_holdout | SOOP | 1282 | 168 | unknown:168 |
| 1 | center_R009 | center_holdout | R009 | 1339 | 111 | ge180d:61, lt180d:50 |
| 2 | center_R038 | center_holdout | R038 | 1356 | 94 | ge180d:18, lt180d:76 |
| 3 | center_R040 | center_holdout | R040 | 1364 | 86 | ge180d:18, lt180d:67, unknown:1 |
| 4 | center_R047 | center_holdout | R047 | 1402 | 48 | ge180d:46, lt180d:2 |
| 5 | center_R048 | center_holdout | R048 | 1406 | 44 | ge180d:44 |
| 6 | center_R069 | center_holdout | R069 | 1410 | 40 | lt180d:40 |
| 7 | center_R001 | center_holdout | R001 | 1411 | 39 | ge180d:39 |
| 8 | center_R059 | center_holdout | R059 | 1412 | 38 | unknown:38 |
| 9 | center_R004 | center_holdout | R004 | 1413 | 37 | ge180d:20, unknown:17 |
| 10 | center_R005 | center_holdout | R005 | 1413 | 37 | ge180d:35, lt180d:1, unknown:1 |
| 11 | center_R031 | center_holdout | R031 | 1413 | 37 | ge180d:1, lt180d:36 |
| 12 | center_R032 | center_holdout | R032 | 1415 | 35 | ge180d:8, lt180d:27 |
| 13 | center_R042 | center_holdout | R042 | 1415 | 35 | ge180d:34, lt180d:1 |
| 14 | center_R058 | center_holdout | R058 | 1415 | 35 | lt180d:35 |
| 15 | center_R027 | center_holdout | R027 | 1418 | 32 | ge180d:29, unknown:3 |
| 16 | center_R052 | center_holdout | R052 | 1418 | 32 | ge180d:32 |
| 17 | center_R011 | center_holdout | R011 | 1421 | 29 | unknown:29 |
| 18 | center_R067 | center_holdout | R067 | 1423 | 27 | ge180d:23, lt180d:4 |
| 19 | center_R010 | center_holdout | R010 | 1424 | 26 | unknown:26 |
| 20 | center_R028 | center_holdout | R028 | 1424 | 26 | ge180d:25, lt180d:1 |
| 21 | center_R071 | center_holdout | R071 | 1424 | 26 | lt180d:26 |
| 22 | center_R034 | center_holdout | R034 | 1426 | 24 | ge180d:24 |
| 23 | center_R049 | center_holdout | R049 | 1426 | 24 | lt180d:6, unknown:18 |
| 24 | center_R015 | center_holdout | R015 | 1427 | 23 | unknown:23 |
| 25 | center_R024 | center_holdout | R024 | 1429 | 21 | ge180d:21 |
| 26 | chronicity_ge180d | chronicity_holdout | ge180d | 802 | 648 | ge180d:648 |
| 27 | chronicity_lt180d | chronicity_holdout | lt180d | 1018 | 432 | lt180d:432 |
| 28 | chronicity_unknown | chronicity_holdout | unknown | 1080 | 370 | unknown:370 |
