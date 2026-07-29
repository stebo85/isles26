# ISLES'26 submission container

Native-space T1w MRI -> binary infarct mask **and** a float32 probability map.

## Why both outputs

The challenge ranks on five metrics
([ezequieldlrosa/isles26](https://github.com/ezequieldlrosa/isles26), `utils/eval_utils.py`):

| # | metric | needs |
|---|---|---|
| 1 | Dice (`panoptica global_bin_dsc`) | binary mask |
| 2 | absolute volume difference (mL) | binary mask |
| 3 | absolute lesion count difference | binary mask |
| 4 | lesion-wise F1 = panoptica RQ, one-to-one matching at **IoU >= 0.25** | binary mask |
| 5 | **PR-AUC** (`precision_recall_curve` then `auc(recall, precision)`) | **soft map** |

A binary-only submission forfeits ~20% of the ranking signal, so `inference.py`
writes both. The output socket slugs are overridable (`ISLES26_MASK_SLUG`,
`ISLES26_PROB_SLUG`) because the organizers had not published the interface when
this was written — see `FORUM_QUESTIONS.md`.

## The three things most likely to break this

1. **Orientation.** The models were trained on `nib.as_closest_canonical` output
   and nnU-Net does not reorient at inference (`transpose_forward [0,1,2]`).
   The training release alone contains RAS 869 / LAS 575 / PSR 8 / PSL 1. A
   mirror or permutation failure emits a *plausible-looking* mask and costs
   ~94% of Dice (this project hit exactly that in commit `f8d88c1`).
   `geometry.py` handles it, and `test_geometry.py` proves the round-trip is
   exact for all 48 axis-aligned orientations and for real ATLAS cases in every
   orientation the release contains. **Re-run it on every rebuild.**
2. **Empty-GT PR-AUC.** `compute_pr_auc` returns `empty_value=1.0` on a
   lesion-free ground truth only if the soft map is *perfectly constant*, else
   `0.0`. A raw softmax is never constant, so a correct lesion-free prediction
   would score 0.0. When our binary mask is empty we therefore emit an
   all-zero (constant) map. Gated behind `ISLES26_FLATTEN_EMPTY=1`.
3. **Torch/nnU-Net version skew.** The checkpoints were written by
   `nnunetv2 2.6.2`. `requirements.txt` pins the whole stack; do not float it.

## Build

**Not on Sherlock.** The cluster has no `docker`/`podman`/`buildah`/`skopeo`,
and its apptainer cannot emit the `docker save` tarball Grand Challenge needs.
Build on a workstation or a small cloud VM; no GPU is required to build or to
run the local test.

```bash
# 1. stage the weights (on the cluster)
sbatch baselines/models/nnunet_atlas_t1w/analysis_nnunet_50_export_submission_weights.sh
#    -> work/submission_model/Dataset507_ATLASR30_CORRECTED/
tar -czf model.tar.gz -C work/submission_model/Dataset507_ATLASR30_CORRECTED .

# 2. copy model.tar.gz + this directory to the build host, then
docker build -t isles26-algorithm .
docker save isles26-algorithm | gzip -c > isles26-algorithm.tar.gz
```

Upload the image and `model.tar.gz` separately: Grand Challenge mounts the
archive at `/opt/ml/model`, which keeps the image small and lets the weights be
swapped without a rebuild.

## Test locally before every submission

```bash
# geometry only, no GPU, no weights  (8 checks incl. all 48 orientations)
PYTHONPATH=. python test_geometry.py

# end to end, needs weights
mkdir -p /tmp/in/images/t1w-brain-mri /tmp/out
cp <a raw native-space case>.nii.gz /tmp/in/images/t1w-brain-mri/
docker run --rm --gpus all \
  -v /tmp/in:/input:ro -v /tmp/out:/output \
  -v $PWD/model_unpacked:/opt/ml/model:ro \
  isles26-algorithm
```

The acceptance check that actually matters is not "did it run" but
**"does the mask it produces on the RAW native-space file reproduce the Dice we
recorded for that same case in cross-validation, to within 1e-3?"** That is the
only test that catches a silent mirror flip, and it is what
`analysis_nnunet_51_container_equivalence_test.sh` runs on the cluster.

## Configuration

| env var | default | notes |
|---|---|---|
| `ISLES26_MODEL_DIR` | `/opt/ml/model` | nnU-Net trained-model folder |
| `ISLES26_FOLDS` | `0,1,2,3,4` | 5-fold ensemble; measured ~30 s/case with TTA |
| `ISLES26_THRESHOLD` | `0.35` | selected on the center-held-out surface against all 5 official metrics |
| `ISLES26_MIN_CC_VOXELS` | `20` | min-component pruning, 26-connectivity. Worth **+0.030 lesion-F1 (RQ)** and -0.13 lesion-count error for -0.0 Dice |
| `ISLES26_MIN_CC_MM3` | `0` | spacing-invariant alternative to the voxel threshold (0 = use voxels) |
| `ISLES26_DISABLE_TTA` | `0` | mirroring TTA; there is no runtime reason to disable it |
| `ISLES26_FLATTEN_EMPTY` | `1` | constant soft map when the mask is empty |
| `ISLES26_MASK_SLUG` / `ISLES26_PROB_SLUG` | see above | output socket directories |

Runtime is not a constraint: measured 17 s min / 30 s median / 51 s p95 per case
for the full 5-fold + TTA path including startup, across eight GPU generations
(`sacct` on the `nnunet_atlas_val_predict` array). Only ~8 s of that is GPU
compute; the rest is fixed overhead that the `/invoke` server model amortizes.
