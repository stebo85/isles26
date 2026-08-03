# ISLES'26 submission container

Native-space T1w MRI -> binary infarct mask **and** a float32 probability map.

Structure follows the organizers' official template
([isles26-docker-template](https://github.com/ezequieldlrosa/isles26-docker-template)).

## The interface, verbatim from the template

    inputs   /input/images/t1-brain-mri          slug: t1-brain-mri
             /input/stroke-metadata.json         slug: stroke-metadata
             /input/inputs.json                  system-generated, selects the interface
    outputs  /output/images/stroke-lesion-segmentation
             /output/images/lesion-probability-map

Grand Challenge runs this as an **HTTP server on port 4743**, not as a batch job:
it polls `GET /health` until 200 (~5 min budget), then calls `POST /invoke` once
per case, expecting 201. `app.py` provides that server; `inference.py` provides
`init_model()` (called once at startup — load weights here, not in `/invoke`) and
`run(model)`.

Both outputs are required. PR-AUC is one of five ranked metrics and is computed
over the continuous map, so a mask-only submission silently forfeits ~20% of the
ranking signal.

## The three things most likely to break this

1. **Orientation.** Models were trained on `nib.as_closest_canonical` output and
   nnU-Net does not reorient at inference (`transpose_forward [0,1,2]`); the
   training release alone contains RAS 869 / LAS 575 / PSR 8 / PSL 1. A mirror or
   permutation error emits a *plausible-looking* mask and costs ~94% of Dice —
   this project hit exactly that in commit `f8d88c1`. `geometry.py` derives the
   exact inverse; `test_geometry.py` proves the round-trip over all 48
   axis-aligned orientations and runs as a Docker build step.
2. **Empty-GT PR-AUC.** `compute_pr_auc` returns `empty_value=1.0` on a
   lesion-free ground truth only if the soft map is *perfectly constant*, else
   `0.0`. When our mask is empty we emit an all-zero map
   (`ISLES26_FLATTEN_EMPTY=1`). Note the model is confidently wrong on lesion-free
   scans and a confidence gate cannot fix it — see
   `baselines/reports/qc_test_predictions/FINDING.md`.
3. **Version skew.** Checkpoints were written by `nnunetv2 2.6.2`;
   `requirements.txt` pins the whole stack. Do not float it.

## Build and test

The image is built by GitHub Actions
([`publish-container.yml`](../../.github/workflows/publish-container.yml)) — the
cluster has no docker/podman/buildah and its apptainer cannot emit a `docker save`
tarball. CI runs the geometry test as a gate, downloads the weights from the
public Hugging Face release and verifies every sha256, builds, runs the
organizers' `do_test_run.sh` through the real health-then-invoke lifecycle, and
uploads `isles26-algorithm.tar.gz`.

Locally, with `model/` populated:

    ./do_build.sh        # build
    ./do_test_run.sh     # full server lifecycle against test/input/interf0
    ./do_save.sh         # produce the upload tarball

Weights are **baked into the image** rather than mounted from Grand Challenge's
separate `model.tar.gz`: the image grows ~600 MB, but the container is
self-contained and there is no second upload to go stale against a hard deadline.
`model/` is gitignored; populate it from
`https://huggingface.co/sbollmann/isles26-nnunet-d507-topk10`.

## Configuration

| env var | default | notes |
|---|---|---|
| `ISLES26_FOLDS` | `0,1,2,3,4` | **may be reduced** — see the CPU budget below |
| `ISLES26_THRESHOLD` | `0.35` | selected on the center-held-out surface against all 5 official metrics |
| `ISLES26_MIN_CC_VOXELS` | `20` | 26-connectivity; worth +0.030 lesion-F1 and -0.13 count error |
| `ISLES26_MIN_CC_MM3` | `0` | spacing-invariant alternative (0 = use voxels) |
| `ISLES26_DISABLE_TTA` | `0` | mirroring TTA |
| `ISLES26_FLATTEN_EMPTY` | `1` | constant soft map when the mask is empty |

## CPU runtime budget — unresolved

The guidelines quote **~10 minutes on CPU**, and the template says the GPU is used
"when enabled", so a GPU is not guaranteed. Our 5-fold + TTA configuration is
~30 s/case *on a GPU* (~8 s of it GPU compute); on CPU, 5 folds x 8 mirror
configurations may exceed the budget. `analysis_nnunet_60_cpu_runtime_budget.sh`
measures 5-fold/1-fold x TTA-on/off on a median and a large case. **Set
`ISLES26_FOLDS` and `ISLES26_DISABLE_TTA` from that measurement** — dropping folds
is a real accuracy trade and should not be guessed.
