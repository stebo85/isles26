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

They are baked to **`/opt/app/model`, never `/opt/ml/model`.** Grand Challenge
bind-mounts `/opt/ml/model` read-only for an uploaded model tarball, and it does
so **even when no tarball exists** — an empty directory that shadows whatever the
image put there. Submission `0b843f9e` (2026-08-11) failed exactly this way:
`FileNotFoundError: /opt/ml/model/dataset.json` was raised inside the FastAPI
lifespan, so uvicorn never bound port 4743 and the only thing Grand Challenge
reported was a health-check timeout. `resolve_model_dir()` now prefers the mount
when it genuinely holds a model and falls back to the image copy otherwise, so
both deployment styles work. Note `do_test_run.sh` mounts the host `model/` dir
over that path, so a *populated* local `model/` hides this class of bug — CI
empties it before the end-to-end test for that reason.

## Configuration

| env var | default | notes |
|---|---|---|
| `ISLES26_FOLDS` | `0,1,2,3,4` | full 5-fold ensemble; affordable once TTA is off |
| `ISLES26_THRESHOLD` | `0.35` | selected on the center-held-out surface against all 5 official metrics |
| `ISLES26_MIN_CC_VOXELS` | `20` | 26-connectivity; worth +0.030 lesion-F1 and -0.13 count error |
| `ISLES26_MIN_CC_MM3` | `0` | spacing-invariant alternative (0 = use voxels) |
| `ISLES26_DISABLE_TTA` | unset = **auto** | mirroring TTA **on with CUDA, off on CPU**; `1` forces off, `0` forces on |
| `ISLES26_FLATTEN_EMPTY` | `1` | constant soft map when the mask is empty |
| `ISLES26_NUM_THREADS` | unset | overrides the cgroup-derived CPU thread count (see below) |

These defaults are final. The two that were still open at the sanity-phase
deadline — TTA-vs-folds and the operating point — are resolved below.

## CPU runtime budget — resolved

Measured per case on CPU (8 cores), against the guidelines' ~10 minute budget.
The first table is the original sweep (`analysis_nnunet_60`); the second is the
same question re-measured through this container's own `inference.py` on two real
ATLAS cases (`analysis_nnunet_66`), which is what settled the trade.

| configuration | per invoke | verdict |
|---|---:|---|
| 5 folds + TTA | **12.9 min** | **over budget** |
| 5 folds, no TTA | 2.0 min | shipped |
| 1 fold + TTA | 2.8 min | fits, but see below |
| 1 fold, no TTA | 0.5 min | fits |

| case | fold4 + TTA | fold4, no TTA | 5 folds, no TTA |
|---|---:|---:|---:|
| ATLAS_0001 (173×213×181) | 183.9 s | 32.9 s | **114.2 s** |
| ATLAS_1316 (325×512×512) | 129.9 s | 59.4 s | **100.1 s** |

Grand Challenge does not guarantee a GPU, so the CPU figure governs. **Shipped
configuration: 5 folds, mirroring TTA off**, with ~5x headroom.

### Thread count must come from the cgroup, not `os.cpu_count()`

`docker run --cpus=N` — how Grand Challenge caps CPU — sets a cgroup CFS quota
and leaves `os.cpu_count()` reporting the *host's* core count. This container
originally handed that number straight to `torch.set_num_threads()`, so under any
CPU cap it oversubscribed and thrashed. Measured on one real case
(`analysis_nnunet_67`, 5-CPU cgroup on a 24-core node):

| threads requested | per invoke | output |
|---|---:|---|
| `os.cpu_count()` = 24 | 240.6 s | 81668 lesion voxels |
| `usable_cpus()` = 5 | **146.9 s** | 81668 lesion voxels |

39% faster, identical segmentation. `usable_cpus()` prefers an explicit
`ISLES26_NUM_THREADS`/`OMP_NUM_THREADS`, then the cpuset affinity mask, then the
CFS quota, and only falls back to the host count.

### Why not 1 fold + TTA

This file previously carried a pre-registered trigger: *revisit in favour of
1 fold + TTA if the ablation shows TTA is worth more than ~0.005 Dice.* The
ablation landed, TTA is worth **+0.0141 Dice**, the trigger fired, and the
revisit was done — it does not overturn the choice.

The reason is in the second table: **1 fold + TTA is slower than 5 folds without
it** (184 s vs 114 s on the same case and machine). The alternative therefore
costs ~60% more wall time to deliver a *single-model* prediction. For it to win,
the 5-fold ensemble would have to be worth less than TTA's +0.0141 Dice, and
fold-ensembling is the better-established of the two effects — it also reduces
variance across unseen centers, which is exactly this submission's risk (the test
set spans 60+ centers).

That ensemble gain **cannot be measured on our data** — every case was seen by 4
of the 5 members (`work/nested_holdout/RESULT.md`) — so this is a judgement under
one unmeasured quantity, stated rather than hidden.

**What it costs**, so the leaderboard is not a surprise. Per
`baselines/reports/tta_ablation/` (fold 0, n=281, TTA the only variable):

| metric | TTA | no TTA | delta |
|---|---:|---:|---:|
| Dice | 0.6605 | 0.6464 | −0.0141 |
| AVD mL | 6.515 | 7.371 | +0.856 |
| lesion-F1 (RQ) | 0.6601 | 0.6396 | −0.0205 |
| PR-AUC | 0.7768 | 0.7577 | −0.0191 |

Every out-of-fold number in this repository was measured *with* TTA. **Quote
Dice 0.6283 as an optimistic bound, not a forecast.**

### Reduced mirroring — measured, rejected

A subset of `allowed_mirroring_axes` costs 2^k passes instead of 8, so one axis
fits the budget at 5 folds. Measured on the same fold-0 surface
(`baselines/reports/reduced_mirroring/`), it does not buy enough: the gain is
strongly super-additive across axes, and no single axis recovers half of it.

| axes | passes | est. min/case @5 folds | ΔDice vs no-TTA | ΔPR-AUC |
|---|---:|---:|---:|---:|
| (0,) L-R | 2 | ~4.0 | +0.0041 | +0.0080 |
| (1,) P-A | 2 | ~4.0 | +0.0046 | +0.0045 |
| (2,) I-S | 2 | ~4.0 | +0.0023 | +0.0058 |
| (0,1) | 4 | ~7–8 | +0.0113 | +0.0171 |
| (0,1,2) full | 8 | 12.9 — over budget | +0.0141 | +0.0191 |

Absolute volume difference — one of the five ranked metrics — improves **only**
under full mirroring (−0.856 mL, t=−3.46); every subset is marginally *worse*
than no-TTA on it. The pre-registered rule (recover ≥ half the Dice gain,
regress nothing else beyond noise, keep ≥ 2x CPU headroom) rejects all four.
The question is closed; the container ships unchanged.

### Why the operating point did not move

`thr 0.35 / minCC 20` is **rank 1 of 54** by mean rank across the five official
metrics on the center-held-out surface (n=1452). On the deployment-matched
surface (no-TTA probabilities, fold 0, n=281) it is rank 16 of 54, where the
argmax is `thr0.25_k50`. It stays anyway:

- that surface is n=281 and single-fold, and the win is an in-sample argmax over
  a 54-point grid worth 0.0030 Dice — the exact error already retracted once in
  `docs/models/status.md` (the threshold-0.54 entry was chosen this way);
- its top 20 configs span mean rank 24.9–26.8, inside the table's own noise;
- PR-AUC is computed over the soft map and is identical (0.7417) at every
  threshold, so no threshold change can move one of the five metrics at all.

`thr0.35_k50` was the serious alternative (rank 5 on n=1452, rank 3 on the no-TTA
surface, trading ~0.003 Dice for lesion-F1) and was rejected under the same rule:
the gap is smaller than the noise it is measured with.
