#!/usr/bin/env python3
"""Direct smoke tests for synthetic_lesion_augmentation.py.

Run from the repo root:
  LD_LIBRARY_PATH=/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-} \
    .venv-nnunet/bin/python baselines/models/nnunet_atlas_t1w/nnunet_helpers/test_synthetic_lesion_augmentation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from baselines.models.nnunet_atlas_t1w.nnunet_helpers.synthetic_lesion_augmentation import (
    SyntheticLesionInsertionTransform,
)


def _make_multi_tissue_patch(shape=(36, 36, 36)):
    """Z-scored multi-tissue phantom, the only kind that can expose contrast bugs.

    A constant-intensity phantom is useless here: the lesion ends up darker than
    its surroundings whether the target intensity tracks the requested
    contrast-to-noise ratio or is pinned to a low brain quantile, so it passes
    either way. This phantom mimics Dataset502 preprocessing (nnU-Net
    ZScoreNormalization with use_mask_for_norm=True): three tissue levels plus
    mild noise, z-scored inside the brain mask, exact zero outside.
    """
    image = torch.zeros((1, *shape), dtype=torch.float32)
    segmentation = torch.zeros((1, *shape), dtype=torch.int16)

    brain = torch.zeros(shape, dtype=torch.bool)
    brain[4:32, 4:32, 4:32] = True
    raw = torch.zeros(shape, dtype=torch.float32)
    raw[brain] = 1.00                      # white-matter-like
    raw[8:28, 8:28, 8:28] = 0.70           # grey-matter-like
    raw[14:22, 14:22, 14:22] = 0.45        # CSF-like
    raw = raw + 0.03 * torch.randn(shape, dtype=torch.float32)

    vals = raw[brain]
    image[0][brain] = (vals - vals.mean()) / vals.std(unbiased=False)
    return image, segmentation


def _dilate(mask):
    return torch.nn.functional.max_pool3d(mask[None, None].float(), 3, stride=1, padding=1)[0, 0] > 0


def _measure_contrast(image, lesion, brain, region=None):
    """Realized hypointense contrast-to-noise ratio, positive = darker than ring.

    Mirrors the transform's own reference: perilesional 1-voxel ring for the
    level, global brain std for the scale. `region` selects which lesion voxels
    to average -- the core is set exactly to the target, the boundary shell is
    alpha-blended with the original tissue to fake partial volume.
    """
    img = image[0]
    ring = _dilate(lesion) & ~lesion & brain
    reference = ring if int(ring.sum().item()) >= 8 else (brain & ~lesion)
    brain_std = img[brain & ~lesion].std(unbiased=False)
    measured = lesion if region is None else region
    return float((img[reference].mean() - img[measured].mean()) / brain_std)


def _make_patch(shape=(32, 32, 32)):
    image = torch.zeros((1, *shape), dtype=torch.float32)
    segmentation = torch.zeros((1, *shape), dtype=torch.int16)
    image[:, 6:26, 6:26, 6:26] = 1.0
    ramp = torch.linspace(-0.2, 0.2, 20, dtype=torch.float32)
    image[:, 6:26, 6:26, 6:26] += ramp[None, :, None, None]
    return image, segmentation


def test_insert_small_hypointense_component():
    torch.manual_seed(2)
    image, segmentation = _make_patch()
    transform = SyntheticLesionInsertionTransform(
        apply_probability=1.0,
        max_lesions=1,
        size_bins=((12, 18, 1.0),),
        min_candidate_voxels=8,
        contrast_std=(1.2, 1.2),
        texture_std_fraction=(0.0, 0.0),
        rim_shift_std_fraction=(0.0, 0.0),
    )
    out = transform(image=image.clone(), segmentation=segmentation.clone())
    lesion = out["segmentation"][0] == 1
    inserted = int(lesion.sum().item())
    assert 12 <= inserted <= 18, inserted
    assert transform.last_inserted_voxels == [inserted]

    lesion_mean = out["image"][0][lesion].mean()
    brain_nonlesion = (out["image"][0].abs() > 1e-6) & ~lesion
    ref_mean = out["image"][0][brain_nonlesion].mean()
    assert lesion_mean < ref_mean, (float(lesion_mean), float(ref_mean))


def test_noop_when_probability_zero():
    torch.manual_seed(3)
    image, segmentation = _make_patch()
    transform = SyntheticLesionInsertionTransform(apply_probability=0.0)
    out = transform(image=image.clone(), segmentation=segmentation.clone())
    assert torch.equal(out["image"], image)
    assert torch.equal(out["segmentation"], segmentation)
    assert transform.last_inserted_voxels == []


def test_respects_padding_and_existing_lesion():
    torch.manual_seed(4)
    image, segmentation = _make_patch()
    segmentation[:, :5] = -1
    segmentation[:, 12:15, 12:15, 12:15] = 1
    transform = SyntheticLesionInsertionTransform(
        apply_probability=1.0,
        max_lesions=1,
        size_bins=((20, 20, 1.0),),
        min_candidate_voxels=8,
        texture_std_fraction=(0.0, 0.0),
        rim_shift_std_fraction=(0.0, 0.0),
    )
    out = transform(image=image.clone(), segmentation=segmentation.clone())
    assert torch.equal(out["segmentation"][0, :5], segmentation[0, :5])
    assert torch.all(out["segmentation"][0, 12:15, 12:15, 12:15] == 1)
    assert int((out["segmentation"][0] == 1).sum().item()) >= 27


def test_candidate_excludes_darkest_voxels():
    image, segmentation = _make_patch()
    image[:, 10:22, 10:22, 10:22] = 0.05
    transform = SyntheticLesionInsertionTransform(
        apply_probability=1.0,
        candidate_low_quantile=0.20,
        min_candidate_voxels=8,
    )
    valid = segmentation[0] >= 0
    brain = transform._brain_mask(image, valid)
    candidate = transform._candidate_mask(image, segmentation[0], valid, brain)
    assert not bool(candidate[10:22, 10:22, 10:22].any().item())
    assert bool(candidate[22:26, 22:26, 22:26].any().item())


def _insert_with_contrast(seed: int, contrast: float):
    """Insert one lesion at a fixed geometry.

    Returns (core cnr, whole-lesion cnr, transform).
    """
    torch.manual_seed(seed)
    image, segmentation = _make_multi_tissue_patch()
    transform = SyntheticLesionInsertionTransform(
        apply_probability=1.0,
        max_lesions=1,
        size_bins=((120, 160, 1.0),),
        contrast_std=(contrast, contrast),
        low_quantile=(0.05, 0.05),
        texture_std_fraction=(0.0, 0.0),
        rim_shift_std_fraction=(0.0, 0.0),
    )
    valid = segmentation[0] >= 0
    brain = transform._brain_mask(image, valid)
    out = transform(image=image.clone(), segmentation=segmentation.clone())
    lesion = out["segmentation"][0] == 1
    assert bool(lesion.any().item())
    eroded = transform._erode(lesion)
    core = eroded if bool(eroded.any().item()) else lesion
    return (
        _measure_contrast(out["image"], lesion, brain, region=core),
        _measure_contrast(out["image"], lesion, brain),
        transform,
    )


def test_realized_contrast_tracks_contrast_std():
    """The regression guard for the inert-contrast bug (audit finding 14).

    Previously the target intensity was min(ref_mean - c * ring_std, low_quantile)
    and the low quantile always won, so c had no effect whatsoever. Both runs
    below share a seed, hence identical lesion geometry and identical draws for
    every other randomized quantity; the only difference is c.
    """
    low_core, low_lesion, low_tf = _insert_with_contrast(11, 0.5)
    high_core, high_lesion, high_tf = _insert_with_contrast(11, 2.5)
    assert low_tf.last_inserted_voxels == high_tf.last_inserted_voxels, (
        low_tf.last_inserted_voxels,
        high_tf.last_inserted_voxels,
    )
    # Hypointense in both cases, but measurably more so when more is requested.
    assert low_core > 0.0, low_core
    assert high_core - low_core >= 1.5, (low_core, high_core)
    assert high_lesion - low_lesion >= 0.75, (low_lesion, high_lesion)


def test_realized_contrast_matches_request():
    """Realized CNR must track the request across the whole operating range.

    Checked at both ends of the default contrast_std span: the plausibility
    floor must not bind at the dark end, and nothing may inflate the subtle,
    near-isointense end that this augmentation exists to produce.
    """
    for requested in (0.6, 2.2):
        core_cnr, lesion_cnr, _ = _insert_with_contrast(23, requested)
        assert abs(core_cnr - requested) <= 0.15, (requested, core_cnr)
        # The whole-lesion average also folds in the alpha-blended boundary
        # shell, so it is weaker than the core but must stay hypointense.
        assert 0.0 < lesion_cnr <= core_cnr + 0.05, (requested, lesion_cnr, core_cnr)


def test_rejects_hyperintense_contrast_range():
    try:
        SyntheticLesionInsertionTransform(contrast_std=(-0.5, 1.0))
    except ValueError:
        return
    raise AssertionError("negative contrast_std must be rejected")


def main() -> int:
    tests = [
        test_insert_small_hypointense_component,
        test_noop_when_probability_zero,
        test_respects_padding_and_existing_lesion,
        test_candidate_excludes_darkest_voxels,
        test_realized_contrast_tracks_contrast_std,
        test_realized_contrast_matches_request,
        test_rejects_hyperintense_contrast_range,
    ]
    for test in tests:
        test()
        print(f"[ok] {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
