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


def main() -> int:
    tests = [
        test_insert_small_hypointense_component,
        test_noop_when_probability_zero,
        test_respects_padding_and_existing_lesion,
        test_candidate_excludes_darkest_voxels,
    ]
    for test in tests:
        test()
        print(f"[ok] {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
