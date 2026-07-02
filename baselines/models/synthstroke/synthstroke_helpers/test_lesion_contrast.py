#!/usr/bin/env python3
"""Smoke tests for LesionContrastD in our_train.py.

Run from repo root with the SynthStroke environment, for example:

  module load devel python/3.12.1 py-pytorch/2.4.1_py312
  LD_LIBRARY_PATH=/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-} \
    PYTHONPATH=work/synthstroke/repo:. \
    .venv-synthstroke/bin/python \
    baselines/models/synthstroke/synthstroke_helpers/test_lesion_contrast.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from baselines.models.synthstroke.synthstroke_helpers.our_train import LesionContrastD


LESION_IDX = 2


def _make_patch(shape=(32, 32, 32)):
    image = torch.zeros((1, *shape), dtype=torch.float32)
    label = torch.zeros((1, *shape), dtype=torch.long)
    label[:, 5:27, 5:27, 5:27] = 1
    label[:, 12:20, 12:20, 12:20] = LESION_IDX

    # Smooth tissue gradient gives a stable nonzero tissue std. The lesion starts
    # visually uninformative: same intensity scale as surrounding tissue.
    x = torch.linspace(-0.5, 0.5, shape[0], dtype=torch.float32)
    y = torch.linspace(-0.25, 0.25, shape[1], dtype=torch.float32)
    z = torch.linspace(-0.125, 0.125, shape[2], dtype=torch.float32)
    grid = 1.0 + x[:, None, None] + y[None, :, None] + z[None, None, :]
    brain = label[0] > 0
    image[0][brain] = grid[brain]
    return image, label


def _dilate(mask, iterations=3):
    out = mask.bool()
    for _ in range(iterations):
        out = torch.nn.functional.max_pool3d(out[None, None].float(), 3, stride=1, padding=1)[0, 0] > 0
    return out


def _measure_cnr(image, label):
    img = image[0]
    lab = label[0]
    lesion = lab == LESION_IDX
    tissue = (lab > 0) & ~lesion
    ring = _dilate(lesion, 3) & ~lesion & tissue
    contrast = img[lesion].mean() - img[ring].mean()
    return float(contrast / img[tissue].std(unbiased=False))


def test_noop_when_probability_zero():
    image, label = _make_patch()
    transform = LesionContrastD(
        LESION_IDX,
        prob=0.0,
        cnr_lo=1.25,
        cnr_hi=1.25,
        texture_std_fraction=(0.0, 0.0),
    )
    out = transform({"image": image.clone(), "label": label.clone()})
    assert torch.equal(out["image"], image)
    assert torch.equal(out["label"], label)


def test_enforces_abs_cnr_floor():
    image, label = _make_patch()
    transform = LesionContrastD(
        LESION_IDX,
        prob=1.0,
        cnr_lo=1.25,
        cnr_hi=1.25,
        texture_std_fraction=(0.0, 0.0),
    )
    transform.set_random_state(seed=7)
    out = transform({"image": image.clone(), "label": label.clone()})
    cnr = _measure_cnr(out["image"], out["label"])
    assert 1.20 <= abs(cnr) <= 1.30, cnr


def test_random_polarity_exercises_bright_and_dark():
    image, label = _make_patch()
    signs = []
    for seed in range(30):
        transform = LesionContrastD(
            LESION_IDX,
            prob=1.0,
            cnr_lo=1.0,
            cnr_hi=1.0,
            texture_std_fraction=(0.0, 0.0),
        )
        transform.set_random_state(seed=seed)
        out = transform({"image": image.clone(), "label": label.clone()})
        signs.append(np.sign(_measure_cnr(out["image"], out["label"])))
    assert -1.0 in signs
    assert 1.0 in signs


def test_seeded_determinism():
    image, label = _make_patch()
    kwargs = dict(
        lesion_idx=LESION_IDX,
        prob=1.0,
        cnr_lo=0.75,
        cnr_hi=2.0,
        texture_std_fraction=(0.05, 0.18),
    )
    first = LesionContrastD(**kwargs)
    second = LesionContrastD(**kwargs)
    first.set_random_state(seed=13)
    second.set_random_state(seed=13)
    out_a = first({"image": image.clone(), "label": label.clone()})
    out_b = second({"image": image.clone(), "label": label.clone()})
    assert torch.equal(out_a["image"], out_b["image"])
    assert torch.equal(out_a["label"], out_b["label"])


def main() -> int:
    tests = [
        test_noop_when_probability_zero,
        test_enforces_abs_cnr_floor,
        test_random_polarity_exercises_bright_and_dark,
        test_seeded_determinism,
    ]
    for test in tests:
        test()
        print(f"[ok] {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
