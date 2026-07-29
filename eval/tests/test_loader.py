"""Unit tests for eval/loader.py geometry handling and soft-map loading.

These pin two behaviours that used to fail silently (audit findings 20 and 3):

  - a prediction on the wrong grid was nearest-neighbour resampled and nothing
    recorded it, so a pipeline bug scored as a mediocre model instead of as a
    crash. The official scorer raises instead; strict mode reproduces that, and
    the permissive default now at least counts what it resampled.
  - every load was binarised at `> 0`, so a probability map could not be scored
    at all -- and PR-AUC, one of the five ranked metrics, is defined only on the
    soft map.

Run with:
    cd /scratch/users/sciget/isles26challenge
    PYTHONPATH=. .venv-eval/bin/python eval/tests/test_loader.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.loader import (
    GeometryMismatchError,
    geometry_stats,
    load_mask,
    load_soft,
    load_soft_to_reference,
    reset_geometry_stats,
    resample_to_reference,
)


SHAPE = (10, 10, 10)


def _fixtures(tmp: Path) -> tuple[nib.Nifti1Image, Path, Path]:
    """Reference image, an on-grid soft prediction, and an off-grid one."""
    ref = nib.Nifti1Image(np.zeros(SHAPE, np.uint8), np.eye(4))
    soft = np.zeros(SHAPE, np.float32)
    soft[2:5, 2:5, 2:5] = 0.7
    on_grid = tmp / "on_grid.nii.gz"
    nib.save(nib.Nifti1Image(soft, np.eye(4)), on_grid)
    off_grid = tmp / "off_grid.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((5, 5, 5), np.uint8), np.diag([2.0, 2.0, 2.0, 1.0])), off_grid)
    return ref, on_grid, off_grid


def test_load_mask_binarises_by_default():
    with tempfile.TemporaryDirectory() as d:
        _, on_grid, _ = _fixtures(Path(d))
        data, _, zooms = load_mask(on_grid)
        assert data.dtype == np.uint8
        assert int(data.sum()) == 27  # 3^3 voxels above 0
        assert zooms == (1.0, 1.0, 1.0)


def test_load_soft_keeps_probabilities():
    with tempfile.TemporaryDirectory() as d:
        _, on_grid, _ = _fixtures(Path(d))
        data, _, _ = load_soft(on_grid)
        assert data.dtype == np.float32
        assert abs(float(data.max()) - 0.7) < 1e-6
        # binarise=False on load_mask is the same entry point
        alt, _, _ = load_mask(on_grid, binarise=False)
        assert np.array_equal(alt, data)


def test_resample_default_still_resamples():
    """The permissive default must not change: existing soop_bench/ATLAS
    callers compare a model-grid prediction against acquisition-grid GT."""
    with tempfile.TemporaryDirectory() as d:
        ref, _, off_grid = _fixtures(Path(d))
        reset_geometry_stats()
        out = resample_to_reference(off_grid, ref)
        assert out.shape == SHAPE
        assert out.dtype == np.uint8
        stats = geometry_stats()
        assert stats["n_resampled"] == 1
        assert stats["n_checked"] == 1
        assert stats["mismatch_examples"][0]["pred_shape"] == [5, 5, 5]


def test_strict_geometry_raises_on_mismatch():
    with tempfile.TemporaryDirectory() as d:
        ref, on_grid, off_grid = _fixtures(Path(d))
        reset_geometry_stats()
        # on-grid input passes strict mode untouched
        out = resample_to_reference(on_grid, ref, strict_geometry=True)
        assert int(out.sum()) == 27
        raised = False
        try:
            resample_to_reference(off_grid, ref, strict_geometry=True)
        except GeometryMismatchError:
            raised = True
        assert raised, "strict mode did not reject an off-grid prediction"
        stats = geometry_stats()
        assert stats["n_on_grid"] == 1
        assert stats["n_refused_strict"] == 1
        assert stats["n_resampled"] == 0


def test_strict_and_allow_resample_is_rejected():
    with tempfile.TemporaryDirectory() as d:
        ref, on_grid, _ = _fixtures(Path(d))
        raised = False
        try:
            resample_to_reference(on_grid, ref, strict_geometry=True, allow_resample=True)
        except ValueError:
            raised = True
        assert raised, "contradictory flags were accepted"


def test_soft_to_reference_is_strict_by_default():
    with tempfile.TemporaryDirectory() as d:
        ref, on_grid, off_grid = _fixtures(Path(d))
        reset_geometry_stats()
        out = load_soft_to_reference(on_grid, ref)
        assert out.dtype == np.float32
        assert abs(float(out.max()) - 0.7) < 1e-6
        raised = False
        try:
            load_soft_to_reference(off_grid, ref)
        except GeometryMismatchError:
            raised = True
        assert raised, "soft loading silently resampled a grid mismatch"


def test_geometry_stats_reset():
    with tempfile.TemporaryDirectory() as d:
        ref, on_grid, _ = _fixtures(Path(d))
        resample_to_reference(on_grid, ref)
        reset_geometry_stats()
        assert geometry_stats()["n_checked"] == 0


def _run_all() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERR  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(_run_all())
