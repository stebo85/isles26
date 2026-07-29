#!/usr/bin/env python3
"""CPU-only geometry tests for the submission container.

These pin the single failure mode most likely to lose this challenge with a good
model: a silent orientation or mirror flip between the caller's native-space
volume and the RAS-canonical grid the nnU-Net models were trained on. Such a
failure still emits a well-formed mask of plausible size, so nothing downstream
notices. This project has already paid for it once (commit f8d88c1: 0.016 Dice
where 0.2576 was correct).

Two levels:
  1. Synthetic volumes written in ALL 48 storable axis-aligned orientations, with
     anisotropic spacing and a deliberately asymmetric marker so a flip cannot
     cancel out.
  2. Real ATLAS R3.0 raw files, one per storage orientation actually present in
     the release (RAS 869, LAS 575, PSR 8, PSL 1), if the data is available.

Run:
    PYTHONPATH=submission/isles26_algorithm .venv-eval/bin/python \
        submission/isles26_algorithm/test_geometry.py
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import SimpleITK as sitk  # noqa: E402

from geometry import (  # noqa: E402
    assert_roundtrip,
    from_training_orientation,
    read_image,
    to_training_orientation,
    write_like_input,
)

PASSED = 0
FAILED = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"[PASS] {name}")
    else:
        FAILED += 1
        print(f"[FAIL] {name}: {detail}")


def make_asymmetric_volume(shape=(23, 19, 17)) -> np.ndarray:
    """A volume with no symmetry in any axis, so any flip or transpose shows up.

    A ramp alone is not enough (the flip of a ramp is another ramp); the blob at
    an off-centre, non-equal index makes all 48 orientations distinct.
    """
    x, y, z = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    vol = (x * 100 + y * 10 + z).astype(np.float32)
    vol[3:6, 11:13, 2:9] += 5000.0
    return vol


def direction_matrices():
    """All 48 axis-aligned direction cosine matrices (24 rotations + reflections)."""
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            m = np.zeros((3, 3))
            for axis, src in enumerate(perm):
                m[src, axis] = signs[axis]
            yield perm, signs, m


def test_synthetic_all_orientations(tmpdir: Path) -> None:
    import nibabel as nib

    vol = make_asymmetric_volume()
    spacing = (0.9375, 3.0, 0.9375)  # a real anisotropic spacing from ATLAS R3.0
    n_ok = 0
    n_total = 0
    for i, (perm, signs, direction) in enumerate(direction_matrices()):
        n_total += 1
        img = sitk.GetImageFromArray(vol.transpose(2, 1, 0))
        img.SetSpacing(spacing)
        img.SetOrigin((11.5, -7.25, 3.0))
        img.SetDirection(tuple(direction.flatten()))
        path = tmpdir / f"orient_{i:02d}.mha"
        sitk.WriteImage(img, str(path), useCompression=True)

        data, geom = read_image(str(path))
        canonical, _ = to_training_orientation(data, geom)
        restored = from_training_orientation(canonical, geom)

        if not np.array_equal(restored, data):
            check(
                f"round-trip perm={perm} signs={signs}",
                False,
                "restored array differs from the original",
            )
            continue

        # The canonical volume must actually be RAS, otherwise we are feeding the
        # network something it was not trained on.
        canon_affine = geom.affine_ras @ nib.orientations.inv_ornt_aff(
            geom.to_canonical, data.shape
        )
        codes = nib.orientations.aff2axcodes(canon_affine)
        if codes != ("R", "A", "S"):
            check(f"canonical is RAS for perm={perm} signs={signs}", False, f"got {codes}")
            continue

        # Writing a prediction back must land on the caller's exact grid.
        out_path = tmpdir / f"out_{i:02d}.mha"
        pred = (restored > np.percentile(restored, 90)).astype(np.uint8)
        write_like_input(pred, geom, str(out_path), np.uint8)
        back = sitk.ReadImage(str(out_path))
        same_grid = (
            back.GetSize() == geom.sitk_image.GetSize()
            and np.allclose(back.GetSpacing(), geom.sitk_image.GetSpacing())
            and np.allclose(back.GetOrigin(), geom.sitk_image.GetOrigin())
            and np.allclose(back.GetDirection(), geom.sitk_image.GetDirection())
        )
        if not same_grid:
            check(f"output grid matches input for perm={perm} signs={signs}", False, "grid drift")
            continue
        n_ok += 1

    check(
        f"all {n_total} axis-aligned orientations round-trip exactly and land on the input grid",
        n_ok == n_total,
        f"{n_ok}/{n_total} passed",
    )


def test_nifti_roundtrip(tmpdir: Path) -> None:
    """The training data is NIfTI, not .mha; make sure both read paths agree."""
    import nibabel as nib

    vol = make_asymmetric_volume()
    affine = np.array(
        [[0.0, 0.0, 1.0, -90.0], [-1.0, 0.0, 0.0, 126.0], [0.0, -1.0, 0.0, 72.0], [0, 0, 0, 1]]
    )  # a PSR-like storage orientation
    path = tmpdir / "psr.nii.gz"
    nib.save(nib.Nifti1Image(vol, affine), str(path))

    data, geom = read_image(str(path))
    check(
        "NIfTI read reproduces nibabel's array and axcodes",
        np.allclose(data, np.asanyarray(nib.load(str(path)).dataobj))
        and geom.axcodes == nib.orientations.aff2axcodes(nib.load(str(path)).affine),
        f"axcodes {geom.axcodes}",
    )

    canonical, _ = to_training_orientation(data, geom)
    reference = np.asanyarray(nib.as_closest_canonical(nib.load(str(path))).dataobj)
    check(
        "our canonicalisation matches nib.as_closest_canonical exactly",
        np.array_equal(canonical, reference),
        "the training conversion used as_closest_canonical; any difference means "
        "we feed the network a different volume than it was trained on",
    )
    check(
        "round-trip is exact on the NIfTI path",
        np.array_equal(from_training_orientation(canonical, geom), data),
    )


def test_real_cases(per_session_csv: Path, limit_per_orient: int = 1) -> None:
    if not per_session_csv.is_file():
        print(f"[skip] real-case tests: {per_session_csv} not found")
        return
    by_orient: dict[str, list[str]] = {}
    with per_session_csv.open() as fh:
        for row in csv.DictReader(fh):
            orient = row.get("orient") or "?"
            path = row.get("t1_path") or ""
            if path and Path(path).is_file():
                by_orient.setdefault(orient, []).append(path)

    if not by_orient:
        print("[skip] real-case tests: no readable T1w paths in the characterization CSV")
        return

    for orient, paths in sorted(by_orient.items()):
        for path in paths[:limit_per_orient]:
            data, geom = read_image(path)
            try:
                assert_roundtrip(data, geom)
                ok, detail = True, ""
            except AssertionError as exc:
                ok, detail = False, str(exc)
            check(
                f"real ATLAS case round-trips ({orient}, n={len(paths)} available)",
                ok,
                f"{detail} [{path}]",
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--per-session-csv",
        type=Path,
        default=Path("baselines/reports/training_data_characterization_r30/per_session.csv"),
    )
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="isles26_geom_") as tmp:
        tmpdir = Path(tmp)
        test_synthetic_all_orientations(tmpdir)
        test_nifti_roundtrip(tmpdir)
    test_real_cases(args.per_session_csv)

    print(f"\n{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
