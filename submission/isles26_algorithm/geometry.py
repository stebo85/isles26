"""Exact, invertible geometry handling between the caller's grid and the grid
the nnU-Net models were actually trained on.

WHY THIS FILE IS THE MOST DANGEROUS CODE IN THE SUBMISSION
----------------------------------------------------------
Every model in this repo was trained on volumes passed through
``nib.as_closest_canonical`` (see
``nnunet_helpers/convert_atlas_r21_to_nnunet.py``), i.e. RAS axis order. The
ISLES'26 test data is explicitly native-space, and the training release alone
contains four different storage orientations: RAS 869, LAS 575, PSR 8, PSL 1
(``baselines/reports/training_data_characterization_r30/per_session.csv``), with
a negative determinant on 583 of 1453 sessions.

nnU-Net does NOT reorient at inference: the plans record
``transpose_forward = [0, 1, 2]``. So if the caller hands us an LAS volume and we
predict on it directly, the network sees a mirrored brain. LAS is a pure axis-0
flip which mirror-augmentation partly absorbs, but PSR/PSL are axis PERMUTATIONS
the network has never seen, and the output would be garbage.

Worse, this fails SILENTLY: the container still emits a well-formed mask with a
plausible lesion volume. This project has already been bitten by exactly this
once -- commit f8d88c1, where a mirror-flipped comparison scored 0.016 Dice
against a correct 0.2576.

The contract here is therefore:
    to_training_orientation()  -> array in the orientation the model expects
    from_training_orientation() -> array back on the caller's exact grid
and ``from_training_orientation(to_training_orientation(x)) == x`` is asserted by
``test_orientation_roundtrip.py`` on real ATLAS volumes in every storage
orientation, before every submission.

We deliberately use the explicit ``nibabel.orientations`` transform rather than
``as_closest_canonical`` alone, because we need the inverse and we want it to be
derived, not guessed.
"""

from __future__ import annotations

from dataclasses import dataclass

import nibabel as nib
import numpy as np
import SimpleITK as sitk

# nibabel works in RAS+; SimpleITK works in LPS. Converting between the two
# affines is a sign flip on the first two world axes.
_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])

_RAS_ORNT = nib.orientations.axcodes2ornt(("R", "A", "S"))


@dataclass(frozen=True)
class InputGeometry:
    """Everything needed to put a prediction back exactly where it came from."""

    sitk_image: sitk.Image
    affine_ras: np.ndarray  # 4x4, nibabel convention, on the ORIGINAL grid
    original_shape: tuple[int, int, int]
    axcodes: tuple[str, str, str]
    to_canonical: np.ndarray  # nibabel ornt transform, original -> RAS
    from_canonical: np.ndarray  # its exact inverse


def sitk_to_ras_affine(image: sitk.Image) -> np.ndarray:
    """Build the nibabel (RAS+) affine of a SimpleITK image."""
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)

    affine_lps = np.eye(4, dtype=np.float64)
    affine_lps[:3, :3] = direction @ np.diag(spacing)
    affine_lps[:3, 3] = origin
    return _LPS_TO_RAS @ affine_lps


def read_image(path: str) -> tuple[np.ndarray, InputGeometry]:
    """Read any SimpleITK-supported volume (.mha, .nii.gz, ...).

    Returns the voxel array in nibabel index order (x, y, z) on the ORIGINAL
    grid, plus the geometry needed to invert everything later.
    """
    image = sitk.ReadImage(path)

    if image.GetDimension() == 4:
        # TRACE-style 4-D singleton volumes exist in this project's data; a
        # 4-D image reaching the network is a hard error, not something to
        # squeeze silently, because the singleton axis position is ambiguous.
        size = image.GetSize()
        if size[3] != 1:
            raise ValueError(f"{path}: expected a 3-D volume, got 4-D with size {size}")
        image = image[:, :, :, 0]

    if image.GetDimension() != 3:
        raise ValueError(f"{path}: expected a 3-D volume, got {image.GetDimension()}-D")

    # SimpleITK arrays are (z, y, x); nibabel's convention is (x, y, z).
    data = sitk.GetArrayFromImage(image).transpose(2, 1, 0)
    affine_ras = sitk_to_ras_affine(image)

    orig_ornt = nib.orientations.io_orientation(affine_ras)
    to_canonical = nib.orientations.ornt_transform(orig_ornt, _RAS_ORNT)
    from_canonical = nib.orientations.ornt_transform(_RAS_ORNT, orig_ornt)

    geom = InputGeometry(
        sitk_image=image,
        affine_ras=affine_ras,
        original_shape=tuple(int(s) for s in data.shape),
        axcodes=tuple(nib.orientations.aff2axcodes(affine_ras)),
        to_canonical=to_canonical,
        from_canonical=from_canonical,
    )
    return np.ascontiguousarray(data), geom


def to_training_orientation(data: np.ndarray, geom: InputGeometry) -> tuple[np.ndarray, np.ndarray]:
    """Reorient the caller's array into the RAS order the models were trained on.

    Returns (array, affine) both on the canonical grid, ready to be written as
    the NIfTI that nnU-Net will read.
    """
    canonical = nib.orientations.apply_orientation(data, geom.to_canonical)
    affine = geom.affine_ras @ nib.orientations.inv_ornt_aff(geom.to_canonical, data.shape)
    return np.ascontiguousarray(canonical), affine


def from_training_orientation(canonical: np.ndarray, geom: InputGeometry) -> np.ndarray:
    """Map a prediction on the canonical grid back onto the caller's exact grid."""
    restored = nib.orientations.apply_orientation(canonical, geom.from_canonical)
    if tuple(restored.shape) != geom.original_shape:
        raise ValueError(
            f"orientation round-trip changed the shape: got {restored.shape}, "
            f"expected {geom.original_shape}. Refusing to emit a misaligned mask."
        )
    return np.ascontiguousarray(restored)


def write_like_input(
    data: np.ndarray, geom: InputGeometry, path: str, dtype: type[np.generic]
) -> None:
    """Write a volume onto the caller's exact grid.

    ``CopyInformation`` is what actually guarantees identical size, origin,
    spacing and direction -- recomputing them from our own affine would
    reintroduce floating-point drift that the evaluator sees as a grid mismatch.
    """
    if tuple(data.shape) != geom.original_shape:
        raise ValueError(
            f"refusing to write {path}: array shape {data.shape} != input shape "
            f"{geom.original_shape}"
        )
    arr = np.ascontiguousarray(data.astype(dtype).transpose(2, 1, 0))
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(geom.sitk_image)
    sitk.WriteImage(out, path, useCompression=True)


def assert_roundtrip(data: np.ndarray, geom: InputGeometry) -> None:
    """Fail loudly if the orientation transform is not exactly invertible.

    Cheap enough to run on every case in production. A silent mirror flip costs
    ~94% of Dice on 100% of cases; an assertion costs microseconds.
    """
    canonical, _ = to_training_orientation(data, geom)
    restored = from_training_orientation(canonical, geom)
    if not np.array_equal(restored, data):
        raise AssertionError(
            f"orientation round-trip is not exact for axcodes {geom.axcodes}; "
            "refusing to predict on a volume we cannot map back"
        )
