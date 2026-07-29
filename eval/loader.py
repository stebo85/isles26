"""Loaders for the soop_bench BIDS-style dataset.

Layout assumed (real on-disk):
    data/soop_bench/
        sub-N/anat/sub-N_T1w.nii.gz
        sub-N/anat/sub-N_FLAIR.nii.gz
        sub-N/dwi/sub-N_rec-TRACE_dwi.nii.gz
        sub-N/dwi/sub-N_rec-ADC_dwi.nii.gz
        derivatives/lesion_masks/sub-N/dwi/sub-N_space-TRACE_desc-lesion_mask.nii.gz
        derivatives/lesion_masks/sub-N/dwi/sub-N_space-TRACE_desc-lesionAcute_mask.nii.gz
        derivatives/lesion_masks/sub-N/dwi/sub-N_space-TRACE_desc-lesionChronic_mask.nii.gz

The combined `desc-lesion_mask` is used as the primary GT. Per-phase masks are
also returned so we can stratify by chronicity. Subjects that have only an
acute (or only a chronic) sub-mask are tagged accordingly; ones with both are
"mixed". This is a coarse approximation: the soop_bench dataset does not
include explicit per-subject phase metadata files, so we derive chronicity
from the presence of the per-phase masks.

Geometry policy (audit finding 20)
    `resample_to_reference` used to nearest-neighbour resample ANY grid
    mismatch and say nothing about it, so a prediction written on the wrong
    grid scored as a merely mediocre prediction instead of as the pipeline bug
    it is. The official scorer does not forgive this: `compute_pr_auc` in the
    organizers' `utils/eval_utils.py` raises ValueError on a shape mismatch.
    So:
      - `strict_geometry=True` asserts shape and affine equality and raises
        `GeometryMismatchError`. Use it for anything that validates a
        submission, where a resample would hide a real defect.
      - Resampling still happens by default (`strict_geometry=False`), because
        the existing soop_bench/ATLAS callers legitimately compare predictions
        produced on a model grid against GT on the acquisition grid.
      - EITHER WAY the number of resampled cases is accumulated in a
        module-level counter, exposed by `geometry_stats()` and folded into
        report.json by `eval/report.py`, so a silently-resampled run is
        visible after the fact.

Soft maps (audit finding 3)
    Loading binarised everything at `(data > 0)`, which made it impossible to
    score a probability map at all — and PR-AUC, one of the five official
    ranked metrics, is defined only on the soft map. `load_soft` /
    `binarise=False` return the un-binarised float volume. Defaults are
    unchanged, so existing binary call sites behave exactly as before.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import nibabel as nib


# Affines are compared with a tolerance rather than exactly: NIfTI headers
# round-trip through float32 sform/qform storage, so a genuinely identical grid
# can differ in the last bits. 1e-3 mm is far below any voxel size we handle.
_AFFINE_ATOL = 1e-3


class GeometryMismatchError(ValueError):
    """Raised in strict mode when a prediction is not on the reference grid."""


@dataclass
class SubjectInputs:
    subject_id: str
    t1w: Path | None
    flair: Path | None
    trace: Path | None
    adc: Path | None
    lesion_mask: Path
    acute_mask: Path | None
    chronic_mask: Path | None
    chronicity: str  # "acute", "chronic", "mixed", or "unknown"
    center: str | None  # derived from JSON sidecar Manufacturer + Model
    metadata: dict = field(default_factory=dict)


def _exists(p: Path | None) -> Path | None:
    return p if (p is not None and p.exists()) else None


def _read_sidecar(p: Path) -> dict:
    j = p.with_suffix("").with_suffix(".json")
    if not j.exists():
        return {}
    try:
        return json.loads(j.read_text())
    except Exception:
        return {}


def discover_soop_bench(root: Path) -> list[SubjectInputs]:
    """Return all subjects with a combined lesion mask available."""
    root = Path(root)
    masks_root = root / "derivatives" / "lesion_masks"
    subjects: list[SubjectInputs] = []
    for sub_dir in sorted(masks_root.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
            continue
        sid = sub_dir.name
        mask_dir = sub_dir / "dwi"
        combined = mask_dir / f"{sid}_space-TRACE_desc-lesion_mask.nii.gz"
        if not combined.exists():
            continue
        acute = _exists(mask_dir / f"{sid}_space-TRACE_desc-lesionAcute_mask.nii.gz")
        chronic = _exists(mask_dir / f"{sid}_space-TRACE_desc-lesionChronic_mask.nii.gz")
        if acute and chronic:
            chronicity = "mixed"
        elif acute:
            chronicity = "acute"
        elif chronic:
            chronicity = "chronic"
        else:
            chronicity = "unknown"

        anat = root / sid / "anat"
        dwi = root / sid / "dwi"
        t1w = _exists(anat / f"{sid}_T1w.nii.gz")
        flair = _exists(anat / f"{sid}_FLAIR.nii.gz")
        trace = _exists(dwi / f"{sid}_rec-TRACE_dwi.nii.gz")
        adc = _exists(dwi / f"{sid}_rec-ADC_dwi.nii.gz")

        # Center identifier: prefer T1w sidecar, fall back to TRACE.
        sidecar = {}
        for source in (t1w, trace, flair, adc):
            if source is not None:
                sidecar = _read_sidecar(source)
                if sidecar:
                    break
        manufacturer = sidecar.get("Manufacturer", "Unknown")
        model = sidecar.get("ManufacturersModelName", "Unknown")
        field_strength = sidecar.get("MagneticFieldStrength", None)
        center = f"{manufacturer}|{model}"

        subjects.append(SubjectInputs(
            subject_id=sid,
            t1w=t1w, flair=flair, trace=trace, adc=adc,
            lesion_mask=combined,
            acute_mask=acute,
            chronic_mask=chronic,
            chronicity=chronicity,
            center=center,
            metadata={
                "Manufacturer": manufacturer,
                "ManufacturersModelName": model,
                "MagneticFieldStrength": field_strength,
            },
        ))
    return subjects


# --- geometry bookkeeping ---------------------------------------------------

@dataclass
class GeometryStats:
    """How many predictions were on the reference grid, and how many were not.

    Accumulated process-wide because `resample_to_reference` returns a bare
    array and is called from several runners we do not want to re-plumb. Call
    `reset_geometry_stats()` at the start of a run.
    """
    n_checked: int = 0
    n_on_grid: int = 0
    n_resampled: int = 0
    n_refused: int = 0  # strict-mode rejections
    examples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_checked": self.n_checked,
            "n_on_grid": self.n_on_grid,
            "n_resampled": self.n_resampled,
            "n_refused_strict": self.n_refused,
            "mismatch_examples": self.examples,
        }


# Only the first few mismatches are kept verbatim; the counters carry the rest.
_MAX_MISMATCH_EXAMPLES = 20

_GEOMETRY_STATS = GeometryStats()


def geometry_stats() -> dict:
    """Snapshot of the grid-mismatch bookkeeping, for embedding in report.json."""
    return _GEOMETRY_STATS.to_dict()


def reset_geometry_stats() -> None:
    global _GEOMETRY_STATS
    _GEOMETRY_STATS = GeometryStats()


def _same_grid(img: nib.Nifti1Image, ref_img: nib.Nifti1Image) -> bool:
    return (tuple(img.shape[:3]) == tuple(ref_img.shape[:3])
            and np.allclose(img.affine, ref_img.affine, atol=_AFFINE_ATOL))


# --- loading ----------------------------------------------------------------

def _squeeze_trailing(data: np.ndarray) -> np.ndarray:
    """Drop a length-1 4th dimension, which some writers emit for 3-D volumes."""
    if data.ndim == 4 and data.shape[-1] == 1:
        return data[..., 0]
    return data


def load_mask(path: Path, binarise: bool = True) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Load a volume. Returns (data, affine, voxel_size_mm).

    With `binarise=True` (the default, and what every existing call site wants)
    the data comes back as uint8 `data > 0`. With `binarise=False` it comes back
    as float32 with the stored values intact, which is what PR-AUC needs — see
    `load_soft`.
    """
    img = nib.load(str(path))
    data = _squeeze_trailing(np.asanyarray(img.dataobj))
    out = (data > 0).astype(np.uint8) if binarise else data.astype(np.float32)
    zooms = img.header.get_zooms()[:3]
    return out, img.affine, tuple(float(z) for z in zooms)


def load_soft(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Load an un-binarised probability/score volume as float32.

    Thin alias for `load_mask(path, binarise=False)`, named so that call sites
    scoring PR-AUC read unambiguously.
    """
    return load_mask(path, binarise=False)


def resample_to_reference(
    pred_path: Path,
    ref_img: nib.Nifti1Image,
    strict_geometry: bool = False,
    allow_resample: bool | None = None,
    binarise: bool = True,
) -> np.ndarray:
    """Bring a prediction NIfTI onto the reference grid.

    `strict_geometry=True` requires the prediction to already be on the
    reference grid (same shape, same affine to `_AFFINE_ATOL`) and raises
    `GeometryMismatchError` otherwise — the behaviour the official scorer has,
    and the one to use when validating a submission.

    `allow_resample` defaults to the opposite of `strict_geometry`, i.e.
    resampling stays on for the existing callers and off in strict mode.
    Passing both flags True is contradictory and rejected. When resampling is
    allowed, binary data is resampled nearest-neighbour and soft data
    trilinearly; nearest neighbour on a probability map would quantise it to
    the source grid's values and distort PR-AUC.

    Every call is counted in the module-level `geometry_stats()` regardless of
    mode, so a run that quietly resampled everything is detectable afterwards.
    """
    from nibabel.processing import resample_from_to

    if allow_resample is None:
        allow_resample = not strict_geometry
    elif strict_geometry and allow_resample:
        raise ValueError(
            "strict_geometry=True and allow_resample=True are contradictory: "
            "strict mode exists precisely to refuse the resample"
        )

    pred_img = nib.load(str(pred_path))
    on_grid = _same_grid(pred_img, ref_img)
    _GEOMETRY_STATS.n_checked += 1

    if on_grid:
        _GEOMETRY_STATS.n_on_grid += 1
        data = _squeeze_trailing(np.asanyarray(pred_img.dataobj))
    else:
        detail = {
            "path": str(pred_path),
            "pred_shape": list(pred_img.shape[:3]),
            "ref_shape": list(ref_img.shape[:3]),
            "max_affine_abs_diff": float(np.max(np.abs(pred_img.affine - ref_img.affine))),
        }
        if len(_GEOMETRY_STATS.examples) < _MAX_MISMATCH_EXAMPLES:
            _GEOMETRY_STATS.examples.append(detail)
        if not allow_resample:
            _GEOMETRY_STATS.n_refused += 1
            raise GeometryMismatchError(
                f"prediction {pred_path} is not on the reference grid "
                f"(shape {detail['pred_shape']} vs {detail['ref_shape']}, "
                f"max |Δaffine| = {detail['max_affine_abs_diff']:.4g}); "
                "resampling is disabled in strict mode because the official "
                "scorer raises on a grid mismatch rather than resampling"
            )
        _GEOMETRY_STATS.n_resampled += 1
        out = resample_from_to(
            pred_img, (ref_img.shape, ref_img.affine),
            order=0 if binarise else 1, mode="constant", cval=0,
        )
        data = _squeeze_trailing(np.asanyarray(out.dataobj))

    return (data > 0).astype(np.uint8) if binarise else data.astype(np.float32)


def load_soft_to_reference(
    pred_path: Path,
    ref_img: nib.Nifti1Image,
    strict_geometry: bool = True,
) -> np.ndarray:
    """Soft-map counterpart of `resample_to_reference`.

    Strict by default: a probability map is only ever consumed for PR-AUC, the
    official implementation of which raises on a shape mismatch, so silently
    resampling here would make our PR-AUC disagree with the leaderboard's.
    """
    return resample_to_reference(
        pred_path, ref_img, strict_geometry=strict_geometry, binarise=False
    )


def iter_subjects(root: Path) -> Iterator[SubjectInputs]:
    yield from discover_soop_bench(root)
