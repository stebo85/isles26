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
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import nibabel as nib


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


def load_mask(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Load a binary mask. Returns (data_uint8, affine, voxel_size_mm)."""
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    bin_ = (data > 0).astype(np.uint8)
    zooms = img.header.get_zooms()[:3]
    return bin_, img.affine, tuple(float(z) for z in zooms)


def resample_to_reference(pred_path: Path, ref_img: nib.Nifti1Image) -> np.ndarray:
    """Resample a prediction NIfTI onto a reference grid using nearest neighbour.

    The simplest correct path: use nibabel.processing.resample_from_to. If the
    prediction and reference share the same affine and shape this is a no-op.
    """
    from nibabel.processing import resample_from_to

    pred_img = nib.load(str(pred_path))
    # If grids match exactly, skip resampling for speed.
    if (pred_img.shape == ref_img.shape and
            np.allclose(pred_img.affine, ref_img.affine, atol=1e-3)):
        data = np.asanyarray(pred_img.dataobj)
    else:
        out = resample_from_to(pred_img, (ref_img.shape, ref_img.affine), order=0, mode="constant", cval=0)
        data = np.asanyarray(out.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    return (data > 0).astype(np.uint8)


def iter_subjects(root: Path) -> Iterator[SubjectInputs]:
    yield from discover_soop_bench(root)
