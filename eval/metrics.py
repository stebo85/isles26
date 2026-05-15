"""Sensitive segmentation metrics for ISLES'26-style stroke lesion evaluation.

Design notes:

- Inputs are always 3-D binary uint8/bool arrays of identical shape, in voxel
  space, with an associated voxel size (a 3-tuple in mm). Callers must align
  prediction and ground-truth to the same grid before passing them in. The
  evaluator does this with nearest-neighbour resampling using the GT affine.
- Edge cases (empty GT, empty prediction) are handled explicitly per metric and
  documented in each function's docstring. The convention is:
    - Dice/IoU/F1: empty-on-both -> 1.0 (perfect match of "no lesion"),
      empty-on-one -> 0.0.
    - HD95/ASSD: NaN if either mask is empty. Reporting code aggregates with
      nanmean so empty-on-both cases do not penalize boundary metrics.
- All "lesion-wise" metrics are computed on connected components from a 26-
  neighbourhood (3x3x3 connectivity structure). A GT component is "detected"
  when ANY predicted component overlaps it by at least `min_overlap` voxels
  (default 1, i.e. the ATLAS definition). A predicted component is a "true
  positive" when it overlaps any GT component by at least `min_overlap`.
- Lesion-volume bins are in mL (cm^3). Defaults follow ATLAS-style small-lesion
  literature (Shang et al. 2024 uses <100 voxels; we report in mL for
  generality and also expose the raw voxel-count bins).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy import ndimage as ndi


# --- low-level helpers ------------------------------------------------------

_CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


def _as_bool(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr).astype(bool)


def _voxel_volume_ml(voxel_size_mm: tuple[float, float, float]) -> float:
    return float(np.prod(voxel_size_mm)) / 1000.0  # mm^3 -> mL


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label connected components with 26-neighbour connectivity."""
    return ndi.label(_as_bool(mask), structure=_CONNECTIVITY_26)


def component_voxel_counts(label_volume: np.ndarray, n_components: int) -> np.ndarray:
    """Return a (n_components+1,) array of voxel counts; index 0 is background."""
    if n_components == 0:
        return np.zeros(1, dtype=np.int64)
    counts = np.bincount(label_volume.ravel(), minlength=n_components + 1)
    return counts.astype(np.int64)


# --- voxel-wise overlap metrics --------------------------------------------

@dataclass
class OverlapMetrics:
    dice: float
    iou: float
    sensitivity: float  # = recall = TPR
    specificity: float  # = TNR
    precision: float    # = PPV
    npv: float
    tp: int
    fp: int
    fn: int
    tn: int


def overlap_metrics(pred: np.ndarray, gt: np.ndarray) -> OverlapMetrics:
    """Voxel-wise confusion-matrix-based metrics. Returns 1.0 for Dice/IoU when
    both masks are empty (perfect agreement on background), and 0.0 when only
    one is empty."""
    p = _as_bool(pred)
    g = _as_bool(gt)
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs gt {g.shape}")
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    tn = int(np.logical_and(~p, ~g).sum())

    def _safe(num: int, denom: int, both_empty_value: float = 1.0) -> float:
        if denom == 0:
            return both_empty_value
        return num / denom

    p_sum, g_sum = int(p.sum()), int(g.sum())
    if p_sum == 0 and g_sum == 0:
        dice = 1.0
        iou = 1.0
    elif p_sum == 0 or g_sum == 0:
        dice = 0.0
        iou = 0.0
    else:
        dice = 2 * tp / (2 * tp + fp + fn)
        iou = tp / (tp + fp + fn)

    sensitivity = _safe(tp, tp + fn, both_empty_value=math.nan)  # undefined if no GT positives
    specificity = _safe(tn, tn + fp, both_empty_value=math.nan)
    precision = _safe(tp, tp + fp, both_empty_value=math.nan)  # undefined if no predicted positives
    npv = _safe(tn, tn + fn, both_empty_value=math.nan)

    return OverlapMetrics(
        dice=float(dice),
        iou=float(iou),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        precision=float(precision),
        npv=float(npv),
        tp=tp, fp=fp, fn=fn, tn=tn,
    )


# --- volume metrics ---------------------------------------------------------

@dataclass
class VolumeMetrics:
    pred_vol_ml: float
    gt_vol_ml: float
    abs_volume_diff_ml: float
    signed_volume_diff_ml: float  # pred - gt
    relative_abs_volume_diff: float  # AVD / max(gt, eps); NaN if gt empty


def volume_metrics(pred: np.ndarray, gt: np.ndarray, voxel_size_mm: tuple[float, float, float]) -> VolumeMetrics:
    vox_ml = _voxel_volume_ml(voxel_size_mm)
    p_vox = int(_as_bool(pred).sum())
    g_vox = int(_as_bool(gt).sum())
    p_ml = p_vox * vox_ml
    g_ml = g_vox * vox_ml
    avd = abs(p_ml - g_ml)
    rvd = avd / g_ml if g_ml > 0 else math.nan
    return VolumeMetrics(
        pred_vol_ml=p_ml,
        gt_vol_ml=g_ml,
        abs_volume_diff_ml=avd,
        signed_volume_diff_ml=p_ml - g_ml,
        relative_abs_volume_diff=float(rvd) if not math.isnan(rvd) else math.nan,
    )


# --- boundary metrics -------------------------------------------------------

def _surface_distances(mask_a: np.ndarray, mask_b: np.ndarray, voxel_size_mm: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Compute distances from boundary voxels of A to boundary of B and vice versa, in mm.

    A "boundary voxel" is a foreground voxel adjacent to a background voxel
    using 6-connectivity (face-touching). Distances are computed via the EDT
    on the *complement* of the other mask's boundary, sampled at boundary
    voxels of this mask.
    """
    a = _as_bool(mask_a)
    b = _as_bool(mask_b)
    if a.shape != b.shape:
        raise ValueError("shape mismatch in _surface_distances")
    if not a.any() or not b.any():
        return np.array([]), np.array([])

    # 6-connectivity surface: foreground voxels with at least one face-neighbor
    # of background.
    struct = ndi.generate_binary_structure(3, 1)
    a_eroded = ndi.binary_erosion(a, structure=struct, border_value=0)
    b_eroded = ndi.binary_erosion(b, structure=struct, border_value=0)
    a_surf = a & ~a_eroded
    b_surf = b & ~b_eroded

    # If a mask is exactly one voxel thick everywhere, erosion empties it; fall
    # back to the mask itself.
    if not a_surf.any():
        a_surf = a
    if not b_surf.any():
        b_surf = b

    # EDT to nearest non-surface voxel of the OTHER mask.
    sampling = tuple(float(v) for v in voxel_size_mm)
    dist_to_b = ndi.distance_transform_edt(~b_surf, sampling=sampling)
    dist_to_a = ndi.distance_transform_edt(~a_surf, sampling=sampling)

    return dist_to_b[a_surf], dist_to_a[b_surf]


@dataclass
class BoundaryMetrics:
    hd95_mm: float
    assd_mm: float           # average symmetric surface distance
    masd_mm: float           # mean average surface distance (average of two directional ASDs)
    surface_dice_3mm: float  # surface dice with 3 mm tolerance (ISLES standard)


def boundary_metrics(pred: np.ndarray, gt: np.ndarray, voxel_size_mm: tuple[float, float, float]) -> BoundaryMetrics:
    """Boundary distance metrics. NaN if either mask is empty."""
    p = _as_bool(pred)
    g = _as_bool(gt)
    if not p.any() or not g.any():
        return BoundaryMetrics(math.nan, math.nan, math.nan, math.nan)
    d_p_to_g, d_g_to_p = _surface_distances(p, g, voxel_size_mm)
    if d_p_to_g.size == 0 or d_g_to_p.size == 0:
        return BoundaryMetrics(math.nan, math.nan, math.nan, math.nan)
    all_d = np.concatenate([d_p_to_g, d_g_to_p])
    hd95 = float(np.percentile(all_d, 95))
    assd = float(all_d.mean())
    masd = float(0.5 * (d_p_to_g.mean() + d_g_to_p.mean()))
    tol_mm = 3.0
    sd = (np.sum(d_p_to_g <= tol_mm) + np.sum(d_g_to_p <= tol_mm)) / (d_p_to_g.size + d_g_to_p.size)
    return BoundaryMetrics(hd95_mm=hd95, assd_mm=assd, masd_mm=masd, surface_dice_3mm=float(sd))


# --- lesion-wise (detection) metrics ---------------------------------------

@dataclass
class LesionWiseMetrics:
    gt_lesion_count: int
    pred_lesion_count: int
    tp_lesions: int
    fp_lesions: int
    fn_lesions: int
    lesion_precision: float
    lesion_recall: float
    lesion_f1: float
    abs_lesion_count_diff: int  # |pred_count - gt_count|
    signed_lesion_count_diff: int  # pred_count - gt_count
    per_gt_overlap: list[int]  # voxel overlap each GT component received (len == gt_lesion_count)
    per_gt_size_voxels: list[int]
    per_pred_size_voxels: list[int]
    detected_gt_indices: list[int]  # 1-based labels of GT comps that were detected


def lesion_wise_metrics(pred: np.ndarray, gt: np.ndarray, min_overlap_voxels: int = 1) -> LesionWiseMetrics:
    """Connected-component-based lesion detection metrics.

    A GT component is detected if any predicted component overlaps it by at
    least `min_overlap_voxels`. A predicted component is a true positive if it
    overlaps any GT component by at least `min_overlap_voxels`. (This is the
    ATLAS convention.) An empty mask contributes 0 lesions in that direction.

    Conventions when both counts are zero:
        precision = recall = f1 = 1.0
    When only one side is empty:
        the metrics that divide by zero are returned as 0.0 (consistent with
        Dice/IoU empty-mismatch convention).
    """
    p_lab, n_pred = connected_components(pred)
    g_lab, n_gt = connected_components(gt)

    pred_sizes = component_voxel_counts(p_lab, n_pred).tolist()[1:]
    gt_sizes = component_voxel_counts(g_lab, n_gt).tolist()[1:]

    detected_gt = []
    tp_pred = 0
    per_gt_overlap = [0] * n_gt

    if n_pred > 0 and n_gt > 0:
        # Cross-tabulate overlap voxels via flat indices.
        # Use ndi.labeled_comprehension or histogram2d-like approach.
        # Build a (n_gt+1) x (n_pred+1) overlap matrix, skipping background.
        mask_both = (p_lab > 0) & (g_lab > 0)
        flat_pred = p_lab[mask_both].ravel()
        flat_gt = g_lab[mask_both].ravel()
        idx = flat_gt.astype(np.int64) * (n_pred + 1) + flat_pred.astype(np.int64)
        overlap_flat = np.bincount(idx, minlength=(n_gt + 1) * (n_pred + 1))
        overlap = overlap_flat.reshape(n_gt + 1, n_pred + 1)
        # GT detection
        gt_max_overlap = overlap[1:, 1:].max(axis=1) if n_pred > 0 else np.zeros(n_gt)
        per_gt_overlap = gt_max_overlap.astype(int).tolist()
        for gi in range(n_gt):
            if per_gt_overlap[gi] >= min_overlap_voxels:
                detected_gt.append(gi + 1)
        # Pred TP
        pred_max_overlap = overlap[1:, 1:].max(axis=0) if n_gt > 0 else np.zeros(n_pred)
        tp_pred = int((pred_max_overlap >= min_overlap_voxels).sum())

    tp_gt = len(detected_gt)  # number of detected GT lesions
    fn = n_gt - tp_gt
    fp = n_pred - tp_pred
    # For precision/recall we follow the convention that the per-direction
    # "TP" can differ (one pred can cover multiple GTs, and vice versa). This
    # matches the ATLAS reference implementation.
    if n_gt == 0 and n_pred == 0:
        precision = recall = f1 = 1.0
    elif n_gt == 0:
        precision = 0.0
        recall = 1.0  # nothing to find; all "found"
        f1 = 0.0
    elif n_pred == 0:
        precision = 1.0  # vacuously no FPs
        recall = 0.0
        f1 = 0.0
    else:
        precision = tp_pred / n_pred
        recall = tp_gt / n_gt
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return LesionWiseMetrics(
        gt_lesion_count=n_gt,
        pred_lesion_count=n_pred,
        tp_lesions=tp_gt,
        fp_lesions=fp,
        fn_lesions=fn,
        lesion_precision=float(precision),
        lesion_recall=float(recall),
        lesion_f1=float(f1),
        abs_lesion_count_diff=abs(n_pred - n_gt),
        signed_lesion_count_diff=n_pred - n_gt,
        per_gt_overlap=per_gt_overlap,
        per_gt_size_voxels=gt_sizes,
        per_pred_size_voxels=pred_sizes,
        detected_gt_indices=detected_gt,
    )


# --- size-binned recall ----------------------------------------------------

DEFAULT_SIZE_BINS_VOXELS: tuple[tuple[str, int, int], ...] = (
    ("tiny_<10",      0,    10),
    ("small_10-100",  10,   100),
    ("medium_100-1k", 100,  1000),
    ("large_1k-10k",  1000, 10000),
    ("huge_>=10k",    10000, 10**12),
)

DEFAULT_SIZE_BINS_ML: tuple[tuple[str, float, float], ...] = (
    ("vsmall_<0.1mL",   0.0,   0.1),
    ("small_0.1-1mL",   0.1,   1.0),
    ("med_1-10mL",      1.0,   10.0),
    ("large_10-50mL",   10.0,  50.0),
    ("vlarge_>=50mL",   50.0,  1e9),
)


def size_binned_recall_voxels(lesion_metrics: LesionWiseMetrics,
                              bins: Iterable[tuple[str, int, int]] = DEFAULT_SIZE_BINS_VOXELS,
                              min_overlap_voxels: int = 1) -> dict[str, dict[str, float]]:
    """Per-bin lesion recall keyed by GT component size in voxels."""
    out: dict[str, dict[str, float]] = {}
    sizes = np.asarray(lesion_metrics.per_gt_size_voxels, dtype=np.int64)
    overlaps = np.asarray(lesion_metrics.per_gt_overlap, dtype=np.int64)
    for name, lo, hi in bins:
        in_bin = (sizes >= lo) & (sizes < hi)
        n = int(in_bin.sum())
        det = int(((overlaps >= min_overlap_voxels) & in_bin).sum())
        out[name] = {
            "n_gt": n,
            "n_detected": det,
            "recall": (det / n) if n > 0 else float("nan"),
        }
    return out


def size_binned_recall_ml(lesion_metrics: LesionWiseMetrics,
                          voxel_size_mm: tuple[float, float, float],
                          bins: Iterable[tuple[str, float, float]] = DEFAULT_SIZE_BINS_ML,
                          min_overlap_voxels: int = 1) -> dict[str, dict[str, float]]:
    vox_ml = _voxel_volume_ml(voxel_size_mm)
    sizes_ml = np.asarray(lesion_metrics.per_gt_size_voxels, dtype=np.float64) * vox_ml
    overlaps = np.asarray(lesion_metrics.per_gt_overlap, dtype=np.int64)
    out: dict[str, dict[str, float]] = {}
    for name, lo, hi in bins:
        in_bin = (sizes_ml >= lo) & (sizes_ml < hi)
        n = int(in_bin.sum())
        det = int(((overlaps >= min_overlap_voxels) & in_bin).sum())
        out[name] = {
            "n_gt": n,
            "n_detected": det,
            "recall": (det / n) if n > 0 else float("nan"),
        }
    return out


# --- top-level per-case bundle ---------------------------------------------

@dataclass
class CaseMetrics:
    subject_id: str
    voxel_size_mm: tuple[float, float, float]
    overlap: OverlapMetrics
    volume: VolumeMetrics
    boundary: BoundaryMetrics
    lesionwise: LesionWiseMetrics
    size_bins_voxels: dict[str, dict[str, float]]
    size_bins_ml: dict[str, dict[str, float]]
    chronicity: str | None = None    # "acute", "chronic", "mixed", or None
    center: str | None = None
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "voxel_size_mm": list(self.voxel_size_mm),
            "chronicity": self.chronicity,
            "center": self.center,
            "overlap": self.overlap.__dict__,
            "volume": self.volume.__dict__,
            "boundary": self.boundary.__dict__,
            "lesionwise": {
                k: v for k, v in self.lesionwise.__dict__.items()
            },
            "size_bins_voxels": self.size_bins_voxels,
            "size_bins_ml": self.size_bins_ml,
            "extras": self.extras,
        }


def evaluate_case(pred: np.ndarray, gt: np.ndarray,
                  voxel_size_mm: tuple[float, float, float],
                  subject_id: str,
                  chronicity: str | None = None,
                  center: str | None = None,
                  min_overlap_voxels: int = 1) -> CaseMetrics:
    overlap = overlap_metrics(pred, gt)
    volume = volume_metrics(pred, gt, voxel_size_mm)
    boundary = boundary_metrics(pred, gt, voxel_size_mm)
    lesion = lesion_wise_metrics(pred, gt, min_overlap_voxels=min_overlap_voxels)
    bins_vox = size_binned_recall_voxels(lesion, min_overlap_voxels=min_overlap_voxels)
    bins_ml = size_binned_recall_ml(lesion, voxel_size_mm, min_overlap_voxels=min_overlap_voxels)
    return CaseMetrics(
        subject_id=subject_id,
        voxel_size_mm=tuple(float(v) for v in voxel_size_mm),
        overlap=overlap,
        volume=volume,
        boundary=boundary,
        lesionwise=lesion,
        size_bins_voxels=bins_vox,
        size_bins_ml=bins_ml,
        chronicity=chronicity,
        center=center,
    )
