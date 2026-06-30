"""Synthetic lesion insertion augmentation for nnU-Net v2.

The transform operates on one nnU-Net training patch after spatial/intensity
augmentation and before deep-supervision target downsampling. It inserts compact
ellipsoidal T1w-hypointense lesions with a size distribution biased toward tiny
and small connected components, then writes the corresponding lesion label.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


SizeBin = Tuple[int, int, float]


class SyntheticLesionInsertionTransform(BasicTransform):
    """Insert size-rebalanced synthetic lesions into a binary lesion patch.

    Expected inputs are nnU-Net v2 per-sample tensors:
      image: (C, X, Y, Z), float
      segmentation: (1, X, Y, Z), integer labels, with -1 for padded regions

    The default size bins intentionally over-represent the challenge-risk region:
    tiny (<10 voxels) and small (10-100 voxels).
    """

    default_size_bins: Tuple[SizeBin, ...] = (
        (4, 9, 0.25),
        (10, 99, 0.45),
        (100, 999, 0.25),
        (1000, 5000, 0.05),
    )

    def __init__(
        self,
        apply_probability: float = 0.35,
        lesion_label: int = 1,
        max_lesions: int = 2,
        size_bins: Sequence[SizeBin] | None = None,
        brain_epsilon: float = 1e-6,
        candidate_low_quantile: float = 0.10,
        min_candidate_voxels: int = 32,
        max_attempts_per_lesion: int = 16,
        axis_log_sigma: float = 0.45,
        contrast_std: Tuple[float, float] = (0.8, 1.8),
        low_quantile: Tuple[float, float] = (0.03, 0.20),
        texture_std_fraction: Tuple[float, float] = (0.05, 0.18),
        boundary_blend: Tuple[float, float] = (0.35, 0.75),
        rim_shift_std_fraction: Tuple[float, float] = (-0.10, 0.20),
    ):
        super().__init__()
        if not 0 <= apply_probability <= 1:
            raise ValueError("apply_probability must be in [0, 1]")
        if max_lesions < 1:
            raise ValueError("max_lesions must be >= 1")
        bins = tuple(size_bins) if size_bins is not None else self.default_size_bins
        if not bins:
            raise ValueError("size_bins must not be empty")
        for lo, hi, weight in bins:
            if lo < 1 or hi < lo or weight < 0:
                raise ValueError(f"invalid size bin {(lo, hi, weight)}")
        if sum(b[2] for b in bins) <= 0:
            raise ValueError("at least one size bin must have positive weight")
        if not 0 <= candidate_low_quantile <= 1:
            raise ValueError("candidate_low_quantile must be in [0, 1]")

        self.apply_probability = float(apply_probability)
        self.lesion_label = int(lesion_label)
        self.max_lesions = int(max_lesions)
        self.size_bins = bins
        self.brain_epsilon = float(brain_epsilon)
        self.candidate_low_quantile = float(candidate_low_quantile)
        self.min_candidate_voxels = int(min_candidate_voxels)
        self.max_attempts_per_lesion = int(max_attempts_per_lesion)
        self.axis_log_sigma = float(axis_log_sigma)
        self.contrast_std = contrast_std
        self.low_quantile = low_quantile
        self.texture_std_fraction = texture_std_fraction
        self.boundary_blend = boundary_blend
        self.rim_shift_std_fraction = rim_shift_std_fraction
        self.last_inserted_voxels: list[int] = []

    def apply(self, data_dict: dict, **params) -> dict:
        self.last_inserted_voxels = []
        if torch.rand((), device="cpu").item() >= self.apply_probability:
            return data_dict

        image = data_dict.get("image")
        segmentation = data_dict.get("segmentation")
        if image is None or segmentation is None:
            return data_dict
        if image.ndim != 4 or segmentation.ndim != 4 or segmentation.shape[0] < 1:
            return data_dict

        seg0 = segmentation[0]
        valid = seg0 >= 0
        brain = self._brain_mask(image, valid)
        candidate = self._candidate_mask(image, seg0, valid, brain)
        if int(candidate.sum().item()) < self.min_candidate_voxels:
            return data_dict

        center_mask = self._erode(candidate)
        if int(center_mask.sum().item()) < self.min_candidate_voxels:
            center_mask = candidate

        n_lesions = int(torch.randint(1, self.max_lesions + 1, (), device="cpu").item())
        inserted_masks = []
        for _ in range(n_lesions):
            lesion = self._sample_one_lesion(center_mask, candidate)
            if lesion is None or not bool(lesion.any().item()):
                continue
            if not self._apply_t1w_appearance(image, lesion, brain):
                continue
            seg0[lesion] = self.lesion_label
            candidate = candidate & ~lesion
            center_mask = center_mask & ~self._dilate(lesion)
            inserted_masks.append(lesion)
            self.last_inserted_voxels.append(int(lesion.sum().item()))

        if inserted_masks:
            data_dict["image"] = image
            data_dict["segmentation"] = segmentation
        return data_dict

    def _sample_target_voxels(self) -> int:
        total = float(sum(b[2] for b in self.size_bins))
        draw = torch.rand((), device="cpu").item() * total
        acc = 0.0
        lo, hi = self.size_bins[-1][0], self.size_bins[-1][1]
        for lo_i, hi_i, weight in self.size_bins:
            acc += float(weight)
            if draw <= acc:
                lo, hi = lo_i, hi_i
                break
        if lo == hi:
            return int(lo)
        log_lo = math.log(float(lo))
        log_hi = math.log(float(hi))
        value = math.exp(log_lo + torch.rand((), device="cpu").item() * (log_hi - log_lo))
        return max(1, int(round(value)))

    def _sample_one_lesion(self, center_mask: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor | None:
        centers = center_mask.nonzero(as_tuple=False)
        if centers.numel() == 0:
            return None
        spatial_shape = tuple(int(i) for i in candidate.shape)
        for _ in range(self.max_attempts_per_lesion):
            target_voxels = self._sample_target_voxels()
            center = centers[torch.randint(0, centers.shape[0], (), device=centers.device)].to(torch.float32)
            radii = self._sample_radii(target_voxels).to(center.device)
            margin = int(max(2, math.ceil(float(radii.max().item()) * 1.75 + 2)))
            slices = []
            local_center = []
            for dim, size in enumerate(spatial_shape):
                c = int(round(float(center[dim].item())))
                lo = max(0, c - margin)
                hi = min(size, c + margin + 1)
                slices.append(slice(lo, hi))
                local_center.append(float(center[dim].item() - lo))

            local_candidate = candidate[tuple(slices)]
            if not bool(local_candidate.any().item()):
                continue
            dist = self._ellipsoid_distance(local_candidate.shape, local_candidate.device, local_center, radii)
            raw = (dist <= 1.0) & local_candidate
            available = int(raw.sum().item())
            if available < max(1, min(target_voxels, self.size_bins[0][0]) // 2):
                continue

            keep = min(target_voxels, available)
            lesion_local = torch.zeros_like(local_candidate, dtype=torch.bool)
            raw_scores = dist[raw] + 0.025 * torch.rand(available, device=dist.device)
            selected_raw = torch.argsort(raw_scores)[:keep]
            raw_coords = raw.nonzero(as_tuple=False)
            selected_coords = raw_coords[selected_raw]
            lesion_local[selected_coords[:, 0], selected_coords[:, 1], selected_coords[:, 2]] = True

            lesion = torch.zeros_like(candidate, dtype=torch.bool)
            lesion[tuple(slices)] = lesion_local
            return lesion
        return None

    def _sample_radii(self, target_voxels: int) -> torch.Tensor:
        base = (3.0 * float(target_voxels) / (4.0 * math.pi)) ** (1.0 / 3.0)
        factors = torch.exp(torch.randn(3, device="cpu") * self.axis_log_sigma)
        factors = factors / torch.pow(torch.prod(factors), 1.0 / 3.0)
        radii = torch.clamp(torch.as_tensor(base, dtype=torch.float32) * factors, min=0.75)
        return radii

    @staticmethod
    def _ellipsoid_distance(
        shape: torch.Size | Tuple[int, ...],
        device: torch.device,
        center: Sequence[float],
        radii: torch.Tensor,
    ) -> torch.Tensor:
        coords = torch.meshgrid(
            [torch.arange(int(s), device=device, dtype=torch.float32) for s in shape],
            indexing="ij",
        )
        dist = torch.zeros(shape, device=device, dtype=torch.float32)
        for dim in range(3):
            r = float(max(float(radii[dim].item()), 0.5))
            dist = dist + ((coords[dim] - float(center[dim])) / r) ** 2
        return dist

    def _brain_mask(self, image: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        img0 = image[0].detach()
        finite = torch.isfinite(img0)
        nonzero = img0.abs() > self.brain_epsilon
        brain = valid & finite & nonzero
        if int(brain.sum().item()) < self.min_candidate_voxels:
            brain = valid & finite
        return brain

    def _candidate_mask(
        self,
        image: torch.Tensor,
        seg0: torch.Tensor,
        valid: torch.Tensor,
        brain: torch.Tensor,
    ) -> torch.Tensor:
        candidate = brain & valid & (seg0 != self.lesion_label)
        finite_candidate = candidate & torch.isfinite(image[0])
        vals = image[0][finite_candidate].float()
        if vals.numel() < self.min_candidate_voxels:
            return finite_candidate
        if bool((vals.max() <= vals.min()).item()):
            return finite_candidate
        threshold = torch.quantile(vals, self.candidate_low_quantile)
        return finite_candidate & (image[0] > threshold.to(image[0].dtype))

    def _apply_t1w_appearance(self, image: torch.Tensor, lesion: torch.Tensor, brain: torch.Tensor) -> bool:
        if int(lesion.sum().item()) == 0:
            return False
        ring = self._dilate(lesion) & ~lesion & brain
        lesion_eroded = self._erode(lesion)
        boundary = lesion & ~lesion_eroded
        core = lesion_eroded if bool(lesion_eroded.any().item()) else lesion

        bbox = self._bbox_from_mask(lesion | ring)
        if bbox is None:
            return False

        changed_any = False
        for ch in range(image.shape[0]):
            img = image[ch]
            brain_vals = img[brain & torch.isfinite(img)]
            if brain_vals.numel() < self.min_candidate_voxels:
                continue
            ref_mask = ring if int(ring.sum().item()) >= 8 else brain
            ref_vals = img[ref_mask & torch.isfinite(img)]
            if ref_vals.numel() < 8:
                ref_vals = brain_vals
            ref_mean = ref_vals.mean()
            ref_std = torch.clamp(ref_vals.std(unbiased=False), min=0.05)
            q = self._uniform(*self.low_quantile)
            low_ref = torch.quantile(brain_vals.float(), q)
            contrast = self._uniform(*self.contrast_std)
            target_mean = torch.minimum(ref_mean - contrast * ref_std, low_ref)

            local_slices = tuple(bbox)
            local_shape = image[ch][local_slices].shape
            texture = self._low_frequency_texture(local_shape, image.device, image.dtype)
            texture = texture * (ref_std * self._uniform(*self.texture_std_fraction)).to(image.dtype)
            target = (target_mean.to(image.dtype) + texture).to(image.dtype)

            img_local = img[local_slices].clone()
            core_l = core[local_slices]
            boundary_l = boundary[local_slices]
            ring_l = ring[local_slices]
            img_local[core_l] = target[core_l]
            if bool(boundary_l.any().item()):
                alpha = self._uniform(*self.boundary_blend)
                img_local[boundary_l] = (1.0 - alpha) * img_local[boundary_l] + alpha * target[boundary_l]
            if bool(ring_l.any().item()):
                rim_shift = ref_std.to(image.dtype) * self._uniform(*self.rim_shift_std_fraction)
                img_local[ring_l] = img_local[ring_l] + rim_shift
            img[local_slices] = img_local
            changed_any = True
        return changed_any

    @staticmethod
    def _bbox_from_mask(mask: torch.Tensor) -> list[slice] | None:
        coords = mask.nonzero(as_tuple=False)
        if coords.numel() == 0:
            return None
        lo = torch.clamp(coords.min(dim=0).values - 1, min=0)
        hi = torch.minimum(coords.max(dim=0).values + 2, torch.as_tensor(mask.shape, device=coords.device))
        return [slice(int(lo[i].item()), int(hi[i].item())) for i in range(3)]

    @staticmethod
    def _dilate(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        pad = kernel_size // 2
        return F.max_pool3d(mask[None, None].float(), kernel_size, stride=1, padding=pad)[0, 0] > 0

    @staticmethod
    def _erode(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        pad = kernel_size // 2
        pooled = F.avg_pool3d(mask[None, None].float(), kernel_size, stride=1, padding=pad)[0, 0]
        return pooled >= 1.0

    @staticmethod
    def _low_frequency_texture(shape: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if any(int(s) <= 1 for s in shape):
            return torch.zeros(shape, device=device, dtype=dtype)
        coarse_shape = tuple(max(1, min(4, int(s))) for s in shape)
        coarse = torch.randn((1, 1, *coarse_shape), device=device, dtype=torch.float32)
        tex = F.interpolate(coarse, size=tuple(int(s) for s in shape), mode="trilinear", align_corners=False)[0, 0]
        tex = tex - tex.mean()
        tex = tex / torch.clamp(tex.std(unbiased=False), min=1e-6)
        return tex.to(dtype)

    @staticmethod
    def _uniform(lo: float, hi: float) -> float:
        return float(lo + (hi - lo) * torch.rand((), device="cpu").item())
