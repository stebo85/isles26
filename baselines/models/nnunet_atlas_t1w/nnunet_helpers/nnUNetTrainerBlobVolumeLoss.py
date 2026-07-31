"""R1+R2: a loss aligned with the metrics the challenge actually ranks on.

WHY THIS TRAINER EXISTS
-----------------------
On the center-held-out surface our best model scores Dice 0.6283 but official
lesion-F1 (panoptica Recognition Quality, one-to-one matching at IoU >= 0.25)
only 0.5791. That gap is the whole problem: Dice+TopK10 optimises *voxel*
overlap, which is dominated by large lesions, while RQ needs *every instance*
recovered -- a 3-voxel lesion counts exactly as much as a 200 mL one.

Separately, absolute volume difference is one of the five ranked metrics and is
essentially uncorrelated with Dice (Spearman -0.081 on 1452 cases). Nothing in
the current objective optimises it, and three independent post-hoc attempts to
fix it after the fact all failed (see docs/lessons/problems.md Problem 22) --
the information simply is not in the predicted probability map. It has to be
trained in.

So this trainer adds two terms to the existing Dice+TopK10 objective:

1. BLOB (per-instance) term, after Kofler et al., "blob loss: instance imbalance
   aware loss functions for semantic segmentation" (2022). For each ground-truth
   connected component, compute a Dice against the prediction with every OTHER
   component masked out, then average over components. Each instance therefore
   contributes equally regardless of size. This is the term that targets RQ.

   Note this is NOT what MSL/DBL did -- those changed the LABELS (size-stratified
   classes). This changes the loss WEIGHTING and leaves the label space binary,
   so it composes with everything else and needs no new dataset.

2. VOLUME term: a bounded symmetric relative volume error
   |sum(p) - sum(g)| / (sum(p) + sum(g) + eps). Bounded and scale-free on
   purpose -- an unnormalised mL error would let a handful of very large lesions
   dictate the gradient, which is exactly the failure mode the blob term exists
   to avoid.

Both extra terms are applied at FULL RESOLUTION ONLY. Deep supervision keeps
wrapping the global Dice+TopK10 term as before. Running connected components at
every supervision scale would multiply the cost for no benefit: the instance
structure we care about is defined at full resolution.

Connected components use 26-connectivity, matching what
eval/probe_panoptica_semantics.py MEASURED the official scorer to use. Filtering
or weighting with a different connectivity than the metric counts with would
train against a different definition of "a lesion".
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.compound_losses import DC_and_topk_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# cc3d is a C++ connected-components implementation and is ~10x faster than
# scipy.ndimage.label on a 128^3 patch. It ships with nnU-Net's dependency set.
import cc3d

LESION_CHANNEL = 1


class BlobVolumeCompoundLoss(nn.Module):
    """Global Dice+TopK10 (deep-supervised) + per-instance term + volume term."""

    def __init__(self, base_loss: nn.Module, weight_blob: float, weight_volume: float,
                 max_instances: int, min_instance_voxels: int):
        super().__init__()
        self.base_loss = base_loss
        self.weight_blob = weight_blob
        self.weight_volume = weight_volume
        self.max_instances = max_instances
        self.min_instance_voxels = min_instance_voxels

    @staticmethod
    def _highest_res(x):
        return x[0] if isinstance(x, (list, tuple)) else x

    def _instance_labels(self, gt_bool: torch.Tensor) -> torch.Tensor:
        """26-connected component labels for one sample, computed on CPU."""
        arr = gt_bool.detach().to("cpu", torch.uint8).numpy()
        lab = cc3d.connected_components(arr, connectivity=26)
        return torch.as_tensor(lab.astype(np.int32), device=gt_bool.device)

    def blob_term(self, prob: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Mean over ground-truth instances of a per-instance soft Dice.

        `prob` is (B, X, Y, Z) lesion probability; `gt` is (B, X, Y, Z) binary.
        """
        total = prob.new_zeros(())
        n_terms = 0
        for b in range(prob.shape[0]):
            gt_b = gt[b] > 0.5
            if not torch.any(gt_b):
                continue  # a patch with no lesion has no instances to balance
            lab = self._instance_labels(gt_b)
            ids, counts = torch.unique(lab[lab > 0], return_counts=True)
            if ids.numel() == 0:
                continue
            keep = counts >= self.min_instance_voxels
            if not torch.any(keep):
                keep = counts == counts.max()  # never drop every instance
            ids, counts = ids[keep], counts[keep]
            if ids.numel() > self.max_instances:
                # Bound the cost on pathological cases (one ATLAS case has 84
                # components). Largest-first keeps the instances that carry the
                # most signal; the rest are still covered by the global term.
                order = torch.argsort(counts, descending=True)[: self.max_instances]
                ids = ids[order]
            other = lab > 0
            for i in ids:
                inst = lab == i
                # Mask out the OTHER instances so their voxels neither reward nor
                # punish this instance's Dice. This is what makes the term
                # per-instance rather than just size-weighted.
                valid = ~(other & ~inst)
                p = prob[b][valid]
                g = inst[valid].to(prob.dtype)
                inter = (p * g).sum()
                denom = p.sum() + g.sum()
                total = total + (1.0 - (2.0 * inter + 1e-5) / (denom + 1e-5))
                n_terms += 1
        if n_terms == 0:
            return prob.new_zeros(())
        return total / n_terms

    def volume_term(self, prob: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        p = prob.flatten(1).sum(1)
        g = (gt > 0.5).flatten(1).to(prob.dtype).sum(1)
        return (torch.abs(p - g) / (p + g + 1.0)).mean()

    def forward(self, output, target):
        loss = self.base_loss(output, target)
        logits = self._highest_res(output)
        tgt = self._highest_res(target)
        if tgt.ndim == logits.ndim:
            tgt = tgt[:, 0]
        prob = torch.softmax(logits, dim=1)[:, LESION_CHANNEL]
        if self.weight_blob > 0:
            loss = loss + self.weight_blob * self.blob_term(prob, tgt)
        if self.weight_volume > 0:
            loss = loss + self.weight_volume * self.volume_term(prob, tgt)
        return loss


class nnUNetTrainerBlobVolumeLoss(nnUNetTrainer):
    """Dice+TopK10 (the current best objective) + blob term + volume term."""

    weight_blob = 1.0
    weight_volume = 0.5
    max_instances = 16
    min_instance_voxels = 5

    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        # Identical to nnUNetTrainerDiceTopK10Loss so the comparison against the
        # existing fold-0 model isolates the two new terms and nothing else.
        base = DC_and_topk_loss(
            {"batch_dice": self.configuration_manager.batch_dice, "smooth": 1e-5,
             "do_bg": False, "ddp": self.is_ddp},
            {"k": 10, "label_smoothing": 0.0},
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
        )
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            base = DeepSupervisionWrapper(base, weights)
        return BlobVolumeCompoundLoss(
            base,
            weight_blob=self.weight_blob,
            weight_volume=self.weight_volume,
            max_instances=self.max_instances,
            min_instance_voxels=self.min_instance_voxels,
        )


class nnUNetTrainerBlobVolumeLoss_blobOnly(nnUNetTrainerBlobVolumeLoss):
    """Ablation: the instance term alone, to attribute any gain."""
    weight_volume = 0.0


class nnUNetTrainerBlobVolumeLoss_volumeOnly(nnUNetTrainerBlobVolumeLoss):
    """Ablation: the volume term alone."""
    weight_blob = 0.0
