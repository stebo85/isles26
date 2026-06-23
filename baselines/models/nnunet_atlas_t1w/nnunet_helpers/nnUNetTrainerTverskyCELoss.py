"""Precision-aware Tversky+CE trainer for nnU-Net v2.

The MSL (size-stratified) relabeling raised small-lesion recall but cost ~0.8%
Dice because the extra small-class predictions are mostly false positives. The
Tversky loss generalizes soft Dice with separate weights on FP and FN:

    Tversky = TP / (TP + alpha*FP + beta*FN)

Setting alpha > beta (here 0.7 / 0.3) penalizes false positives more than false
negatives, i.e. it is precision-aware. This mirrors the SynthStroke campaign's
Run H, where Tversky(alpha=0.7) fixed over-segmentation (precision 0.18 -> 0.41).

This file defines the loss, a Tversky+CE compound (mirroring DC_and_CE_loss),
and the trainer. It is copied into the installed nnU-Net package by
install_tversky_trainer.py so nnU-Net can auto-discover the trainer class.
"""
from typing import Callable

import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1


class MemoryEfficientSoftTverskyLoss(nn.Module):
    """Soft Tversky loss, structured exactly like MemoryEfficientSoftDiceLoss."""

    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True,
                 smooth: float = 1., ddp: bool = True, alpha: float = 0.7, beta: float = 0.3):
        super().__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))
            if x.shape == y.shape:
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes)
            sum_pred = x.sum(axes)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)

        if self.batch_dice:
            if self.ddp:
                from nnunetv2.utilities.ddp_allgather import AllGatherGrad
                intersect = AllGatherGrad.apply(intersect).sum(0)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0)
            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        fp = sum_pred - intersect
        fn = sum_gt - intersect
        tversky = (intersect + self.smooth) / torch.clip(
            intersect + self.alpha * fp + self.beta * fn + self.smooth, 1e-8)
        return -tversky.mean()


class Tversky_and_CE_loss(nn.Module):
    """Tversky + cross-entropy compound, mirroring DC_and_CE_loss."""

    def __init__(self, tversky_kwargs, ce_kwargs, weight_ce=1, weight_tversky=1, ignore_label=None):
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.weight_ce = weight_ce
        self.weight_tversky = weight_tversky
        self.ignore_label = ignore_label
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.tv = MemoryEfficientSoftTverskyLoss(apply_nonlin=softmax_helper_dim1, **tversky_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1
            mask = target != self.ignore_label
            target_tv = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_tv = target
            mask = None

        tv_loss = self.tv(net_output, target_tv, loss_mask=mask) if self.weight_tversky != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        return self.weight_ce * ce_loss + self.weight_tversky * tv_loss


class nnUNetTrainerTverskyCELoss(nnUNetTrainer):
    """Default 1000-epoch nnU-Net with a precision-aware Tversky(0.7/0.3)+CE loss."""

    alpha = 0.7
    beta = 0.3

    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = Tversky_and_CE_loss(
            {"batch_dice": self.configuration_manager.batch_dice, "smooth": 1e-5, "do_bg": False,
             "ddp": self.is_ddp, "alpha": self.alpha, "beta": self.beta},
            {},
            weight_ce=1,
            weight_tversky=1,
            ignore_label=self.label_manager.ignore_label,
        )
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerTverskyCELoss_250epochs(nnUNetTrainerTverskyCELoss):
    """250-epoch schedule for cheap fold-0 hypothesis tests."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        self.save_every = 10
