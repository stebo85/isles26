"""nnU-Net trainer variants with size-rebalanced synthetic lesion insertion."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerTopkLoss import nnUNetTrainerDiceTopK10Loss


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def _find_repo_root() -> Path | None:
    rel = Path("baselines/models/nnunet_atlas_t1w/nnunet_helpers/synthetic_lesion_augmentation.py")
    for candidate in [Path.cwd(), *Path(__file__).resolve().parents]:
        if (candidate / rel).is_file():
            return candidate
    return None


def _synthetic_lesion_transform_class():
    """Lazy import so nnU-Net trainer discovery cannot fail on repo PYTHONPATH."""
    try:
        from baselines.models.nnunet_atlas_t1w.nnunet_helpers.synthetic_lesion_augmentation import (
            SyntheticLesionInsertionTransform,
        )
        return SyntheticLesionInsertionTransform
    except ModuleNotFoundError as first_exc:
        repo = _find_repo_root()
        if repo is not None and str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
            from baselines.models.nnunet_atlas_t1w.nnunet_helpers.synthetic_lesion_augmentation import (
                SyntheticLesionInsertionTransform,
            )
            return SyntheticLesionInsertionTransform
        raise first_exc


class nnUNetTrainerDiceTopK10SynthLesion(nnUNetTrainerDiceTopK10Loss):
    """Dice+TopK10 trainer plus synthetic lesion insertion augmentation."""

    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        base = nnUNetTrainerDiceTopK10Loss.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=use_mask_for_norm,
            is_cascaded=is_cascaded,
            foreground_labels=foreground_labels,
            regions=regions,
            ignore_label=ignore_label,
        )
        transforms = list(base.transforms)
        SyntheticLesionInsertionTransform = _synthetic_lesion_transform_class()
        synth = SyntheticLesionInsertionTransform(
            apply_probability=_env_float("NNUNET_SYNTHLESION_PROB", 0.35),
            lesion_label=_env_int("NNUNET_SYNTHLESION_LABEL", 1),
            max_lesions=_env_int("NNUNET_SYNTHLESION_MAX_LESIONS", 2),
        )

        insert_at = len(transforms)
        for i, transform in enumerate(transforms):
            name = type(transform).__name__
            if name in {
                "RemoveLabelTansform",
                "MoveSegAsOneHotToDataTransform",
                "ConvertSegmentationToRegionsTransform",
                "DownsampleSegForDSTransform",
            }:
                insert_at = i
                break
        transforms.insert(insert_at, synth)
        return ComposeTransforms(transforms)


class nnUNetTrainerDiceTopK10SynthLesion_250epochs(nnUNetTrainerDiceTopK10SynthLesion):
    """Reduced schedule for fold-0 hypothesis tests."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        self.save_every = 10
