"""R3+R4: heavy domain augmentation, on top of the Dice+TopK10 objective.

WHY
---
Center shift is the single largest measured loss in this project. Going from
stratified-random folds (52 of 55 centers appear on both sides of every fold) to
folds that hold out whole CENTERS costs:

    Dice   0.6558 -> 0.6347   (-0.021)
    AVD    4.94   -> 6.28 mL  (+27%)

and the hidden ISLES'26 test set spans 60+ centers of native-space clinical data.
Resolution is part of the same story: Dice by voxel spacing is 0.670 at 1 mm
isotropic (n=914), 0.637 anisotropic (n=133), 0.534 at (1.0, 0.977, 0.977)
(n=55) and 0.374 at (1.5, 1.5, 1.5) (n=21).

nnU-Net already ships `nnUNetTrainerDA5`, a substantially more aggressive
augmentation policy than the default: wider rotation and scaling, stronger
Gaussian noise and blur, brightness/contrast/gamma, and -- directly relevant to
the spacing spread above -- more aggressive simulate-low-resolution. Rather than
write an augmentation policy from scratch (the SynthStroke campaign in this repo
is a cautionary tale about hand-rolled augmentation with silently inert knobs,
see docs/problems_and_lessons.md), this composes the shipped, tested DA5
transforms with our best loss.

Method resolution order note: DA5 does not define `_build_loss`, and
nnUNetTrainerDiceTopK10Loss does, so `_build_loss` resolves to the TopK10 one
while every `get_*_transforms` resolves to DA5's. The result is exactly
"our current best model, trained with heavier augmentation", which is the single
variable we want to test.
"""

from __future__ import annotations

import torch

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import (
    nnUNetTrainerDA5,
)
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerTopkLoss import (
    nnUNetTrainerDiceTopK10Loss,
)


class nnUNetTrainerDA5TopK10(nnUNetTrainerDA5, nnUNetTrainerDiceTopK10Loss):
    """DA5 augmentation policy + Dice+TopK10 loss."""

    # nnU-Net reconstructs a trainer from its checkpoint by reading the __init__
    # kwargs back out of locals() BY NAME, so this signature must match the base
    # class exactly -- *args/**kwargs raises KeyError: 'args'. This is documented
    # in agents.md as gotcha (6) and has bitten this project before.
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
