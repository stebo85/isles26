from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_250epochs(nnUNetTrainer):
    """Time-bounded fold-0 baseline trainer: default nnU-Net with 250 epochs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 250
        self.save_every = 10
