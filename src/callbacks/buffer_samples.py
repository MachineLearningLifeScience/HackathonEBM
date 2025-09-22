import random

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
import wandb


class BufferSamples(pl.Callback):

    def __init__(self, num_samples=32, every_n_epochs=5, **kwargs):
        super().__init__()
        self.num_samples = num_samples  # Number of images to plot
        self.every_n_epochs = every_n_epochs  # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        imgs = torch.stack(random.choices(pl_module.buffer, k=self.num_samples), dim=0)

        # Move to [0,1]
        imgs = (imgs + 1) / 2
        imgs = torch.clamp(imgs, min=0, max=1)

        grid = vutils.make_grid(imgs, nrow=int(np.sqrt(self.num_samples)), padding=2)
        model_name = pl_module.__class__.__name__
        image = wandb.Image(
            grid,
            caption=f"Epoch {trainer.current_epoch+1} ({model_name} Bufer Samples)",
        )

        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.log(
                {"Buffer Samples": image, "epoch": trainer.current_epoch}
            )
