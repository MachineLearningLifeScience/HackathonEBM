import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
import wandb


class ImageSamples(pl.Callback):
    def __init__(self, num_samples=8, every_n_epochs=1, **kwargs):
        super().__init__()
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs
        self.kwargs = kwargs

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        with torch.no_grad():
            samples = pl_module.sample(self.num_samples, **self.kwargs)

        if type(samples) is tuple:
            samples, init = samples

            # Move to [0,1]
            init = (init + 1) / 2
            init = torch.clamp(init, min=0, max=1)

            grid = vutils.make_grid(
                init, nrow=int(np.sqrt(self.num_samples)), padding=2
            )
            model_name = pl_module.__class__.__name__
            image = wandb.Image(
                grid,
                caption=f"Epoch {trainer.current_epoch+1} ({model_name} Init Samples)",
            )
            if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
                trainer.logger.experiment.log(
                    {"Init samples": image, "epoch": trainer.current_epoch}
                )

        # Move to [0,1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, min=0, max=1)

        grid = vutils.make_grid(samples, nrow=int(np.sqrt(self.num_samples)), padding=2)
        model_name = pl_module.__class__.__name__
        image = wandb.Image(
            grid, caption=f"Epoch {trainer.current_epoch+1} ({model_name} Samples)"
        )
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.log(
                {"Samples": image, "epoch": trainer.current_epoch}
            )
