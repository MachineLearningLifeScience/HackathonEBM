import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
import wandb


class ImageReconstruction(pl.Callback):
    def __init__(self, dataloader, num_images=8, every_n_epochs=1, **kwargs):
        super().__init__()
        self.dataloader = dataloader
        self.num_images = num_images
        self.every_n_epochs = every_n_epochs
        self.kwargs = kwargs

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        with torch.no_grad():
            batch = next(iter(self.dataloader))
            if isinstance(batch, (tuple, list)):
                images, mask = batch[0][: self.num_images], batch[1][: self.num_images]
            else:
                images = batch[: self.num_images]
            images = images.to(pl_module.device)
            mask = mask.to(pl_module.device)
            reconstructions = pl_module.reconstruct(images, mask, **self.kwargs)

        # Move to [0,1]
        images = (images + 1) / 2
        reconstructions = (reconstructions + 1) / 2
        reconstructions = torch.clamp(reconstructions, min=0, max=1)

        # Create a grid of originals and reconstructions
        grid = vutils.make_grid(
            torch.cat([images, reconstructions], dim=0),
            nrow=self.num_images,
            padding=2,
        )

        # Log to wandb
        image = wandb.Image(
            grid,
            caption=f"Epoch {trainer.current_epoch+1} (Top: Originals, Bottom: Reconstructions)",
        )
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.log(
                {"Reconstructions": image, "epoch": trainer.current_epoch}
            )
