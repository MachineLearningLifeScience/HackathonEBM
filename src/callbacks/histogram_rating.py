from time import time, sleep
import numpy as np
import pytorch_lightning as pl
import wandb
import torch
import matplotlib.pyplot as plt

class HistogramRatingCallback(pl.Callback):
    def __init__(self, num_samples=256, every_n_epochs=1, **kwargs):
        super().__init__()
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs
        self.kwargs = kwargs

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Callback to create an histogram of the ratings of generated samples.
        The ratings are in [0, 4], so we create 5 bins for the histogram.
        The idea is to show the bias in the generated samples.
        """
        if (trainer.current_epoch) % self.every_n_epochs != 0:
            return
        with torch.no_grad():
            samples = pl_module.sample(self.num_samples, return_init=True, **self.kwargs) # Get samples from the model
            

        if type(samples) is tuple:
            samples, init = samples

            init = init.flatten(1)
            fig = plt.figure()
            # plt.hist(samples.cpu().numpy(), bins=5, range=(0, 4))
            unique, counts = np.unique(init.flatten().cpu().numpy(), return_counts=True)
            counts = counts.astype(np.float32) / counts.sum()
            plt.bar(unique, counts, width=0.5, align='center')
            plt.xlabel("Rating")
            plt.ylabel("Frequency")
            plt.title(f"Epoch {trainer.current_epoch+1} Histogram")
            plt.grid()
            image = wandb.Image(
                fig, caption=f"Epoch {trainer.current_epoch+1} ({self.num_samples} Init Samples)"
            )
            if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
                trainer.logger.experiment.log(
                    {"Init Samples": image, "epoch": trainer.current_epoch}
                )
        fig = plt.figure()
        unique, counts = np.unique(samples.flatten().cpu().numpy(), return_counts=True)
        counts = counts.astype(np.float32) / counts.sum()
        plt.bar(unique, counts, width=0.5, align='center')
        plt.xlabel("Rating")
        plt.ylabel("Frequency")
        plt.title(f"Epoch {trainer.current_epoch+1} Histogram")
        plt.grid()
        image = wandb.Image(
            fig, caption=f"Epoch {trainer.current_epoch+1} ({self.num_samples} Samples)"
        )
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.log(
                {"Samples": image, "epoch": trainer.current_epoch}
            )
