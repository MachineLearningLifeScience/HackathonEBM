import random

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
import wandb


class ClassifySamples(pl.Callback):
    """
    Callback to classify generated samples and plot the bar results.
    """

    def __init__(self, classifier, num_samples=256, every_n_epochs=5, **kwargs):
        super().__init__()
        self.classifier = classifier  # A pretrained classifier model
        self.num_samples = num_samples  # Number of images to plot
        self.every_n_epochs = every_n_epochs  # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.kwargs = kwargs  # Additional arguments to pass to the sample function
        self.class_names = [str(i) for i in range(10)]  # Assuming 10 classes (0-9)

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        with torch.no_grad():
            samples = pl_module.sample(
                self.num_samples,
                **self.kwargs,
            )

        if type(samples) is tuple:
            samples, _ = samples

        self.classifier = self.classifier.to(samples.device)
        self.classifier.eval()
        # Classify the samples
        logits = self.classifier(samples)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        # Count occurrences of each class
        counts = np.bincount(preds, minlength=len(self.class_names))

        fig, ax = plt.subplots()
        ax.bar(self.class_names, counts)
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Class Distribution of Generated Samples at Epoch {trainer.current_epoch+1}"
        )

        # Log to wandb
        image = wandb.Image(
            fig,
            caption=f"Epoch {trainer.current_epoch+1} Class Distribution",
        )
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.log(
                {f"Class Distribution": image, "epoch": trainer.current_epoch}
            )
        plt.close(fig)
