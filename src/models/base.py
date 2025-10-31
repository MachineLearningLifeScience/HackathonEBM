import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from hydra.utils import instantiate
import numpy as np
import random

class BaseModel(pl.LightningModule):
    def __init__(self, cfg, ignore_keys=(), *args, **kwargs):
        """
        Args:
            encoder: A neural network that encodes input x to a latent representation.
            energy_net: A network that outputs a scalar energy given the encoder's output.
        """
        super().__init__()
        self.save_hyperparameters(cfg, logger=False)

    def init_from_ckpt(self, path, ignore_keys=()):
        """Initialize model from a checkpoint, ignoring specified keys.
        Args:
            path (str): Path to the checkpoint file.
            ignore_keys (tuple, optional): Keys to ignore when loading the state dict. Defaults to ().
        """ 
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def forward(self, x, *args, return_losses=False, **kwargs):
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.
            return_losses (bool, optional): Whether to return losses. Defaults to False.
        """
        pass

    def loss(self, *args, return_losses=False, **kwargs):
        """
        Args:
            return_losses (bool, optional): Whether to return a dictionary of losses. Defaults to False.
        Returns:
            loss: A scalar loss value
            loss_dict: A dictionary of individual losses (if return_losses is True)
        """
        pass

    def sample(self, num_samples=None, *args, **kwargs):
        """
        Sample from the energy model using the provided sampler.
        Args:
            num_samples: Number of samples to generate
        Returns:
            samples: Generated samples from the energy model
        """
        pass

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            x = batch[0]
            mask = batch[1] if len(batch) == 2 else None
        else:
            x = batch
            mask = None
        loss, loss_dict = self.forward(x, mask, return_losses=True)
        self.log('train/loss', loss.mean(), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        loss_dict = {f'train/{k}': v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        return loss.mean()
        
    def validation_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            x = batch[0]
            mask = batch[1] if len(batch) == 2 else None
        else:
            x = batch
            mask = None
        loss, loss_dict = self.forward(x, mask, return_losses=True)
        self.log('val/loss', loss.mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Metric for ckpt callback
        self.log('val_loss', loss.mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        loss_dict = {f'val/{k}': v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss.mean()
    
    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.train.optimizer, params=self.parameters())
        if hasattr(self.hparams.train, "scheduler"):
            scheduler = instantiate(self.hparams.train.scheduler, optimizer=optimizer)
            return [optimizer], [scheduler]
        return optimizer