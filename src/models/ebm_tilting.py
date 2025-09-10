import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from hydra.utils import instantiate
import numpy as np
import random
from src.models.ebm import EBM
from src.utils import freeze, load_model

class EBMTilting(EBM):
    def __init__(self, base_model, energy_net, sampler, cfg=None, ckpt_path=None, *args, **kwargs):
        """
        Args:
            energy_net: A network that outputs a scalar energy given the encoder's output.
            sampler: A sampler class that generates samples from the energy model.
            cfg: Configuration DictConfig
        """
        super().__init__(energy_net=energy_net, sampler=sampler, cfg=cfg, ckpt_path=ckpt_path, *args, **kwargs)

        self.base_model, _ = load_model(ckpt_path=base_model)
        
        # Freeze base model
        freeze(self.base_model)    

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def forward(self, x, *args, return_losses=False, **kwargs):
        """Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor.
            return_losses (bool, optional): Whether to return losses. Defaults to False.
        """ 

        init = self.base_model.sample(num_samples=x.shape[0])

        x_sampled = self.sample(num_samples=x.shape[0], init=init)

        # Add small noise to the input
        small_noise = torch.randn_like(x) * 0.005
        x.add_(small_noise).clamp_(min=-1.0, max=1.0)
        
        if return_losses:
            loss, loss_dict = self.loss(x, x_sampled, return_losses=return_losses)
            return loss, loss_dict
        else:
            return self.loss(x, x_sampled)
        
    def loss(self, x_real, x_sampled, return_losses=False):
        """
        Args:
            x_real: Samples from the true data distribution
            x_sampled: Negative samples (e.g., from a generator or Langevin dynamics)
        Returns:
            loss: A scalar loss value
            loss_dict: A dictionary of individual losses (if return_losses is True)
        """
        if self.training:
            x_real.requires_grad_(True)

        x_all = torch.cat([x_real, x_sampled], dim=0)
        # Compute energy for real and sampled data
        energy_real, energy_sampled = self.energy_net(x_all)[:,0].chunk(2, dim=0)

        # Energy loss
        energy_loss =  energy_real - energy_sampled

        # Reg loss
        reg_loss = self.hparams.model.lambda_reg * (energy_real ** 2 + energy_sampled ** 2)

        loss = reg_loss + energy_loss

        if return_losses:
            loss_dict = {
                'energy_real': energy_real.mean(),
                'energy_sampled': energy_sampled.mean(),
                'energy_loss': energy_loss.mean(),
                'reg_loss': reg_loss.mean(),
                }
            return loss, loss_dict
        else:        
            return loss


    def u_log_prob(self, x):
        """
        Unnormalized log probability of the data under the energy model.
        Args:

            x: Input data
        Returns:
            log_prob: Unnormalized log probability
        """
        energy = self.energy_net(x)[:,0]

        base_logp = self.base_model.log_prob(x)

        return -energy + base_logp # EBM outputs energy, we return -energy for log probability
    
    def sample(self, num_samples=None, init=None, return_init=False, **kwargs):
        """
        Sample from the energy model using the provided sampler.
        Args:
            num_samples: Number of samples to generate
        Returns:
            samples: Generated samples from the energy model
        """
        init = self.base_model.sample(num_samples=num_samples)
        
        # Before MCMC: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input. 
        is_training = self.training
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

        # Generate samples using the sampler
        samples = self.sampler(init=init, **kwargs)

        # Reactivate gradients for parameters for training
        for n, p in self.named_parameters():
            if not n.startswith("base_model."):   # skip base_model params
                p.requires_grad = True
        self.train(is_training)
        
        if return_init:
            return samples, init
        else:
            return samples            


    