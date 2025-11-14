import random

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from torch.nn import functional as F

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

        x_sampled = self.sample(num_samples=x.shape[0])

        # Add small noise to the input
        x = x + torch.randn_like(x) * 0.005
        
        if return_losses:
            loss, loss_dict = self.loss(x, x_sampled, return_losses=return_losses)
            return loss, loss_dict
        else:
            return self.loss(x, x_sampled)

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
    
    def get_init(self, num_samples):
        """
        Get initial samples for MCMC from the base model.
        Args:
            num_samples: Number of samples to generate
        Returns:
            init: (num_samples, *data_shape)
        """
        init = self.base_model.sample_prior(num_samples)
        init = init.reshape(init.shape[0], self.hparams.data.dim, *self.hparams.data.shape)
        return init

    
    def sample(self, num_samples=None, init=None, return_init=False, **kwargs):
        """
        Sample from the energy model using the provided sampler.
        Args:
            num_samples: Number of samples to generate
        Returns:
            samples: Generated samples from the energy model
        """

        if init is None:
            init = self.base_model.sample(num_samples=num_samples)
        init = init.reshape(init.shape[0], self.hparams.data.dim, *self.hparams.data.shape)
        # Before MCMC: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input. 
        is_training = self.training
        self.eval()
        freeze(self)

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
