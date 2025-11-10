import random

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from torch.nn import functional as F

from src.models.ebm import EBM
from src.utils import freeze, load_model


class EBMLatentTilting(EBM):
    def __init__(
        self, base_model, energy_net, sampler, cfg=None, ckpt_path=None, *args, **kwargs
    ):
        """
        Args:
            energy_net: A network that outputs a scalar energy given the encoder's output.
            sampler: A sampler class that generates samples from the energy model.
            cfg: Configuration DictConfig
        """
        super().__init__(
            energy_net=energy_net,
            sampler=sampler,
            cfg=cfg,
            ckpt_path=ckpt_path,
            *args,
            **kwargs
        )

        self.base_model, _ = load_model(ckpt_path=base_model)

        # Freeze base model
        freeze(self.base_model)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def forward(self, x, mask, *args, return_losses=False, **kwargs):
        """Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor.
            return_losses (bool, optional): Whether to return losses. Defaults to False.
        """

        z_true = self.base_model.encode(x=x, mask=mask)[0]
        z_prior = self.base_model.sample_prior(num_samples=z_true.shape[0])
        z_sampled = self.sample_prior(num_samples=x.shape[0], init=z_prior)

        # Add small noise to the input
        # small_noise = torch.randn_like(x) * 0.005
        # x.add_(small_noise).clamp_(min=-1.0, max=1.0)

        if return_losses:
            loss, loss_dict = self.loss(z_true, z_sampled, return_losses=return_losses)
            return loss, loss_dict
        else:
            return self.loss(z_true, z_sampled)

    def loss(self, z_real, z_sampled, return_losses=False):
        """
        Args:
            z_real: Samples from the true data distribution encoded in the latent space
            z_sampled: Negative samples (e.g., from a generator or Langevin dynamics) starting from prior samples
        Returns:
            loss: A scalar loss value
            loss_dict: A dictionary of individual losses (if return_losses is True)
        """
        if self.training:
            z_real.requires_grad_(True)

        z_all = torch.cat([z_real, z_sampled], dim=0)

        # Compute energy for real and sampled data
        energy_real, energy_sampled = self.energy_net(z_all)[:, 0].chunk(2, dim=0)

        # Energy loss
        energy_loss = energy_real - energy_sampled

        # Reg loss
        reg_loss, dic_reg = self.regularization_term(
            z_real, z_sampled, energy_real, energy_sampled
        )
        loss = reg_loss + energy_loss

        if return_losses:
            loss_dict = {
                "energy_real": energy_real.mean(),
                "energy_sampled": energy_sampled.mean(),
                "energy_loss": energy_loss.mean(),
            }
            loss_dict.update(dic_reg)
            return loss, loss_dict
        else:
            return loss

    def u_log_prob(self, z):
        """
        Unnormalized log probability of the data under the energy model.
        Args:

            z: Input data
        Returns:
            log_prob: Unnormalized log probability
        """
        energy = self.energy_net(z)[:, 0]

        base_logp = self.base_model.log_prob_prior(z)

        return (
            -energy + base_logp
        )  # EBM outputs energy, we return -energy for log probability

    def sample_prior(self, num_samples=None, init=None, return_init=False, **kwargs):
        """
        Sample from the energy model in the latent space using the provided sampler.
        Args:
            num_samples: Number of samples to generate
            return_init: Whether to return the initial latent samples before MCMC
        Returns:
            samples: Generated samples from the energy model
            init: Initial samples from the base model prior (if return_init is True)
        """
        if init is None:
            init = self.base_model.sample_prior(num_samples=num_samples)

        # Before MCMC: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input.
        is_training = self.training
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

        # Generate samples using the sampler
        samples = self.sampler(init=init, **kwargs)

        # samples = samples.reshape(samples.shape[0], -1, self.energy_net.input_dim)

        # Reactivate gradients for parameters for training
        for n, p in self.named_parameters():
            if not n.startswith("base_model."):  # skip base_model params
                p.requires_grad = True
        self.train(is_training)

        if return_init:
            return samples, init
        else:
            return samples

    def sample(self, num_samples=None, init=None, return_init=False, **kwargs):
        """
        Sample from the energy model and decode to data space.
        Args:
            num_samples: Number of samples to generate
            return_init: Whether to return the initial latent samples before MCMC
        Returns:
            samples: Generated samples from the energy model
            init: Initial samples from the base model (if return_init is True)
        """
        out_prior = self.sample_prior(
            num_samples=num_samples, init=None, return_init=return_init, **kwargs
        )

        if return_init:
            z_samples, z_init = out_prior
        else:
            z_samples = out_prior

        x_samples = self.base_model.decode(z_samples)
        if return_init:
            x_init = self.base_model.decode(z_init)
            x_init = self.base_model.likelihood.logits_to_data(x_init)
            return x_samples, x_init
        else:
            return x_samples
