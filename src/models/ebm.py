import random

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from torch.nn import functional as F
from src.utils import freeze
from src.models.base import BaseModel



class EBM(BaseModel):
    def __init__(
        self,
        energy_net,
        sampler,
        buffer_size=None,
        init_dist='normal',
        cfg=None,
        ckpt_path=None,
        *args,
        **kwargs
    ):
        """
        Args:
            energy_net: A network that outputs a scalar energy given the encoder's output.
            sampler: A sampler class that generates samples from the energy model.
            buffer_size: Size of the replay buffer for persistent contrastive divergence.
            cfg: Configuration DictConfig
        """
        super().__init__(cfg)
        self.energy_net = instantiate(energy_net)
        self.sampler = instantiate(sampler, log_prob=self.u_log_prob)
        self.init_dist = init_dist

        if buffer_size is not None:
            self.register_buffer(
                "buffer",
                self.get_init(buffer_size)
            )
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def forward(self, x, *args, return_losses=False, **kwargs):
        """Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor.
            return_losses (bool, optional): Whether to return losses. Defaults to False.
        """

        if self.training and hasattr(self, "buffer"):
            # Buffer sampling
            init = self.get_buffer(x.shape[0])
        else:
            init = self.get_init(x.shape[0])
        x_sampled = self.sample(num_samples=x.shape[0], init=init)
        self.update_buffer(x_sampled.detach())

        # Add small noise to the input
        x = x + torch.randn_like(x) * 0.005

        if self.init_dist == 'uniform':
            x.clamp_(min=-1.0, max=1.0)

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
        energy_real, energy_sampled = self.energy_net(x_all)[:, 0].chunk(2, dim=0)

        # Energy loss
        energy_loss = energy_real - energy_sampled

        # Reg loss
        reg_all, reg_losses = self.regularization_term(
            x_real, x_sampled, energy_real, energy_sampled
        )

        loss = reg_all + energy_loss

        if return_losses:
            loss_dict = {
                "energy_real": energy_real.mean(),
                "energy_sampled": energy_sampled.mean(),
                "energy_loss": energy_loss.mean(),
                "reg_loss": reg_all.mean(),
            }
            loss_dict.update(reg_losses)
            return loss, loss_dict
        else:
            return loss

    def get_init(self, num_samples):
        """
        Get initial samples for MCMC from the specified distribution.
        Args:
            num_samples: Number of samples to generate
        Returns:
            init: (num_samples, *data_shape)
        """
        if self.init_dist == 'uniform':
            init = torch.rand(
                num_samples,
                self.hparams.data.dim,
                *self.hparams.data.shape,
                device=self.device
            ) * 2 - 1  # Uniform in [-1, 1]
        elif self.init_dist == 'normal':
            init = torch.randn(
                num_samples,
                self.hparams.data.dim,
                *self.hparams.data.shape,
                device=self.device
            )  # Standard normal
        else:
            raise ValueError(f"Unknown init_dist: {self.init_dist}")
        return init


    def u_log_prob(self, x):
        """
        Unnormalized log probability of the data under the energy model.
        Args:

            x: Input data
        Returns:
            log_prob: Unnormalized log probability
        """
        energy = self.energy_net(x)[:, 0]
        return -energy  # EBM outputs energy, we return -energy for log probability

    def get_buffer(self, num_samples=None):
        """
        Function for getting a batch of "fake" images from the buffer.
        Inputs:
            num_samples - Number of samples to return
        Returns:
            init - (num_samples, C, H, W) tensor of images in [-1, 1]
        """
        if num_samples is None:
            num_samples = self.hparams.train.batch_size
        # Choose 95% of the batch from the buffer, 5% generate from scratch
        n_new = np.random.binomial(num_samples, 0.05)
        # fresh noise on the SAME device
        rand_imgs = self.get_init(n_new)

        # sample buffer by indices (fast, on-device)
        if self.buffer.size(0) >= num_samples - n_new:
            idx = torch.randint(0, self.buffer.size(0), (num_samples - n_new,), device=self.buffer.device)
            old_imgs = self.buffer[idx]
        else:
            old_imgs = self.buffer

        init = torch.cat([rand_imgs, old_imgs], dim=0).detach()
        return init

    def update_buffer(self, samples):
        with torch.no_grad():
            B = self.buffer.size(0)
            n = min(samples.size(0), B)
            # simple FIFO: put new in front, shift old down
            self.buffer = torch.cat([samples[:n].detach(), self.buffer], dim=0)[:B]

    def sample(self, num_samples=None, init=None, **kwargs):
        """
        Sample from the energy model using the provided sampler.
        Args:
            num_samples: Number of samples to generate
        Returns:
            samples: Generated samples from the energy model
            Shape: [num_samples, channel_dim, *input_shape]
        """
        if num_samples is None:
            num_samples = self.hparams.train.batch_size
        if init is None:
            init = self.get_init(num_samples)
            
        # Before MCMC: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input.
        is_training = self.training
        self.eval()
        freeze(self)


        # Generate samples using the sampler
        samples = self.sampler(init=init, **kwargs)

        # Reactivate gradients for parameters for training
        for p in self.parameters():
            p.requires_grad = True
        self.train(is_training)

        return samples

    def regularization_term(self, x_real, x_sampled, energy_real, energy_sampled):
        """
        Compute the regularization term for the energy model.
        Args:
            x_real: Real input data
            x_sampled: Sampled input data
            energy_real: Energy of real data
            energy_sampled: Energy of sampled data
        Returns:
            reg_term: The computed regularization term
        """
        reg_term = torch.tensor([0.0], device=x_real.device)

        reg_losses = {}

        if self.hparams.model.get("wgan_gp_lambda", 0.0) > 0.0:
            loss = self.hparams.model.wgan_gp_lambda * self.wgan_gradient_penalty(
                x_real, x_sampled
            )
            reg_losses["wgan_gp"] = loss
            reg_term += loss
        
        # L2 energy penalty
        loss = self.l2_penalty(energy_real, energy_sampled)
        reg_losses["l2_energy"] = loss
        reg_term += self.hparams.model.get("l2_energy_lambda", 0.0) * loss

        # L2 parameter penalty
        loss = self.l2_params()
        reg_losses["l2_params"] = loss
        reg_term += self.hparams.model.get("l2_params_lambda", 0.0) * loss

        return reg_term, reg_losses

    def wgan_gradient_penalty(
        self,
        x_real,
        x_sampled,
    ):
        """
        Compute the WGAN gradient penalty regularization term.
        The gradient penalty is computed on random interpolations between real and sampled data.
        The intuition is to enforce the Lipschitz constraint by penalizing the norm of the gradients.
        Args:
            x_real: Real input data
            x_sampled: Sampled input data
            model: The model whose gradients are to be computed
        Returns:
            reg_term: The computed gradient regularization term
        """
        with torch.enable_grad():
            x_real.requires_grad_(True)
            max_batch = min(x_real.size(0), x_sampled.size(0))
            epsilon = torch.randn(
               *([max_batch] + [1] * (x_real.dim() - 1)), device=x_real.device
            )  # Interpolation value

            x_interpolated = (
                (epsilon * x_real[:max_batch] + (1 - epsilon) * x_sampled[:max_batch])
                .detach()
                .requires_grad_(True)
            )
            
            energy_interpolated = self.energy_net(x_interpolated).mean()
            gradients = torch.autograd.grad(
                outputs=energy_interpolated,
                inputs=x_interpolated,
                grad_outputs=torch.ones_like(energy_interpolated),
                create_graph=True,
                retain_graph=True,
            )[0]
            gradients = gradients.view(gradients.size(0), -1)
            gradient_norm = gradients.norm(2, dim=1)
            gradient_penalty = ((gradient_norm - 1) ** 2).mean()
            return gradient_penalty

    def l2_penalty(self, energy_real, energy_sampled):
        """
        Penalize large absolute energy values (stabilizes scale).
        """
        return ((energy_real**2) + (energy_sampled**2)).mean()

    def l2_params(
        self,
    ):
        """
        Compute the L2 norm of the model parameters.
        Returns:
            l2_norm: The computed L2 norm of the parameters
        """
        l2_norm = 0.0
        for param in self.parameters():
            l2_norm += torch.sum(param**2)
        return l2_norm
