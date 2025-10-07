import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from hydra.utils import instantiate
import numpy as np
import random
from src.models.base import BaseModel

class EBM(BaseModel):
    def __init__(self, energy_net, sampler, buffer_size=1024, cfg=None, ckpt_path=None, *args, **kwargs):
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

        self.register_buffer("buffer", torch.randn([buffer_size, self.hparams.data.dim, *list(self.hparams.data.shape)]))

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def forward(self, x, *args, return_losses=False, **kwargs):
        """Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor.
            return_losses (bool, optional): Whether to return losses. Defaults to False.
        """ 

        if self.training:
            # Buffer sampling
            init = self.get_buffer(x.shape[0])
        else:
            init = torch.rand(x.shape[0], self.hparams.data.dim, *self.hparams.data.shape, device=self.device) * 2 - 1

        x_sampled = self.sample(num_samples=x.shape[0], init=init)
        self.update_buffer(x_sampled.detach())

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
        rand_imgs = torch.rand(n_new, self.hparams.data.dim, *self.hparams.data.shape, device=self.device) * 2 - 1
        old_imgs = torch.stack(random.choices(self.buffer, k=num_samples-n_new), dim=0)
        init = torch.cat([rand_imgs, old_imgs], dim=0).detach()
        return init

    def update_buffer(self, samples):
        """
        Function for getting a new batch of "fake" images.
        Inputs:
            steps - Number of iterations in the MCMC algorithm
            step_size - Learning rate nu in the algorithm above
        """
        # Add new images to the buffer and remove old ones if needed
        buffer = torch.cat([samples, self.buffer])
        self.buffer = buffer[:self.buffer.size(0)] 


    def sample(self, num_samples=None, init=None, **kwargs):
        """
        Sample from the energy model using the provided sampler.
        Args:
            num_samples: Number of samples to generate
        Returns:
            samples: Generated samples from the energy model
        """
        if num_samples is None:
            num_samples = self.hparams.train.batch_size
        if init is None:
            init = torch.rand(num_samples, self.hparams.data.dim, *self.hparams.data.shape, device=self.device) * 2 - 1
        
        # Before MCMC: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input. 
        is_training = self.training
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

        # Generate samples using the sampler
        samples = self.sampler(init=init, **kwargs)

        # Reactivate gradients for parameters for training
        for p in self.parameters():
            p.requires_grad = True
        self.train(is_training)
        
        return samples            
    