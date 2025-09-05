import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from hydra.utils import instantiate
import numpy as np
import random

class EBM(pl.LightningModule):
    def __init__(self, model, **kwargs):
        """
        Args:
            encoder: A neural network that encodes input x to a latent representation.
            energy_net: A network that outputs a scalar energy given the encoder's output.
        """
        super().__init__()
        self.energy_net = instantiate(model.energy_net)
        self.sampler = instantiate(model.sampler, log_prob=self.u_log_prob)

        self.register_buffer("buffer", torch.randn([model.buffer_size, kwargs['data'].dim, *list(kwargs['data'].shape)]))

        kwargs['model'] = model
        self.save_hyperparameters(kwargs)

    def forward(self, x, *args, return_losses=False, **kwargs):

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
    

    def training_step(self, batch, batch_idx):
        x, mask = batch
        loss, loss_dict = self.forward(x, mask, return_losses=True)
        self.log('train/loss', loss.mean(), on_step=True, on_epoch=False, prog_bar=True)
        loss_dict = {f'train/{k}': v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, on_step=True, on_epoch=False, prog_bar=False)
        return loss.mean()
        
    def validation_step(self, batch, batch_idx):
        x, mask = batch
        loss, loss_dict = self.forward(x, mask, return_losses=True)
        self.log('val/loss', loss.mean(), on_step=False, on_epoch=True, prog_bar=True)
        loss_dict = {f'val/{k}': v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=False)
        return loss.mean()
    
    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.train.optimizer, params=self.parameters())
        scheduler = instantiate(self.hparams.train.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]