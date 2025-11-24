import pytorch_lightning as pl
import torch
import torch.distributions as td
from hydra.utils import instantiate

from src.models.base import BaseModel
from src.utils import reparam
import numpy as np


class VAE(BaseModel):
    """
    Variational Autoencoder (VAE) base class.
    This class implements the basic structure of a VAE, including encoding, decoding,
    and the evidence lower bound (ELBO) computation.
    It is intended to be subclassed for specific VAE implementations.
    """

    def __init__(
        self,
        latent_dim,
        encoder,
        decoder,
        likelihood,
        K=1,
        cfg=None,
        ckpt_path=None,
        *args,
        **kwargs
    ):
        """
        Args:
            encoder: A neural network that encodes input x to a latent representation.
            decoder: A neural network that decodes latent representation z to reconstruct x.
            likelihood: A likelihood model that computes log probabilities of the data given reconstructions.
            K: Number of Monte Carlo samples for the ELBO estimation.
            cfg: Configuration DictConfig
        """

        super().__init__(cfg)

        self.latent_dim = latent_dim
        self.encoder = instantiate(encoder)
        self.decoder = instantiate(decoder)
        self.likelihood = instantiate(likelihood)
        self.K = K  # Number of MC samples

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def encode(self, x, mask):
        params = self.encoder(x, mask)  # Some encoders will process the mask, others won't
        if isinstance(params, tuple):
            mean, logvar = params
        else:
            mean, logvar = torch.chunk(params, 2, dim=-1)
        return mean, logvar

    def decode(self, z):
        """
        Decode latent variables z to reconstruct input x.
        Handles both single sample and multiple samples (K) cases.
        Automatically handles the shape of the output based on likelihood parameters.
        E.G., for categorical likelihood, the output shape of the parameter corresponds to one hot
        So categorical with 5 classes and input dim 3 will have output shape [batch, K, 3, ..., 5]
        
        Args:
            z: Latent variables, shape [batch, latent_dim] or [batch, K, latent_dim]
        Returns:
            x_recon: Reconstructed inputs, shape [batch, 1, input_dim/channel*(param_dim), *input_shape,]
            or [batch, K, input_dim/channel*(param_dim), *input_shape,]. 
        """

        # Check reconstruction size
        if hasattr(self.hparams.model, 'param_output_channel'):
            param_output_channel = self.hparams.model.param_output_channel # In case of categorical likelihood, the output shape of the parameter corresponds to one hot
            reconstruction_size = [self.hparams.data.dim*param_output_channel] + list(self.hparams.data.shape)
        else:
            reconstruction_size = [self.hparams.data.dim] + list(self.hparams.data.shape)
        
        if len(z.size()) == 2:
            batch, latent_dim = z.size()
            input_size = self.hparams.data.shape
            z_flat = z
            logits_recon = self.decoder(z_flat)
            return logits_recon.view(batch, 1, *reconstruction_size)
        else:
            batch, K, latent_dim = z.size()
            z_flat = z.view(batch*K, latent_dim)
            logits_recon = self.decoder(z_flat)
            
            return logits_recon.view(batch, K, *reconstruction_size)

    def forward(self, x, mask, return_losses=False):
        if return_losses:
            elbo, losses = self.elbo(x, mask, return_losses=True)
            return -elbo, losses
        else: 
            return -self.elbo(x, mask)

    def sample_prior(self, num_samples, device=None):
        """
        Sample from the prior distribution p(z).
        Args:
            num_samples: Number of samples to draw.
            device: Device to perform the sampling on.
        Returns:
            z: Samples from the prior distribution.
        """
        device = device or next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return z

    def log_prob_prior(self, z):
        """
        Compute the log probability of samples under the prior distribution p(z).
        Args:
            z: Samples to compute the log probability for.
        Returns:
            log_prob: Log probabilities of the samples under the prior.
        """
        p = td.Normal(torch.zeros_like(z), torch.ones_like(z))
        return p.log_prob(z).sum(-1)  # Sum over latent dimensions

    def elbo(self, x, mask, K=None, return_losses=False):
        """
        x: [batch, input_dim]
        mask: [batch, input_dim] binary mask (1=observed, 0=missing)
        """
        if K == None:
            K = self.K

        if mask is None:
            mask = torch.ones(x.shape[0], 1, *x.shape[2:], device=x.device)  # (bs, 1, h, w)
            
        mean, logvar = self.encode(x*mask, mask)
        z = reparam(mean, logvar, K, unsqueeze=True)  # [batch, K, latent_dim]

        # Decode
        logits_recon = self.decode(z)  # [batch, K, channel_dim*param_dim, *input_shape]

        # Compute log p(x_obs|z)
        # Only observed values contribute to likelihood
        log_px = self.likelihood.log_prob_mask(
            x, logits_recon, mask=mask, mode="sum"
        )  # [batch, K]

        # Compute KL(q(z|x_obs) || p(z))
        q = td.Normal(mean, torch.exp(0.5 * logvar))
        p = td.Normal(
            torch.zeros_like(mean), torch.ones_like(mean)
        )  # Standard normal prior
        kl = td.kl_divergence(q, p).sum(-1, keepdim=True)  # [batch, K]

        # ELBO
        elbo = log_px - kl  # [batch, K]
        if return_losses:
            loss_dict = {
                "log_px": log_px.mean(),
                "kl": kl.mean(),
                "elbo": elbo.mean(),
            }
            return elbo.mean(dim=1), loss_dict

        return elbo.mean(dim=1)  # Average over K samples

    def log_prob(self, x, mask=None, K=None):
        return self.elbo(x, mask, K)

    def reconstruct(self, x, mask=None, K=None, *args, **kwargs):
        """
        Args:
            x: [batch, input_dim]
            mask: [batch, input_dim] binary mask (1=observed, 0=missing)
        
        Returns:
            x_recon: Reconstructed input [batch, input_dim]
        """
        if K is None:
            K = self.K

        if mask is None:
            mask = torch.ones(x.shape[0], 1, *x.shape[2:], device=x.device)  # (bs, 1, h, w)

        with torch.no_grad():
            mean, logvar = self.encode(x, mask)
            z = reparam(mean, logvar, K, unsqueeze=True)  # [batch, K, latent_dim]
            # Decode
            logits_recon = self.decode(z)  # [batch, K, input_dim]
            logits_recon = logits_recon.mean(dim=1)  # Weighted sum over K
            x_recon = self.likelihood.logits_to_data(logits_recon)
        return x_recon

    def sample(self, num_samples, device=None, *args, **kwargs):
        """
        Generate samples from the model's prior.
        Returns: [num_samples, channnel_dim, *input_shape]
        """
        device = device or next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        logits_samples = self.decode(z) # [num_samples, 1, channel_dim*param_dim, *input_shape]
        x_samples = self.likelihood.logits_to_data(logits_samples, *args, **kwargs).flatten(0,1)
        return x_samples

    def impute(self, x, mask, K=None):
        """
        Impute missing values in x using the trained model.
        """
        with torch.no_grad():
            x_recon = self.reconstruct(x, mask, K=K)
            x_out = x.clone()
            x_out[mask == 0] = x_recon[mask == 0]
        return x_out
