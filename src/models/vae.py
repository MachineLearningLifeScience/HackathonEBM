import pytorch_lightning as pl
import torch
import torch.distributions as td
from hydra.utils import instantiate

from src.models.base import BaseModel
from src.utils import reparam


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
        if hasattr(self.hparams.model, 'param_shape'):
            reconstruction_size = self.hparams.model.param_shape # In case of categorical likelihood, the output shape of the parameter corresponds to one hot
        else:
            reconstruction_size = self.hparams.data.shape # In this case, the output shape is the same as the data shape
        if len(z.size()) == 2:
            batch, latent_dim = z.size()
            z_flat = z
            x_recon = self.decoder(z_flat)
            return x_recon.view(batch, 1, *reconstruction_size) # [batch, 1, input_dim]
        else:
            batch, K, latent_dim = z.size()
            z_flat = z.view(batch*K, latent_dim)
            x_recon = self.decoder(z_flat)
            return x_recon.view(batch, K, *reconstruction_size)

    def forward(self, x, mask, return_losses=False):
        return self.elbo(x, mask, return_losses=return_losses)

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
        x_recon = self.decode(z)  # [batch, K, input_dim]

        # Compute log p(x_obs|z)
        # Only observed values contribute to likelihood
        log_px = self.likelihood.log_prob_mask(
            x, x_recon, mask=mask, mode="sum"
        )  # [batch, K]

        # Compute KL(q(z|x_obs) || p(z))
        q = td.Normal(mean, torch.exp(0.5 * logvar))
        p = td.Normal(
            torch.zeros_like(mean), torch.ones_like(mean)
        )  # Standard normal prior
        kl = td.kl_divergence(q, p).sum(-1, keepdim=True)  # [batch, K]

        # ELBO
        elbo = log_px - kl  # [batch, K]
        loss = -elbo
        if return_losses:
            loss_dict = {
                "log_px": log_px.mean(),
                "kl": kl.mean(),
                "elbo": elbo.mean(),
            }
            return loss.mean(dim=1), loss_dict

        return loss.mean(dim=1)  # Average over K samples

    def log_prob(self, x, mask=None, K=None):
        return self.elbo(x, mask, K)

    def reconstruct(self, x, mask=None, K=None, *args, **kwargs):
        """
        x: [batch, input_dim]
        mask: [batch, input_dim] binary mask (1=observed, 0=missing)
        """
        if K is None:
            K = self.K

        if mask is None:
            mask = torch.ones(x.shape[0], 1, *x.shape[2:], device=x.device)  # (bs, 1, h, w)

        with torch.no_grad():
            mean, logvar = self.encode(x, mask)
            z = reparam(mean, logvar, K, unsqueeze=True)  # [batch, K, latent_dim]
            # Decode
            x_recon = self.decode(z)  # [batch, K, input_dim]
            x_recon = x_recon.mean(dim=1)  # Weighted sum over K
            x_recon = self.likelihood.logits_to_data(x_recon)
        return x_recon

    def sample(self, num_samples, device=None, *args, **kwargs):
        """
        Generate samples from the model's prior.
        Returns: [num_samples, *input_dim]
        """
        device = device or next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        x_samples = self.decoder(z)
        x_samples = self.likelihood.logits_to_data(x_samples, *args, **kwargs)

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
