import torch
import pytorch_lightning as pl
import torch.distributions as td
from hydra.utils import instantiate
from src.utils import reparam
from src.models.vae import VAE

class MIWAE(VAE):
    """
    Missing data Importance Weighted Autoencoder (MIWAE)
    Reference: Mattei & Frellsen, 2019 (https://arxiv.org/abs/1902.10661)
    """
    def __init__(self, model, **kwargs):
        super(MIWAE, self).__init__(model, **kwargs)

    def elbo(self, x, mask, K=None, return_losses=False):
        """
        x: [batch, input_dim]
        mask: [batch, input_dim] binary mask (1=observed, 0=missing)
        K: Number of importance samples (default: self.hparams.model.K)
        """
        if K==None:
            K = self.hparams.model.K
        mean, logvar = self.encode(x)
        z = reparam(mean, logvar, K)  # [batch, K, latent_dim]

        # Compute log q(z|x_obs)
        qz_mean = mean.unsqueeze(1).expand(-1, K, -1)  # [batch, K, latent_dim]
        qz_logvar = logvar.unsqueeze(1).expand(-1, K, -1)
        qz = td.Normal(qz_mean, torch.exp(0.5 * qz_logvar))
        log_qz = qz.log_prob(z).sum(-1)  # [batch, K]

        # Compute log p(z)
        pz = td.Normal(torch.zeros_like(z), torch.ones_like(z))
        log_pz = pz.log_prob(z).sum(-1)  # [batch, K]

        # Decode
        x_recon = self.decode(z)  # [batch, K, input_dim]

        # Compute log p(x_obs|z)
        # Only observed values contribute to likelihood
        log_px = self.likelihood.log_prob_mask(x, x_recon, mask=mask, mode='sum')  # [batch, K]

        # Importance weights
        log_w = log_px + log_pz - log_qz  # [batch, K]
        log_w = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)

        # MIWAE ELBO
        elbo = torch.logsumexp(log_px + log_pz - log_qz, dim=1) - torch.log(torch.tensor(K, device=x.device, dtype=x.dtype))
        loss = -elbo
        if return_losses:
            loss_dict = {
                'elbo': elbo.mean(),
                'loss': loss.mean() 
            }
            return loss, loss_dict
        return loss

    def reconstruct(self, x, mask, K=None):
        """
        x: [batch, input_dim] 
        mask: [batch, input_dim] binary mask (1=observed, 0=missing)
        """
        if K is None:
            K = self.hparams.model.K
        with torch.no_grad():
            mean, logvar = self.encode(x)
            z = reparam(mean, logvar, K, unsqueeze=True)  # [batch, K, latent_dim]
            
            # Decode
            x_recon = self.decode(z)  # [batch, K, input_dim]
            
            # Get weights
            log_px = self.likelihood.log_prob_mask(x, x_recon, mask=mask, mode='sum')  # [batch, K]
            log_pz = td.Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(-1)  # [batch, K]
            log_qz = td.Normal(mean.unsqueeze(1).expand(-1, K, -1), torch.exp(0.5 * logvar).unsqueeze(1)).log_prob(z).sum(-1)  # [batch, K]
            log_w = log_px + log_pz - log_qz  # [batch, K]
            log_w = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)  # Normalize weights

            # Reconstruct
            # Expand log_w to match x_recon's dimensions for broadcasting
            weights = torch.exp(log_w).view(log_w.shape[0], log_w.shape[1], *([1] * (x_recon.dim() - 2)))
            x_recon = x_recon * weights
            x_recon = x_recon.sum(dim=1)  # Weighted sum over K

        return x_recon