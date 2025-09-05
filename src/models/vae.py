import torch
import pytorch_lightning as pl
import torch.distributions as td
from hydra.utils import instantiate
from src.utils import reparam

class VAE(pl.LightningModule):
    """
    Variational Autoencoder (VAE) base class.
    This class implements the basic structure of a VAE, including encoding, decoding,
    and the evidence lower bound (ELBO) computation.
    It is intended to be subclassed for specific VAE implementations.   
    """
    def __init__(self, model, **kwargs):
        super(VAE, self).__init__()

        self.likelihood = instantiate(model.likelihood)
        self.encoder = instantiate(model.encoder)
        self.decoder = instantiate(model.decoder)

        kwargs['model'] = model
        
        self.save_hyperparameters(kwargs)

    def encode(self, x):
        mean, logvar = self.encoder(x)
        return mean, logvar

    def decode(self, z):
        # z: [batch, K, latent_dim]
        batch, K, latent_dim = z.size()
        input_size = self.hparams.model.input_size
        input_dim = self.hparams.model.input_dim

        z_flat = z.view(-1, latent_dim)
        x_recon = self.decoder(z_flat)
        return x_recon.view(batch, K, input_dim, *input_size)

    def forward(self, x, mask, return_losses=False):
        return self.elbo(x, mask, return_losses=return_losses)

    def elbo(self, x, mask, K=None, return_losses=False):
        """
        x: [batch, input_dim]
        mask: [batch, input_dim] binary mask (1=observed, 0=missing)
        """
        if K==None:
            K = self.hparams.model.K
        mean, logvar = self.encode(x)
        z = reparam(mean, logvar, K, unsqueeze=True)  # [batch, K, latent_dim]

        # Decode
        x_recon = self.decode(z)  # [batch, K, input_dim]

        # Compute log p(x_obs|z)
        # Only observed values contribute to likelihood
        log_px = self.likelihood.log_prob_mask(x, x_recon, mask=mask, mode='sum')  # [batch, K]

        # Compute KL(q(z|x_obs) || p(z))
        q = td.Normal(mean, torch.exp(0.5 * logvar))
        p = td.Normal(torch.zeros_like(mean), torch.ones_like(mean))  # Standard normal prior
        kl = td.kl_divergence(q, p).sum(-1,keepdim=True)  # [batch, K]

        # MIWAE ELBO
        elbo = log_px - kl  # [batch, K]
        loss = -elbo 
        if return_losses:
            loss_dict = {
                'log_px': log_px.mean(),
                'kl': kl.mean(),
                'elbo': elbo.mean(),
            }
            return loss.mean(dim=1), loss_dict

        return loss.mean(dim=1)  # Average over K samples

    def log_prob(self, x, mask, K=None):
        return self.elbo(x, mask, K)
    
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
            
            x_recon = x_recon.mean(dim=1)  # Weighted sum over K

        return x_recon

    def sample(self, num_samples, device=None):
        """
        Generate samples from the model's prior.
        Returns: [num_samples, *input_dim]
        """
        device = device or next(self.parameters()).device
        z = torch.randn(num_samples, *self.hparams.model.latent_dim, device=device)
        x_samples = self.decoder(z)
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
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=True)
        return loss.mean()
    
    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.train.optimizer, params=self.parameters())
        return optimizer