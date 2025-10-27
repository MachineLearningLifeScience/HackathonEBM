import glob
import os

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from wandb import config
import numpy as np

# This allows to reverse a list in the config
OmegaConf.register_new_resolver("reverse", lambda x: list(reversed(x)))
OmegaConf.register_new_resolver("length", lambda x: len(x))
OmegaConf.register_new_resolver("mul", lambda x, y: x * y)
OmegaConf.register_new_resolver("index", lambda lst, i: lst[i])


def reparam(mu, logvar, n_samples=1, unsqueeze=False):
    """
    Reparameterization trick for VAE.
    Args:
        mu (torch.Tensor): Mean of the latent variable.
        logvar (torch.Tensor): Log variance of the latent variable.
        n_samples (int): Number of samples to draw from the latent distribution.
        unsqueeze (bool): If True, unsqueeze samples to (bs, 1, latent_dim) when n_samples = 1
    Returns:
        torch.Tensor: Sampled latent variable.
    """
    if n_samples > 1 or unsqueeze:
        mu = mu.unsqueeze(1).expand(-1, n_samples, -1)
        logvar = logvar.unsqueeze(1).expand(-1, n_samples, -1)

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def gaussian_log_prob(x, mean, logvar):
    """
    Compute the log probability of x under a Gaussian distribution with given mean and log variance.
    Args:
        x (torch.Tensor): Input data.
        mean (torch.Tensor): Mean of the Gaussian distribution.
        logvar (torch.Tensor): Log variance of the Gaussian distribution.
    Returns:
        torch.Tensor: Log probability of x.
    """
    logp = -0.5 * (np.log(2 * np.pi) + logvar + (x - mean) ** 2 / torch.exp(logvar))
    return logp


def freeze(model):
    for module in model.modules():
        module.eval()
        for param in module.parameters():
            param.grad = None
            param.requires_grad = False


def load_model(ckpt_path, map_location="cpu"):
    """
    Load a model from a checkpoint.
    Args:
        ckpt_path (str): Path to the checkpoint file.
        map_location (str): Device to map the model to.
    Returns:
        model (torch.nn.Module): Loaded model.
        config (omegaconf.DictConfig): Model configuration.
    """

    # Config is in parent parent folder
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(ckpt_path)), "config.yaml"
    )
    config = OmegaConf.load(config_path)

    # Model
    model = instantiate(config.model, cfg=config, ckpt_path=ckpt_path)

    return model, config


def get_wandb_run_id(log_dir):
    wandb_dir = os.path.join(log_dir, "wandb")
    if not os.path.exists(wandb_dir):
        return None
    run_dirs = glob.glob(os.path.join(wandb_dir, "run-*"))
    if not run_dirs:
        return None
    # Sort by modification time, descending
    run_dirs.sort(key=os.path.getmtime, reverse=True)
    latest_run = os.path.basename(run_dirs[0])
    if "-" in latest_run:
        return latest_run.split("-")[-1]
    return None
