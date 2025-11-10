import torch
import torch.nn as nn

from torch.nn.modules.loss import BCELoss, BCEWithLogitsLoss
import torch.nn.functional as F
import numpy as np
from src.utils import reparam, gaussian_log_prob


class Likelihood(nn.Module):
    """
    Implements the likelihood functions
    """
    def __init__(self, type, **params):
        """
        Likelihood initialization
        """
        super(Likelihood, self).__init__()
        self.type=type
        self.params=params 
    
    def forward(self, data, logits) -> torch.Tensor:
        """
        Computes the log probability of a given data under parameters theta
        """
        return self.log_prob(data, logits)

    def logits_to_params(self, logits) -> torch.Tensor:
        """
        Apply an operation to the logits to obtain parameters of the data likelihood
        """
        pass

    def logits_to_data(self, logits) -> torch.Tensor:
        """
        Apply an operation to the logits and sample to obtain data
        """
        pass

    def log_prob(self, data, logits, params=False) -> torch.Tensor:
        """
        Computes the log probability of a given data under parameters theta
        """
        pass

    def log_prob_mask(self, data, logits, mask=None, mode="mean", params=False, reduce=True, *args, **kwargs) -> torch.Tensor:
        """
        Computes the log probability of a given masked data under parameters theta
        args:
            data: [batch, dim, *[datashape],]
            logits: [batch, mc, dim, *[datashape], potential_additional_param_shape] -> depending on likelihood. E.g., for Gaussian, last dim is 2 (mean and logvar)
            mask: [batch, mc, dim, *[datashape],] binary mask (1=observed, 0=missing)
            mode: "mean" or "sum" - how to reduce the log prob over observed points
            params: whether the logits are already converted to parameters
        """
        mc_mode = False

        if mask is None:
            mask = torch.ones_like(data[:, :1, ...])  # (bs, 1, *input_dims), assume all observed, enforce channel dim = 1

        
        
        if logits.shape[1]> 1: # Change the conditioning on the mc sampling
            mc_mode = True
            # MC samples
            bs, mc_samples, dim_params = logits.shape[:3]
            data_dims = data.shape[2:]

            logits = logits.flatten(0,1)
            
            # repeat data and mask
            data = data.unsqueeze(1).expand(bs, mc_samples, *data.shape[1:])  # (bs, feats, h, w) -> (bs, mc_samples, feats, h, w)
            mask = mask.unsqueeze(1).expand(bs, mc_samples, *mask.shape[1:])  # (bs, feats, h, w) -> (bs, mc_samples, feats, h, w)

            data = data.flatten(0,1)
            mask = mask.flatten(0,1)


        log_prob = self.log_prob(data, logits, params=params, *args, **kwargs) 
        log_prob = log_prob * mask  # Apply mask (assume channel dim = 1)

        if reduce:
            # Sum over elements and features
            log_prob = log_prob.flatten(1).sum(-1)   # sum across the data dimensions # TODO: Is there a channel specific handling ?

            # Sum over the activated points
            if mode == "mean":
                # Sum over element and features
                mask = mask[..., 0, :,:].sum((-1, -2))
                log_prob = log_prob / mask

            if mc_mode:
                log_prob = log_prob.reshape(bs, mc_samples)
   
        return log_prob


class BernoulliLikelihood(Likelihood):

    # ============= Bernoulli ============= #
    
    def __init__(self):
        # Bernoulli does not require any param
        super(BernoulliLikelihood, self).__init__(type='bernoulli')

    def log_prob(self, data, logits, params=False, **kwargs):

        # Data have to be in {0,1} (binary)
        data = (data + 1) / 2
        if params:
            logp = -BCELoss(reduction='none')(logits, data).sum(1)
        else:
            logp = -BCEWithLogitsLoss(reduction='none')(logits, data).sum(1)

        return logp

    def logits_to_params(self, logits, **kwargs):
        params = torch.sigmoid(logits)
        return params

    def logits_to_data(self, logits, sample=True, **kwargs):
        probs = self.logits_to_params(logits)
        data = self.params_to_data(probs, sample=sample)
        return data
    
    def params_to_data(self, params, sample=True, **kwargs):
        if sample:
            samples = torch.bernoulli(params)
        else:
            samples = torch.round(params)
        # Scale back to [-1, 1]
        samples = samples * 2 - 1
        return samples
        


class GaussianLikelihood(Likelihood):
    """
        Gaussian Distribution 
    """
    def __init__(self, min_logvar=None, learn_var=False):
        super(GaussianLikelihood, self).__init__(type='gaussian')
        self.min_logvar = min_logvar
        self.learn_var = learn_var

    def logits_to_params(self, logits, temperature=1., *args, **kwargs):
        if self.learn_var:
            mean, logvar = logits.chunk(2, dim=1)
            # Add minimum variance
            logvar = torch.clamp(logvar, min=self.min_logvar)
        else:
            mean = logits
            logvar = torch.ones_like(mean) * self.min_logvar
        logvar = logvar + np.log(temperature)
        return mean, logvar
    
    def logits_to_data(self, logits, temperature=1., sample=True, *args, **kwargs):
        mean, logvar = self.logits_to_params(logits, temperature=temperature)
        if sample:
            sample = reparam(mean, logvar)
        else:
            sample = mean
        return sample
   
    def log_prob(self, data, logits, *args, **kwargs):
        mean, logvar = self.logits_to_params(logits)
        logp = gaussian_log_prob(data, mean, logvar)
        return logp.sum(-3)
    
