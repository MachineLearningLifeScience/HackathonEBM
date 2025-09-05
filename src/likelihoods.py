import torch
import torch.nn as nn

from torch.nn.modules.loss import BCELoss, BCEWithLogitsLoss
import torch.nn.functional as F
import numpy as np


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
        """
        mc_mode = False
        if len(logits.shape) == 5:
            mc_mode = True
            # MC samples
            bs, mc_samples, dim_params, h, w = logits.shape
            feats = data.shape[1]
            logits = logits.reshape(bs*mc_samples, dim_params, h, w)
            # repeat data and mask
            data = data.unsqueeze(1).repeat(1, mc_samples, 1, 1, 1)  # (bs, feats, h, w) -> (bs, mc_samples, feats, h, w)
            mask = mask.unsqueeze(1).repeat(1, mc_samples, 1, 1, 1)  # (bs, feats, h, w) -> (bs, mc_samples, feats, h, w)

            data = data.reshape(bs*mc_samples, feats, h, w)
            mask = mask.reshape(bs*mc_samples, -1, h, w)


        if mask is None:
            mask = torch.ones_like(data)
        
        log_prob = self.log_prob(data, logits, params=params, *args, **kwargs) * mask[..., 0, :,:] # mask is (bs, ..., feats, h, w)
        
        if reduce:
            # Sum over elements and features
            log_prob = log_prob.sum((-1, -2))   # sum across, h and w

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
    
    def __init__(self, binary=True):
        # Bernoulli does not require any param
        super(BernoulliLikelihood, self).__init__(type='bernoulli')
        self.binary = binary

    def log_prob(self, data, logits, params=False, **kwargs):

        # Data have to be in {0,1} (binary)
        data = (data + 1) / 2
        if params:
            logp = -BCELoss(reduction='none')(logits, data).sum(1)
        else:
            logp = -BCEWithLogitsLoss(reduction='none')(logits, data).sum(1)

        return logp

    def logits_to_params(self, logits, **kwargs):
        return torch.sigmoid(logits)

    def logits_to_data(self, logits, sample=True, **kwargs):
        probs = self.logits_to_params(logits)
        return self.params_to_data(probs,sample=sample)
    
    def params_to_data(self, params, sample=True, **kwargs):
        if self.binary:
            if sample:
                samples = torch.bernoulli(params)
            else:
                samples = torch.round(params)
        else:
            samples = params
        samples = samples * 2 - 1
        return samples
        


class GaussianLikelihood(Likelihood):
    """
        Gaussian Distribution 
    """
    def __init__(self, var=None):
        super(GaussianLikelihood, self).__init__(type='gaussian')
        if var is not None:
            self.register_buffer('logvar', torch.tensor(np.log(var)))

    def logits_to_params(self, logits, temperature=1., *args, **kwargs):
        if hasattr(self, 'logvar'):
            mean = logits
            logvar = self.logvar * torch.ones_like(mean)
        else:
            mean, logvar = torch.chunk(logits, 2, dim=-3)
            # Add minimum variance
            logvar = torch.clamp(logvar, min=self.logvar)
        logvar = logvar + np.log(temperature)
        return mean, logvar
    
    def logits_to_data(self, logits, temperature=1., sample=True, *args, **kwargs):
        mean, logvar = self.logits_to_params(logits, temperature=temperature)
        if sample:
            sample = self.reparam(mean, logvar)
        else:
            sample = mean
        return sample
   
    def log_prob(self, data, logits, *args, **kwargs):
        mean, logvar = self.logits_to_params(logits)
        logp = -0.5 * (np.log(2 * np.pi) + logvar + (data - mean) ** 2 / torch.exp(logvar))
        return logp.sum(1)
    