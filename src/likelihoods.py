import torch
import torch.nn as nn

from torch.nn.modules.loss import BCELoss, BCEWithLogitsLoss
import torch.nn.functional as F
import numpy as np
from src.utils import reparam, gaussian_log_prob
from typing import Union

class Likelihood(nn.Module):
    """
    Implements the likelihood functions
    """
    def __init__(self, param_dim: int, type:str, **params):
        """
        Likelihood initialization
        """
        super(Likelihood, self).__init__()
        self.type=type
        self.params=params
        self.param_dim=param_dim 
    
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

        if mask is None:
            mask = torch.ones_like(data[:, :1, ...])  # (bs, 1, *input_dims), assume all observed, enforce channel dim = 1

        if logits.shape[1]> 1: # If MC samples, expand data and mask
            mc_mode = True
            # MC samples
            bs, mc_samples  = logits.shape[:2]
            # assert not params or (params and dim_params == self.param_dim), "When providing params, the param dim must match the likelihood param dim"
            # data_dims = data.shape[2:]
            logits = logits.flatten(0,1) # just flatten mc smaples
            
            # repeat data and mask
            data = data.unsqueeze(1).expand(bs, mc_samples, *data.shape[1:])  # (bs, feats, *shape) -> (bs, mc_samples, feats, *shape)
            mask = mask.unsqueeze(1).expand(bs, mc_samples, *mask.shape[1:])  # (bs, feats, *shape) -> (bs, mc_samples, feats, *shape)
            data = data.flatten(0,1)
            mask = mask.flatten(0,1)
        
        # Here logits shape is (bs*mc_samples, param_channel*data_dim, *data_shape)
        log_prob = self.log_prob(data, logits, params=params, *args, **kwargs)
        assert (mask.shape[0] == log_prob.shape[0]) and (mask.shape[2:] == log_prob.shape[2:]), "Mask and log prob must have the same shape, got {} and {}".format(mask.shape, log_prob.shape)
        log_prob = log_prob * mask  

        if reduce:
            # Sum over elements and features
            log_prob = log_prob.flatten(1).sum(-1)   # sum across channel and shapes

            # Sum over the activated points
            if mode == "mean":
                # Sum over element and features
                mask = mask.flatten(1).sum(-1)  # sum across channel and shapes
                log_prob = log_prob / mask

            if mc_mode:
                log_prob = log_prob.reshape(bs, mc_samples)
   
        return log_prob


class BernoulliLikelihood(Likelihood):
    def __init__(self, param_dim=None):
        # Bernoulli does not require any param
        super(BernoulliLikelihood, self).__init__(param_dim=param_dim, type='bernoulli')

    def log_prob(self, data, logits, params=False, **kwargs):
        """
        Compute the log probability of data under Bernoulli distribution
        Args:
            data (torch.Tensor): Input data, should be of shape [batch, dim, *data_shape]
            logits (torch.Tensor): Output logits from the decoder, should be of shape [batch*K, param_channel, *data_shape]
            params (bool): Whether the logits are already converted to parameters
        Returns:
            logp (torch.Tensor): Log probability of the data under the Bernoulli distribution.
        """

        if data.min() == -1 and data.max() == 1:
            data = (data + 1) / 2  # Scale data from [-1, 1] to [0, 1]
        if params:
            logp = -BCELoss(reduction='none')(logits, data)
        else:
            logp = -BCEWithLogitsLoss(reduction='none')(logits, data)
        return logp

    def logits_to_params(self, logits, **kwargs):
        """
        Transform logits to Bernoulli parameters
        Args:
            logits (torch.Tensor): Output logits from the decoder, should be of shape [batch,
            K, param_channel, *data_shape]
        Returns:
            params (torch.Tensor): Parameters of the Bernoulli distribution.
        """
        params = torch.sigmoid(logits)
        return params

    def logits_to_data(self, logits, sample=True, **kwargs):
        """
        Transform logits to data samples
        Args:
            logits (torch.Tensor): Output logits from the decoder, should be of shape [batch,
            K, param_channel, *data_shape]
            sample (bool): Whether to sample from the distribution or return the mode.
        Returns:
            samples (torch.Tensor): Sampled data from the Bernoulli distribution.
        """
        probs = self.logits_to_params(logits)
        data = self.params_to_data(probs, sample=sample)
        return data
    
    def params_to_data(self, params, sample=True, **kwargs):
        """
        Use parameters obtain data samples
        Args:
            params (torch.Tensor): Parameters of the Bernoulli distribution.
            sample (bool): Whether to sample from the distribution or return the mode.
        Returns:
            samples (torch.Tensor): Sampled data from the Bernoulli distribution.
        """
        if sample:
            samples = torch.bernoulli(params)
        else:
            samples = torch.round(params)
        # Scale back to [-1, 1]
        samples = samples * 2 - 1
        return samples
        

class CategoricalLikelihood(Likelihood):
    """
        Categorical Distribution
    """
    def __init__(self, param_dim: int):
        super(CategoricalLikelihood, self).__init__(param_dim=param_dim, type='categorical')


    def log_prob(self, data, logits, params=False, **kwargs):
        """
        Compute the log probability of data under Categorical distribution
        Args:
            data (torch.Tensor): Input data, should be of shape [batch*K, data_dim, *data_shape]
            logits (torch.Tensor): Output logits from the decoder, should be of shape [batch, K,data_dim*param_channel, *data_shape]
            params (bool): Whether the logits are already converted to parameters
        Returns:
            logp (torch.Tensor): Log probability of the data under the Categorical distribution.
        """
        # Data have to be in {0, ..., num_classes-1}
        bK, _, *input_shape, = logits.shape
        new_shape = [bK, self.param_dim, -1,] + list(input_shape)

        logits = logits.view(new_shape)  # (bs, K, dim*param_channel
        # Move the param channel to the end for cross entropy
        logp = -F.cross_entropy(logits, data.long(), reduction='none')
        return logp
    
    def logits_to_params(self, logits, **kwargs):
        """
        Transform logits to Categorical parameters
        Args:
            logits (torch.Tensor): Output logits from the decoder, should be of shape [batch,
            K, param_channel, *data_shape]
        Returns:
            params (torch.Tensor): Parameters of the Categorical distribution.
        """
        
        params = torch.softmax(logits, dim=-1)
        return params
    
    def logits_to_data(self, logits, sample=True, **kwargs):
        """
        Transform logits to data samples
        Args:
            logits (torch.Tensor): Output logits from the decoder, should be of shape [batch,
            K, param_channel, *data_shape]
            sample (bool): Whether to sample from the distribution or return the mode.
        Returns:
            samples (torch.Tensor): Sampled data from the Categorical distribution.
        """
        probs = self.logits_to_params(logits)
        data = self.params_to_data(probs, sample=sample)
        return data
    
    def params_to_data(self, params, sample=True, **kwargs):
        """
        Use parameters obtain data samples
        Args:
            params (torch.Tensor): Parameters of the Categorical distribution.
            sample (bool): Whether to sample from the distribution or return the mode.
        Returns:
            samples (torch.Tensor): Sampled data from the Categorical distribution.
        """
        if sample:
            samples = torch.distributions.categorical.Categorical(probs=params).sample()
        else:
            samples = torch.argmax(params, dim=-1)
        return samples


class GaussianLikelihood(Likelihood):
    """
        Gaussian Distribution 
    """
    def __init__(self,
                param_dim: int=None,
                min_logvar: float=None,
                learn_var: bool=False,
                ):
        super(GaussianLikelihood, self).__init__(param_dim=param_dim, type='gaussian')
        self.min_logvar = min_logvar
        self.learn_var = learn_var
        if self.learn_var :
            assert self.param_dim is not None and self.param_dim ==2, "For learn_var=True, param_dim must allow to have mean and logvar"

    def logits_to_params(self, logits, temperature=1., *args, **kwargs):
        """
        Transform logits to Gaussian parameters 
        Args:
            logits (torch.Tensor): Output logits from the decoder, should be of shape [batch, K, param_channel, *data_shape]
            temperature (float): Temperature scaling for the variance.
        Returns:
            mean (torch.Tensor): Mean of the Gaussian distribution.
            logvar (torch.Tensor): Log variance of the Gaussian distribution.
        """
        if self.learn_var:
            mean, logvar = logits.chunk(2, dim=2) # Split channels between mean and logvar
            mean = mean.flatten(1,2)
            logvar = logvar.flatten(1,2)
            logvar = torch.clamp(logvar, min=self.min_logvar)
        else:
            mean = logits
            logvar = torch.ones_like(mean) * self.min_logvar
        logvar = logvar + np.log(temperature)
        return mean, logvar

    def logits_to_data(self, logits, temperature=1., sample=True, *args, **kwargs):
        """
        Transform logits to data samples
        Args:
            logits (torch.Tensor): Output logits from the decoder, should be of shape [batch, K, param_channel, *data_shape]
            temperature (float): Temperature scaling for the variance.
            sample (bool): Whether to sample from the distribution or return the mean.
        Returns:
            samples (torch.Tensor): Sampled data from the Gaussian distribution.
        """
        mean, logvar = self.logits_to_params(logits, temperature=temperature)
        if sample:
            samples = reparam(mean, logvar)
        else:
            samples = mean
        return samples
   
    def log_prob(self, data, logits, *args, **kwargs):
        """
        Compute the log probability of data under Gaussian distribution
        Args:
            data (torch.Tensor): Input data, should be of shape [batch, dim, *data_shape]
            logits (torch.Tensor): Output logits from the decoder, should be of shape [batch, K, param_channel, *data_shape]
        Returns:
            logp (torch.Tensor): Log probability of the data under the Gaussian distribution.
        """
        mean, logvar = self.logits_to_params(logits)
        logp = gaussian_log_prob(data, mean, logvar)
        return logp 
    
