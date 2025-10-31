import torch
import random
import numpy as np

class SGLD(torch.nn.Module):
    def __init__(self, 
                 input_dim,
                 input_shape, 
                 sample_size, 
                 steps, 
                 step_size, 
                 log_prob,
                 noise_std=1.0,
                 clip_grads=False,
                 clamp=False
                 ):
        """
        Stochastic Gradient Langevin Dynamics (SGLD) sampler.

        Args:
            input_dim: dimensionality of each sample
            input_shape: spatial shape (if applicable)
            sample_size: number of parallel samples to draw
            steps: number of Langevin steps
            step_size: step size η
            log_prob: function returning log p(x) (can be unnormalized)
            noise_std: std multiplier for injected noise
        """
        super().__init__()
        self.input_dim = input_dim
        self.input_shape = list(input_shape)
        self.sample_size = sample_size
        self.steps = steps
        self.step_size = step_size 
        self.noise_std = noise_std
        self.clip_grads = clip_grads
        self.clamp = clamp
        self.log_prob = log_prob


    def __call__(self, num_samples=None, init=None, steps=None, step_size=None, return_steps=False, *args, **kwargs):
        if num_samples is None:
            num_samples = self.sample_size
        if init is None:
            init = torch.rand(num_samples, self.input_dim, *self.input_shape) * 2 - 1
        if steps is None:
            steps = self.steps
        if step_size is None:
            step_size = self.step_size

        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        x = init.clone().detach().requires_grad_(True)
        if self.clamp:                        
            with torch.no_grad():
                x.clamp_(min=-1.0, max=1.0)
            
        per_step = []
        step_sqrt = torch.sqrt(torch.tensor(step_size, device=x.device, dtype=x.dtype))

        for _ in range(steps):
            # 1. Compute ∇ log p(x)
            log_prob = self.log_prob(x)
            log_prob.sum().backward()

            with torch.no_grad():
                if self.clip_grads:
                    x.grad.clamp_(-0.03, 0.03)
                    
                # 2. Langevin drift
                x.add_(0.5 * step_size * x.grad)

                # 3. Add noise
                noise = torch.randn_like(x) * self.noise_std * step_sqrt
                x.add_(noise)

                # 4. Clamp if needed
                if self.clamp:
                    x.clamp_(min=-1.0, max=1.0)

            x.grad.zero_()

            if return_steps:
                per_step.append(x.clone().detach())

        torch.set_grad_enabled(had_gradients_enabled)

        if return_steps:
            return torch.stack(per_step, dim=1)
        else:
            return x
