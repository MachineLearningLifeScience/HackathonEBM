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
                 ):
        """
        Inputs:
            model - Neural network to use for modeling E_theta
            img_shape - Shape of the images to model
            sample_size - Batch size of the samples
            steps - Number of iterations in the MCMC algorithm
            step_size - Learning rate nu in the algorithm above
            u_log_prob - Function to compute the unnormalized log probability of the data
            max_len - Maximum number of data points to keep in the buffer
        """
        super().__init__()
        self.input_dim = input_dim
        self.img_shape = list(input_shape)
        self.sample_size = sample_size
        self.steps = steps
        self.step_size = step_size 
        self.noise_std = noise_std  # Standard deviation of the noise to add
        self.log_prob = log_prob  # Placeholder for the log probability function


    def __call__(self, num_samples=None, init=None, steps=None, step_size=None, return_img_per_step=False):
        """
        Function for sampling images.
        Inputs:
            init - Images to start from for sampling. If you want to generate new images, enter None
            steps - Number of iterations in the MCMC algorithm.
            step_size - Learning rate nu in the algorithm above
            return_img_per_step - If True, we return the sample at every iteration of the MCMC
        """
        if num_samples is None:
            num_samples = self.sample_size
        if init==None:
            # If no initial images are provided, generate random noise
            init = torch.rand(num_samples, self.input_dim, *self.input_shape) * 2 - 1
        if steps is None:
            steps = self.steps

        if step_size is None:
            step_size = self.step_size

        x = init.clone()
        x.requires_grad = True
        
        # Enable gradient calculation if not already the case
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        
        # We use a buffer tensor in which we generate noise each loop iteration.
        # More efficient than creating a new tensor every iteration.
        noise = torch.randn(x.shape, device=x.device)
        
        # List for storing generations at each step (for later analysis)
        imgs_per_step = []
        
        # Loop over K (steps)
        for _ in range(steps):
            # Part 1: Add noise to the input.
            noise.normal_(0, self.noise_std * step_size)
            x.data.add_(noise.data)
            x.data.clamp_(min=-1.0, max=1.0)
            
            # Part 2: calculate gradients for the current input.
            log_prob = self.log_prob(x)
            log_prob.sum().backward()
            x.grad.data.clamp_(-0.03, 0.03) # For stabilizing and preventing too high gradients

            # Apply gradients to our current samples
            x.data.add_(0.5*step_size**2 * x.grad.data)
            x.grad.detach_()
            x.grad.zero_()
            x.data.clamp_(min=-1.0, max=1.0)
            
            if return_img_per_step:
                imgs_per_step.append(x.clone().detach())
        
        # Reset gradient calculation to setting before this function
        torch.set_grad_enabled(had_gradients_enabled)

        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return x