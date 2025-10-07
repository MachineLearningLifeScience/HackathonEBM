import torch.nn as nn
import torch

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
activations = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'softmax': nn.Softmax,
    'softplus': nn.Softplus,
    'elu': nn.ELU,
    'selu': nn.SELU,
    'prelu': nn.PReLU,
    'gelu': nn.GELU,
    'swish': Swish,  # Swish is often implemented as SiLU in PyTorch
}



    