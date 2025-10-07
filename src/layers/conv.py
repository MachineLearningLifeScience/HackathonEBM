import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from src.layers.acts import activations

class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_channels,
        input_shape,  # tuple (H, W)
        hidden_channels,
        kernel_sizes,
        strides,
        paddings,
        latent_dim,
        batch_norm=True,
        activation='relu',
        probabilistic=True
    ):
        """
        CNN Encoder for VAE.

        Args:
            input_channels (int): # input channels.
            input_size (tuple): (H, W) of input image.
            hidden_channels (list[int]): Output channels of each Conv layer.
            kernel_sizes (list[int]): Kernel sizes for each Conv layer.
            strides (list[int]): Strides for each Conv layer.
            paddings (list[int]): Paddings for each Conv layer.
            latent_dim (int): Dimensionality of latent vector.
        """
        super().__init__()

        assert len(hidden_channels) == len(kernel_sizes) == len(strides) == len(paddings), \
            "All Conv layer configs must be same length"

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        activation = activations[activation]

        layers = []
        in_channels = input_channels
        h, w = input_shape

        for out_channels, k, s, p in zip(hidden_channels, kernel_sizes, strides, paddings):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p))
            layers.append(activation())
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels

            # Update feature map size
            h = (h + 2 * p - k) // s + 1
            w = (w + 2 * p - k) // s + 1

        self.conv = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        flattened_size = in_channels * h * w

        self.fc_mean = nn.Linear(flattened_size, latent_dim)
        if probabilistic:
            self.fc_logvar = nn.Linear(flattened_size, latent_dim)

    def forward(self, x):
        h = self.flatten(self.conv(x))
        mu = self.fc_mean(h)
        if hasattr(self, 'fc_logvar'):
            logvar = self.fc_logvar(h)
            return mu, logvar
        return mu
    


class ConvDecoder(nn.Module):
    def __init__(
        self,
        output_channels,
        output_shape,  # tuple (H, W)
        hidden_channels,
        kernel_sizes,
        strides,
        paddings,
        latent_dim,
        batch_norm=True,
        activation='relu'
    ):
        """
        CNN Decoder for VAE.

        Args:
            output_channels (int): # of output image channels.
            output_size (tuple): (H, W) of the output image.
            hidden_channels (list[int]): Channels for each ConvTranspose layer.
            kernel_sizes (list[int]): Kernel sizes for ConvTranspose layers.
            strides (list[int]): Strides for ConvTranspose layers.
            paddings (list[int]): Paddings for ConvTranspose layers.
            latent_dim (int): Dimensionality of latent vector input.
        """
        super().__init__()
        assert len(hidden_channels) == len(kernel_sizes) == len(strides) == len(paddings), \
            "All ConvTranspose layer configs must be same length"

        self.output_shape = output_shape
        self.latent_dim = latent_dim
        activation = activations[activation]

        # Compute initial feature map size using data -> latent flow
        h, w = output_shape
        for hc, k, s, p in zip(reversed(hidden_channels), reversed(kernel_sizes), reversed(strides), reversed(paddings)):
            # Update feature map size
            h = (h + 2 * p - k) // s + 1
            w = (w + 2 * p - k) // s + 1
        in_channels = hidden_channels[0]
        flattened_size = in_channels * h * w

        self.init_h = h
        self.init_w = w
        self.init_channels = hidden_channels[0]

        self.fc = nn.Linear(latent_dim, flattened_size)

        layers = []
        in_channels = hidden_channels[0]
        for out_channels, k, s, p in zip(hidden_channels[:-1], kernel_sizes[:-1], strides[:-1], paddings[:-1]):
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p))
            layers.append(activation())
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels

        # Final output layer
        layers.append(nn.ConvTranspose2d(in_channels, output_channels, kernel_size=kernel_sizes[-1], stride=strides[-1], padding=paddings[-1]))

        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        h = self.fc(z)
        h = h.view(z.size(0), self.init_channels, self.init_h, self.init_w)
        x_recon = self.deconv(h)
        return x_recon


