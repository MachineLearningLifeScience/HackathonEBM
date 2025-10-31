import torch.nn as nn
from hydra.utils import instantiate

from src.layers.acts import activations
from src.layers.norms import norms


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        activation="leaky_relu",
        norm="none",
        spectral_norm=False,
        dropout=0.0,
        output_shape=None
    ):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_shape = output_shape
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            linear = nn.Linear(prev_dim, h_dim)
            if spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.append(linear)
            if norm != "none":
                layers.append(norms[norm](h_dim))
            layers.append(activations[activation]())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        original_shape = x.shape
        x = x.view(x.size(0), -1)  # Flatten input if needed
        out = self.model(x)
        if self.output_shape is not None:
            out = out.view(original_shape[0], -1, *self.output_shape)
        return out
