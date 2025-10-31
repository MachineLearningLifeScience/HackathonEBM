import torch
import torch.nn as nn
from src.layers.mlp import MLP

class PNEncoder(nn.Module):
    """
    Implements the PN Encoder for handling missing data in a Partial VAE (PVAE) model.
    https://arxiv.org/pdf/1809.11142
    """
    def __init__(self,
                 input_dim,
                 emb_dim=10,
                 h_dim=20,
                 hidden_dims=[100, 50, 20],
                 latent_dim=20
                 ):
        super(PNEncoder, self).__init__()
        self.embs = nn.Parameter(torch.randn(input_dim, emb_dim))
        self.h = MLP(
            input_dim=emb_dim+1,
            hidden_dims=[],
            output_dim=h_dim,
        )

        self.encoder = MLP(
            input_dim=h_dim,
            hidden_dims=hidden_dims,
            output_dim=2*latent_dim,
        )
  
    def forward(self, x, mask):
        """
        x: [batch, input_dim]
        mask: [batch, input_dim]
        Returns:
            
        """
        # Enforce x to be (bs, input_dim) shape
        x = x.view(x.size(0), -1)
        mask = mask.view(mask.size(0), -1)

        # Expand input x to match embedding dimensions for broadcasting
        x_exp = x.unsqueeze(-1)                 # [B, D, 1]

        # Expand embeddings to match batch size for broadcasting
        e_exp = self.embs.unsqueeze(0)                  # [1, D, emb_dim]
        e_exp = e_exp.expand(x.size(0), -1, -1)  # [B, D, emb_dim]

        # Concatenate inputs to embeddings
        s = torch.cat((x_exp, e_exp), dim=-1)  # [B, D, emb_dim + 1]

        # Mask out unobserved values
        mask_exp = mask.unsqueeze(-1)           # [B, D, 1]
        s = s * mask_exp                        # [B, D, emb_dim]

        # Forward pass through the MLP
        h = self.h(s.view(-1, s.size(-1)))                           # [B*D, h_dim]

        # Sum across the input dimension
        h = h.view(x.size(0), -1, h.size(-1)).sum(dim=1)  # [B, h_dim]

        # Encode to mean and logvar
        mean_logvar = self.encoder(h)  # [B, 2*latent_dim]
        mean, logvar = torch.chunk(mean_logvar, 2, dim=-1)
        return mean, logvar






