import numpy as np
import torch
from torch.utils.data import Dataset
from torch.distributions import MultivariateNormal, Categorical

class MoGDataset(Dataset):
    """
    Synthetic 2D Mixture of Gaussians dataset with log_prob() support.
    """

    def __init__(self, n_samples=1000, means=None, covariances=None, weights=None, seed=42, device="cpu", *args, **kwargs):
        super().__init__()
        np.random.seed(seed)

        self._set_params(means, covariances, weights)

        # Sample from mixture
        self.data = self._sample_mog(n_samples)
        self.device = device

        # Create torch distributions
        self.components = [
            MultivariateNormal(torch.tensor(mu, dtype=torch.float32),
                               torch.tensor(Sigma, dtype=torch.float32))
            for mu, Sigma in zip(self.means, self.covariances)
        ]
        self.cat = Categorical(torch.tensor(self.weights, dtype=torch.float32))

    def _set_params(self, means, covariances, weights):
        # Default parameters (3 components)
        self.means = np.array([
                [-1.0, -0.33],
                [ 0.0,  0.67],
                [ 1.0, -0.33],
            ]) if means is None else np.array(means)
        self.covariances = np.array([
            [[0.04, 0.02], [0.02, 0.04]],
            [[0.04, -0.012], [-0.012, 0.04]],
            [[0.04, 0.0], [0.0, 0.04]],
            ]) if covariances is None else np.array(covariances)
        self.weights = np.array([0.34, 0.33, 0.33]) if weights is None else np.array(weights)

        # Normalize weights just in case
        self.weights = self.weights / self.weights.sum()

    def _sample_mog(self, n_samples):
        component_choices = np.random.choice(len(self.weights), size=n_samples, p=self.weights)
        samples = []
        for i in range(len(self.weights)):
            n_i = np.sum(component_choices == i)
            if n_i > 0:
                samples_i = np.random.multivariate_normal(self.means[i], self.covariances[i], n_i)
                samples.append(samples_i)
        samples = np.vstack(samples).astype(np.float32)

        # Shuffle samples
        np.random.shuffle(samples)

        # Add two empty dimensions
        samples = samples[..., None, None]
        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

    @torch.no_grad()
    def log_prob(self, x):
        """
        Compute log p(x) under the mixture model.
        x: Tensor of shape (N, 2)
        """
        x = x.to(torch.float32)
        log_probs = torch.stack([comp.log_prob(x) for comp in self.components], dim=1)  # [N, K]
        weighted = log_probs + torch.log(self.cat.probs)[None, :]                       # log w_k + log p_k(x)
        return torch.logsumexp(weighted, dim=1)                                         # logsumexp over components
    
    @torch.no_grad()
    def responsibilities(self, x):
        """
        Compute responsibilities p(z=k | x) for each component k.
        x: Tensor of shape (N, 2)
        Returns: Tensor of shape (N, K)
        """
        x = x.to(torch.float32)
        log_probs = torch.stack([comp.log_prob(x) for comp in self.components], dim=1)  # [N, K]
        weighted = log_probs + torch.log(self.cat.probs)[None, :]                       # log w_k + log p_k(x)
        log_resps = weighted - torch.logsumexp(weighted, dim=1, keepdim=True)          # log p(z=k|x)
        return torch.exp(log_resps)                                                     # p(z=k|x)      
    
    @torch.no_grad()
    def get_components(self, x):
        """
        Returns the predicted mixture component from the responsibilities.
        """
        r = self.responsibilities(x)
        return r.argmax(dim=1)