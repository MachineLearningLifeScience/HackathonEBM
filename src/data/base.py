import numpy as np
import torch
from torch.utils.data import Dataset

class MNARDataset(Dataset):
    def __init__(self, data, mnar_prob=0.2, mechanism='self_mask', seed=None):
        """
        data: numpy array or torch tensor, shape (n_samples, n_features)
        mnar_prob: probability of missingness
        mechanism: 'self_mask' or 'feature_mask'
        seed: random seed for reproducibility
        """
        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data).float()
        else:
            self.data = data.float()
        self.mnar_prob = mnar_prob
        self.mechanism = mechanism
        self.rng = np.random.RandomState(seed)
        self.masks = self._generate_mnar_mask()

    def _generate_mnar_mask(self):
        data_np = self.data.numpy()
        n_samples, n_features = data_np.shape
        mask = np.ones_like(data_np, dtype=np.float32)
        if self.mechanism == 'self_mask':
            # Probability of missingness increases with feature value
            for j in range(n_features):
                probs = self.mnar_prob * (data_np[:, j] - data_np[:, j].min()) / (data_np[:, j].ptp() + 1e-8)
                mask[:, j] = self.rng.binomial(1, 1 - probs)
        elif self.mechanism == 'feature_mask':
            # Randomly select features to be MNAR
            for i in range(n_samples):
                probs = self.mnar_prob * (data_np[i] - data_np[i].min()) / (data_np[i].ptp() + 1e-8)
                mask[i] = self.rng.binomial(1, 1 - probs)
        else:
            raise ValueError("Unknown mechanism: {}".format(self.mechanism))
        return torch.from_numpy(mask)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        mask = self.masks[idx]
        return x, mask