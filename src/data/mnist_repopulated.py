import numpy as np
import torch
from torchvision.datasets import MNIST


class MNISTRepopulated(MNIST):
    """
    MNIST dataset with repopulated digits according to specified proportions.
    Digits with proportion 0 are dropped from the dataset.
    0 <= proportions[digit] <= 1 for all digits 0-9
    """

    def __init__(
        self,
        *args,
        proportions={
            0: 0.12,
            1: 0.33,
            2: 0.28,
            3: 0.20,
            4: 0.07,
            5: 0.0,
            6: 0.0,
            7: 0.0,
            8: 0.0,
            9: 0.0,
        },
        dataset_fraction=1.0,
        **kwargs,
    ):
        """
        Args:
            proportions: Dictionary specifying the fraction of each digit to keep in the dataset.
        """
        super().__init__(*args, **kwargs)
        self.dataset_fraction = dataset_fraction
        self.proportions = proportions
        self.drop_digits = [k for k, v in proportions.items() if v == 0]

        # Check that proportions are valid
        assert all(
            0 <= p <= 1 for p in proportions.values()
        ), "Proportions must be between 0 and 1"
        assert all(
            k in range(10) for k in proportions.keys()
        ), "Proportions keys must be digits 0-9"

        # Drop digits with proportion 0
        if len(self.drop_digits) > 0:
            mask = ~np.isin(self.targets, self.drop_digits)
            self.data = self.data[mask]
            self.targets = self.targets[mask]

        # Repopulate dataset according to proportions
        new_data = []
        new_targets = []
        for digit, prop in proportions.items():
            if prop > 0:
                digit_mask = self.targets == digit
                digit_data = self.data[digit_mask]
                digit_targets = self.targets[digit_mask]
                n_samples = int(len(digit_data) * prop)
                if n_samples > 0:
                    indices = np.random.choice(len(digit_data), n_samples, replace=True)
                    new_data.append(digit_data[indices])
                    new_targets.append(digit_targets[indices])
        self.data = torch.cat(new_data, dim=0)
        self.targets = torch.cat(new_targets, dim=0)

    def __len__(self):
        return int(self.dataset_fraction * len(self.data))

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        if label in self.drop_digits:
            raise ValueError(f"Digit {label} has been dropped from the dataset")
        mask = torch.ones_like(img, dtype=torch.float32)

        return img, mask
