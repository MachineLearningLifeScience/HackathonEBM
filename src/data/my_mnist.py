import torch
from torchvision.datasets import MNIST
import numpy as np

class MyMNIST(MNIST):
    """
    MNIST dataset with a challenging MNAR mechanism:
    - Missingness depends on label, pixel intensity, and spatial location.
    - For digits 3 and 8, central high-intensity pixels are more likely to be missing.
    """
    def __init__(self, *args, missing_rate=0., threshold=255, center_box=10, digits_box = [], drop_digits=None, dataset_fraction=1., **kwargs):
        """ 
        Args:
            missing_rate: Base probability of a pixel being missing if its intensity > threshold.
            threshold: Pixel intensity threshold for missingness.
            center_box: Size of the central square region where missingness is more likely for digits
            digits_box: List of digit labels (e.g., [3,8]) that have special missingness in the center.
            drop_digits: List of digit labels to drop from the dataset (e.g., [0,1,2])
        """
        super().__init__(*args, **kwargs)
        self.missing_rate = missing_rate
        self.intensity_threshold = threshold
        self.center_box = center_box
        self.digits_box = digits_box 
        self.dataset_fraction = dataset_fraction

        # Drop specified digits
        if drop_digits is not None:
            mask = ~np.isin(self.targets, drop_digits)
            self.data = self.data[mask]
            self.targets = self.targets[mask]

        # Precompute masks for all samples
        self.masks = []
        for idx in range(len(self.data)):
            img_np = self.data[idx].numpy().astype(np.float32)
            label = int(self.targets[idx])
            mask = np.ones_like(img_np, dtype=np.float32)

            # Central region mask
            center = img_np.shape[0] // 2
            half_box = self.center_box // 2
            center_mask = np.zeros_like(img_np, dtype=bool)
            center_mask[
                center - half_box:center + half_box,
                center - half_box:center + half_box
            ] = True

            # Base missingness: high intensity pixels
            high_intensity = img_np > self.intensity_threshold
            prob_mask = np.random.rand(*img_np.shape)

            # For digits 3 and 8, central high-intensity pixels are much more likely to be missing
            if label in digits_box:
                # Increase missing rate in the center region for high-intensity pixels
                mask[center_mask & high_intensity & (prob_mask < self.missing_rate * 1.5)] = 0.0
                # Also, some missingness outside center
                mask[~center_mask & high_intensity & (prob_mask < self.missing_rate * 0.5)] = 0.0
            else:
                # For other digits, missingness is less biased
                mask[high_intensity & (prob_mask < self.missing_rate)] = 0.0

            self.masks.append(mask)
        self.masks = np.stack(self.masks)

    def __len__(self):
        return int(len(self.data) * self.dataset_fraction)
    
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        img_np = np.array(img, dtype=np.float32)
        mask = self.masks[index]

        if img_np.shape != mask.shape:
            img_np = img_np.squeeze()

        img_np[mask == 0] = 0  # Set missing pixels to 0

        # Ensure the image has shape (1, 28, 28)
        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, axis=0)
        img_tensor = torch.from_numpy(img_np)

        # Also ensure mask has shape (1, 28, 28)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)

        return img_tensor, mask

