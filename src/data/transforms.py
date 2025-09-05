

import torch

class Binarize:
    """Binarizes inputs in [0, 1] and rescales them to [-1, 1]."""
    def __call__(self, x):
        with torch.no_grad():
            binarized = (binarize_torch(x) - .5) * 2.
            return binarized
        
def binarize_torch(images):
    rand = torch.rand(images.shape, device=images.device)
    return (rand < images).float()