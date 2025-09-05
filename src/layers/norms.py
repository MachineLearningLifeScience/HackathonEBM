import torch.nn as nn

norms = {
    "layer_norm": nn.LayerNorm,
    "batch_norm": nn.BatchNorm2d,
    "instance_norm": nn.InstanceNorm2d,
    "group_norm": nn.GroupNorm,
    "spectral_norm": nn.utils.spectral_norm,
    "none": nn.Identity,
}