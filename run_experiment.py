import argparse
import os
from datetime import datetime

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.callbacks.callbacks import get_callbacks
from src.data.utils import get_data_loaders
from src.utils import load_model

# Create parser only for --exp_config
parser = argparse.ArgumentParser()
parser.add_argument("--exp_config", type=str, required=True)
parser.add_argument("--ckpt_path", type=str, default=None)

# Normally parse args
args = parser.parse_args()


exp_config = omegaconf.OmegaConf.load(args.exp_config)
ckpt_path = args.ckpt_path


def main():

    # Load model from checkpoint
    model, config = load_model(ckpt_path=ckpt_path)
    if torch.cuda.is_available():
        model.cuda()

    # Data loaders
    loaders = get_data_loaders(config)  # [train_loader, val_loader, ?test_loader]

    log_dir = os.path.dirname(os.path.dirname(ckpt_path))

    experiment = instantiate(exp_config)
    experiment.setup(log_dir=log_dir)

    split = exp_config.cfg.get("split", "test")
    if split == "train":
        loader = loaders[0]  # Assuming the first loader is training
    elif split == "val":
        # If a test loader is provided, run the experiment on it
        loader = loaders[1]
    elif split == "test":
        loader = loaders[2]

    # If no test loader is provided, run the experiment on the validation loader
    if loader is None:
        loader = loaders[1]  # Assuming the second loader is validation

    experiment.run(model, loader=loader)


if __name__ == "__main__":
    main()
