import torch

import omegaconf
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from src.data.utils import get_data_loaders
from src.callbacks.callbacks import get_callbacks
from pytorch_lightning.callbacks import ModelCheckpoint

import os
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from src.utils import get_wandb_logger
from omegaconf import OmegaConf

@hydra.main(version_base=None, config_path="configs")
def main(config: omegaconf.DictConfig):

    # Logs and data directories
    if 'LOGDIR' in os.environ:
        log_dir = os.environ['LOGDIR']
        config.data.root = os.path.join(log_dir, 'data', config.data.root)
        config.train.log_dir = os.path.join(log_dir, 'logs', 'EBM_Hackathon')
    
    # Model
    model = instantiate(config.model, cfg=config)

    # Data loaders
    loaders = get_data_loaders(config)  #[train_loader, val_loader, ?test_loader]
    
    # Wandb Logger
    config_name = HydraConfig.get().job.config_name
    wandb_logger, log_dir = get_wandb_logger(config, config_name)

    # Callbacks
    callbacks = get_callbacks(config, loaders)

    # Add ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=1,
        save_last=True,
        monitor="val_loss",
        mode="min",
    )
    callbacks.append(checkpoint_callback)

    # Activate mixed precision
    if torch.cuda.is_available() and config.train.precision == 16:
        torch.set_float32_matmul_precision('medium')
        
    # Always resume from last.ckpt if resume is enabled
    ckpt_path = None
    if config.train.get("resume", False):
        ckpt_path = os.path.join(log_dir, "checkpoints", config.train.get("ckpt", "last.ckpt"))
        print(f"Resuming from checkpoint: {ckpt_path}")

    # Trainer
    trainer = Trainer(
        max_epochs=config.train.epochs,
        logger=wandb_logger,
        accelerator=config.train.accelerator,
        devices=config.train.devices if torch.cuda.is_available() else 1,
        strategy=instantiate(config.train.strategy) if '_target_' in config.train.strategy else config.train.strategy,
        precision="16-mixed" if torch.cuda.is_available() and config.train.precision == 16 else 32,
        default_root_dir=log_dir,
        callbacks=callbacks,   
        gradient_clip_val=config.train.gradient_clip_val,
    )

    trainer.fit(model, *loaders[:2], ckpt_path=ckpt_path)

    if config.data.get("test_dataset", False):
        trainer.test(model, dataloaders=loaders[-1])

if __name__ == "__main__":
    main()