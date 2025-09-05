
from hydra.utils import instantiate
from omegaconf import OmegaConf

def get_data_loaders(cfg):

    train_cfg = {**cfg.data.dataset, **cfg.data.train_dataset}
    train_dataset = instantiate(train_cfg)
    train_loader = instantiate(cfg.data.dataloader, dataset=train_dataset, batch_size=cfg.train.batch_size, shuffle=True)

    if 'val_dataset' in cfg.data:
        val_cfg = {**cfg.data.dataset, **cfg.data.val_dataset}
        val_dataset = instantiate(val_cfg)
        val_loader = instantiate(cfg.data.dataloader, dataset=val_dataset, batch_size=cfg.train.batch_size, shuffle=False)
    else:
        val_loader = None

    if 'test_dataset' in cfg.data:
        test_cfg = {**cfg.data.dataset, **cfg.data.test_dataset}
        test_dataset = instantiate(test_cfg)
        test_loader = instantiate(cfg.data.dataloader, dataset=test_dataset, batch_size=cfg.train.batch_size, shuffle=False)
    else:
        test_loader = None

    return train_loader, val_loader, test_loader
