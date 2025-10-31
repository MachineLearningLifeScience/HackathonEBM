from hydra.utils import instantiate


def get_callbacks(config, loaders):
    splits = {"train": 0, "val": 1, "test": 2}
    exc_callbacks = [
        "lr_monitor",
        "model_ckpt",
        "image_samples",
        "buffer_samples",
        "classify_samples",
    ]
    callbacks = []
    if not hasattr(config, "callbacks"):
        return []
    for name, callback_cfg in config.callbacks.items():
        split = callback_cfg.get("split", None)
        loader = loaders[splits[split]] if split in splits else None
        if name in exc_callbacks:
            callbacks.append(instantiate(callback_cfg))
        else:
            callbacks.append(instantiate(callback_cfg, dataloader=loader))
    return callbacks
