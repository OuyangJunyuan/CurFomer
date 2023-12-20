

def build_optimizer(optim_cfg, model):
    import torch.optim as optim
    cfg = optim_cfg.copy()
    optimizer = getattr(optim, cfg.pop('type'))(params=model.parameters(), **cfg)
    return optimizer


def build_scheduler(lr_cfg, optimizer, **kwargs):
    import torch.optim.lr_scheduler as lr_sched
    cfg = lr_cfg.copy()
    for k, v in cfg.items():
        if isinstance(v, str) and v.startswith('$'):
            cfg[k] = kwargs.get(v[2:-1], None)
    lr_scheduler = getattr(lr_sched, cfg.pop('type'))(optimizer=optimizer, **cfg)
    return lr_scheduler
