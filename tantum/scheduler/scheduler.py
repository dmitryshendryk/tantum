from tantum.utils.ramps import exp_warmup

from torch.optim import lr_scheduler

def create_lr_scheduler(optimizer, config):
    if config.lr_scheduler == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=config.epochs,
                                                   eta_min=config.min_lr)
    elif config.lr_scheduler == 'multistep':
        if config.steps is None: return None
        if isinstance(config.steps, int): config.steps = [config.steps]
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=config.steps,
                                             gamma=config.gamma)
    elif config.lr_scheduler == 'exp-warmup':
        lr_lambda = exp_warmup(config.rampup_length,
                               config.rampdown_length,
                               config.epochs)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif config.lr_scheduler == 'none':
        scheduler = None
    else:
        raise ValueError("No such scheduler: {}".format(config.lr_scheduler))
    return scheduler