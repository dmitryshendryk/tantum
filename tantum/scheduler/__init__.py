

import torch
from torch.optim.lr_scheduler import (CosineAnnealingWarmRestarts, 
                                     CosineAnnealingLR, ReduceLROnPlateau, 
                                     OneCycleLR)

# from tantum.scheduler.gradual_warmup_v2 import GradualWarmupSchedulerV2

def get_scheduler(cfg, optimizer=None):
    if cfg.scheduler=='ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau
    elif cfg.scheduler=='CosineAnnealingLR':
        scheduler = CosineAnnealingLR
    elif cfg.scheduler=='CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts
    elif cfg.scheduler == 'OneCycleLR':
        scheduler = OneCycleLR
    # elif cfg.scheduler=='GradualWarmupSchedulerV2':
        # scheduler_cosine=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, cfg['cosine_epo'])
        # scheduler_warmup=GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=cfg['warmup_epo'], after_scheduler=scheduler_cosine)
        # scheduler=scheduler_warmup
    return scheduler