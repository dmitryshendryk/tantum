import os
import torch
from torch import optim




def create_optimizer(params, config):
    if config.optim == 'sgd':
        print(params, config.lr)
        optimizer = optim.SGD(params,
                              config.lr,
                              momentum=config.momentum,
                              weight_decay=config.weight_decay,
                              nesterov=config.nesterov)
    
    elif config.optim == 'adam':
        optimizer = optim.Adam(params, config.lr)
    
    return optimizer