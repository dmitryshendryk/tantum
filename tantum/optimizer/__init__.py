

from torch import optim

def get_optimizer(config):
    if config.optim == 'SGD':
        optimizer = optim.SGD
    elif config.optim == 'Adam':
        optimizer = optim.Adam
    elif config.optim == 'AdamW':
        optimizer = optim.AdamW
    
    return optimizer