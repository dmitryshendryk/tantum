import numpy as np
import os
import torch

import wandb

from tantum import enums
from tantum.callbacks import Callback



class WeightAndBiasesCallback(Callback):

    def __init__(self, project) -> None:
        isLogged = wandb.login()

        if isLogged:
            print('Logged to Weights & Biases')
            wandb.init(project=project, entity="maverix")
    
    def on_valid_epoch_end(self, model, **kwargs):
        print("Weights and Biases Call")
        wandb.log(model.metrics)
        wandb.watch(model)
    
    def log(self, logs):
        wandb.log(logs)
