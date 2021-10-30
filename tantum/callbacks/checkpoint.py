import numpy as np
import os
import torch

from tantum import enums
from tantum.callbacks import Callback




class Checkpoint(Callback):

    def __init__(self, save_path, file_name) -> None:
        self.save_path = save_path
        self.file_name = file_name
    

    def on_epoch_end(self, model, **kwargs):
        print("Save Checkpoint")

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': model.optimizer.state_dict() if model.optimizer else '',
            'scheduler': model.scheduler.state_dict() if model.scheduler else '' ,
            'scaler': model.scaler.state_dict() if model.scaler else '',
            'epoch': model.current_epoch,
        }

        torch.save(checkpoint, '%s_fold=%d_epoch=%d.pt' %
               (os.path.join(self.save_path, self.file_name), model.fold, model.current_epoch))


