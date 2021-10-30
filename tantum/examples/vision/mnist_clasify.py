import argparse
from operator import imod
import os
import sys

PROJECT_ROOT = '/Users/dmitry/Documents/Personal/study/kaggle/tantum'
sys.path.append(PROJECT_ROOT)

import glob
import albumentations
import pandas as pd
import tantum
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from sklearn import metrics, model_selection, preprocessing

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

from tantum.callbacks import EarlyStopping, OutOfFold, WeightAndBiasesCallback, Checkpoint
from tantum.datasets.datasets import ImageDataset
from tantum.trainer.v1.base_fitter import Model
from torch.nn import functional as F
from tantum.model.simple_net import Net
from tantum.datasets.datasets import create_folds, MnistDataset

INPUT_PATH = "../../../input"
MODEL_PATH = "../models/"
MODEL_NAME = os.path.basename(__file__)[:-3]
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
IMAGE_SIZE = 192
EPOCHS = 20

class Config:
    target_size=10
    


class MnistModel(Model):
    def __init__(self):
        super().__init__()
        self.model = Net()
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-4)

    def monitor_metrics(self, outputs, targets):
        outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(targets, outputs)
        return {"accuracy": accuracy}

    def fetch_scheduler(self):
        scheduler = CosineAnnealingWarmRestarts(self.opt, T_0=10, T_mult=1, eta_min=1e-4, last_epoch=-1)
        return scheduler

    def fetch_optimizer(self):
        # opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        return self.opt

    def forward(self, image, targets=None):
        
        outputs = self.model(image)

        if targets is not None:
            loss = nn.CrossEntropyLoss()(outputs, targets)
            metrics = self.monitor_metrics(outputs, targets)
            return outputs, loss, metrics
        return outputs, 0, {}


if __name__ == "__main__":


    ## Reading Data
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'train.csv') )
    ## Create Folds
    folds = create_folds(df, 5)

    ## Callbacks 

    oof_callback = OutOfFold(output_path='./')
    
    es_callback = EarlyStopping(
        monitor="valid_loss",
        model_path=os.path.join('./', 'mnist' + ".pt"),
        patience=3,
        mode="min",
    )

    chc_callback = Checkpoint(save_path='./', file_name='mnist_checkpoint')

    wb_callback = WeightAndBiasesCallback(project='mnist')

    for fold in range(5):

        model = MnistModel()

        trn_idx = folds[folds['fold'] != fold].index
        val_idx = folds[folds['fold'] == fold].index

        train_folds = folds.loc[trn_idx].reset_index(drop=True)
        valid_folds = folds.loc[val_idx].reset_index(drop=True)

        y = train_folds['label'].values
        X = train_folds.drop(['label','fold'], axis=1).values

        y_val = valid_folds['label'].values
        X_val = valid_folds.drop(['label','fold'], axis=1).values

        
        valid_labels = valid_folds['label'].values
        
        train_dataset = MnistDataset(
            X = X,
            y = y,
        )

        valid_dataset = MnistDataset(
            X = X_val,
            y = y_val,
        )

        model.fit(
            train_dataset,
            valid_dataset=valid_dataset,
            train_bs=TRAIN_BATCH_SIZE,
            valid_bs=VALID_BATCH_SIZE,
            device="cpu",
            epochs=EPOCHS,
            callbacks=[es_callback, oof_callback, chc_callback, wb_callback],
            fp16=False,
            n_jobs=0,
            valid_labels=valid_labels,
            valid_folds=valid_folds,
            target_size=10,
            fold=fold
        )