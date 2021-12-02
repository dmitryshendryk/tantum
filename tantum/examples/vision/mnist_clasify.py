import argparse
from operator import imod
import os
import sys
import numpy as np

from warmup_scheduler import scheduler

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

from tantum.datasets.augmentations import cutmix, mixup, fmix

from tantum.model import get_model
from tantum.optimizer import get_optimizer
from tantum.loss import get_criterion
from tantum.scheduler import get_scheduler

INPUT_PATH = "../../../input"
MODEL_PATH = "../models/"
MODEL_NAME = os.path.basename(__file__)[:-3]
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
IMAGE_SIZE = 192
EPOCHS = 20

class Config:
    # target_size=10
    model_type='Net'
    optim='AdamW'
    criterion='CrossEntropyLoss'
    scheduler='CosineAnnealingWarmRestarts'
    
    
    scheduler_params={'T_0':10, "last_epoch":-1, 'eta_min':1e-4, "T_mult":1}
    model_params = {'target_size': 10}
    use_cutmix=True

class MnistModel(Model):
    def __init__(self):
        super().__init__()
        self.model = get_model(Config)()
        self.opt = get_optimizer(Config)(self.parameters(), lr=1e-4)
        self.loss_fn = get_criterion(Config)()

    def monitor_metrics(self, outputs, targets):
        outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(targets, outputs)
        return {"accuracy": accuracy}

    def fetch_scheduler(self, train_loader):
        scheduler = get_scheduler(Config)(self.opt, **Config.scheduler_params)
        return scheduler

    def fetch_optimizer(self):
        return self.opt

    def forward(self, image, targets=None):
        mix_decision = np.random.rand()
        image = image.float()
        targets = targets.long()
        if Config.use_cutmix and mix_decision < 0.25:
            image, image_labels = cutmix(image, targets, 1.)
        # elif mix_decision >=0.25 and mix_decision < 0.5:
            # image, image_labels = fmix(image, targets, alpha=1., decay_power=5., shape=(28,28))
            
        outputs = self.model(image.float())
        if targets is not None:
            
            if Config.use_cutmix and mix_decision < 0.25:
                loss = self.loss_fn(outputs, image_labels[0]) * image_labels[2] + self.loss_fn(outputs, image_labels[1]) * (1. - image_labels[2])
                metrics = self.monitor_metrics(outputs, image_labels[0])
                return outputs, loss, metrics
            else:
                
                loss = self.loss_fn(outputs, targets)
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

    # wb_callback = WeightAndBiasesCallback(project='mnist')

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
            callbacks=[],
            fp16=False,
            n_jobs=0,
            valid_labels=valid_labels,
            valid_folds=valid_folds,
            target_size=10,
            fold=fold
        )