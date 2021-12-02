import argparse
from operator import imod
import os
import sys
import numpy as np
from numpy.lib.nanfunctions import _nancumprod_dispatcher
from torchvision.datasets import STL10
# from warmup_scheduler import scheduler
from torch.utils.data import Dataset
import cv2


## https://towardsdatascience.com/attention-in-computer-vision-fd289a5bd7ad

sys.path.append('../input/tantum')
sys.path.append('../input/timm-pytorch-image-models/pytorch-image-models-master')

import glob
import albumentations
import pandas as pd
import tantum
import torch
import torch.nn as nn
from sklearn import metrics, model_selection, preprocessing
from torchvision.datasets import STL10
import matplotlib.pyplot as plt
import IPython.display as display
import time

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau


from tantum.callbacks import EarlyStopping, OutOfFold, WeightAndBiasesCallback, Checkpoint, PlotErrAcc
from tantum.datasets.datasets import ImageDataset
from tantum.trainer.v1.base_fitter import Model
from torch.nn import functional as F
from tantum.model.simple_net import Net
from tantum.datasets.datasets import create_folds, MnistDataset
from tantum.enums import ModelState, TrainingState

from tantum.datasets.augmentations import cutmix, mixup, fmix
from tantum.model.attention.multihead import MultiHeadAttention

from tantum.model import get_model
from tantum.optimizer import get_optimizer
from tantum.loss import get_criterion
from tantum.scheduler import get_scheduler

INPUT_PATH = "../../../input"
MODEL_PATH = "../models/"
# MODEL_NAME = os.path.basename(__file__)[:-3]
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
    use_cutmix=False



class STL10_Dataset_train(Dataset):
    def __init__(self):
        self.dataset_train = STL10("../input/stl10-binary-files", split='train', download=True)

    def __len__(self):
        return len(self.dataset_train)
    
    def __getitem__(self, index):
        image, target = self.dataset_train[index]
        image = (np.array(image)-128.0)/128.0
        image = image.transpose(2, 0, 1)
        
        np.random.seed(index + 10)
        corrupt = np.random.randint(2)
        if corrupt:  # To corrupt the image, we'll just copy a patch from somewhere else
            pos_x = np.random.randint(96-16)
            pos_y = np.random.randint(96-16)
            image[:, pos_x:pos_x+16, pos_y:pos_y+16] = 1
        
        return {
            "image": image,
            "targets": torch.tensor(corrupt, dtype=torch.long),
        }



class STL10_Dataset_test(Dataset):
    
    def __init__(self):
        self.dataset_test = STL10("../input/stl10-binary-files", split='test', download=True)
    
    def __len__(self):
        return len(self.dataset_test)
    
    def __getitem__(self, index):
        image, target = self.dataset_test[index]
        image = (np.array(image)-128.0)/128.0
        image = image.transpose(2, 0, 1)
        
        np.random.seed(index + 10)
        corrupt = np.random.randint(2)
        if corrupt:  # To corrupt the image, we'll just copy a patch from somewhere else
            pos_x = np.random.randint(96-16)
            pos_y = np.random.randint(96-16)
            image[:, pos_x:pos_x+16, pos_y:pos_y+16] = 1
        
        return {
            "image": image,
            "targets": torch.tensor(corrupt, dtype=torch.long),
        }



class ConvPart(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1a = nn.Conv2d(3, 32, 5, padding=2)
        self.p1 = nn.MaxPool2d(2)
        self.c2a = nn.Conv2d(32, 32, 5, padding=2)
        self.p2 = nn.MaxPool2d(2)
        self.c3 = nn.Conv2d(32, 32, 5, padding=2)
        self.bn1a = nn.BatchNorm2d(32)
        self.bn2a = nn.BatchNorm2d(32)

    def forward(self, x):
        z = self.bn1a(F.leaky_relu(self.c1a(x)))
        z = self.p1(z)
        z = self.bn2a(F.leaky_relu(self.c2a(z)))
        z = self.p2(z)
        z = self.c3(z)
        return z


class NetMultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvPart()
        self.attn1 = MultiHeadAttention(mem_in=32, query_in=32)
        self.final = nn.Linear(32, 1)
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        z = self.conv(x)
        q = torch.reshape(z, (z.size(0) , z.size(1), -1))
        q, w = self.attn1(q, q)
        q = torch.reshape(q, (z.size(0), z.size(1), z.size(2), z.size(3)))
        z = q.mean(3).mean(2)
        p = torch.sigmoid(self.final(z))[:, 0]
        return p, q



class STL10Model(Model):
    def __init__(self):
        super().__init__()
        # self.model = get_model(Config)()
        self.model = NetMultiheadAttention()
        self.opt = get_optimizer(Config)(self.parameters(), lr=1e-4)
#         self.loss_fn = get_criterion(Config)()
        self.loss_fn = nn.BCELoss()
        
        self.input = None
        self.attention = None 
        
    
    def monitor_metrics(self, outputs, targets):
        outputs = torch.round(outputs)
        accuracy = torch.sum(outputs == targets).cpu().detach().item()/VALID_BATCH_SIZE
        return {"accuracy": accuracy}

    def fetch_scheduler(self, train_loader):
        scheduler = get_scheduler(Config)(self.opt, **Config.scheduler_params)
        return scheduler

    def fetch_optimizer(self):
        return self.opt

    def forward(self, image, targets=None):

        mix_decision = np.random.rand()
        image = image.float()

        targets = targets.float()

        if Config.use_cutmix and mix_decision < 0.25:
            image, image_labels = cutmix(image, targets, 1.)
        # elif mix_decision >=0.25 and mix_decision < 0.5:
            # image, image_labels = fmix(image, targets, alpha=1., decay_power=5., shape=(28,28))
        
        outputs, att = self.model(image.float())
        self.input = image
        self.attention = att
        
        
        if targets is not None:
            
            if Config.use_cutmix and mix_decision < 0.25:
                loss = self.loss_fn(outputs, image_labels[0]) * image_labels[2] + self.loss_fn(outputs, image_labels[1]) * (1. - image_labels[2])
                metrics = self.monitor_metrics(outputs, image_labels[0])
                return outputs, loss, metrics
            else:
                loss = self.loss_fn(outputs, targets)
#                 loss = -torch.mean(targets*torch.log(outputs+1e-8) + (1-targets)*torch.log(1-outputs+1e-8))
                metrics = self.monitor_metrics(outputs, targets)
            
            
                
            return outputs, loss, metrics
        return outputs, 0, {}

if __name__ == "__main__":
    
    model = STL10Model()

    ## Callbacks 

    oof_callback = OutOfFold(output_path='./')
    
    es_callback = EarlyStopping(
        monitor="valid_loss",
        model_path=os.path.join('./', 'mnist' + ".pt"),
        patience=3,
        mode="min",
    )

    chc_callback = Checkpoint(save_path='./', file_name='mnist_checkpoint')
    plot_callback = PlotErrAcc(attention_show=True)

    # wb_callback = WeightAndBiasesCallback(project='mnist')
    train_dataset = STL10_Dataset_train()

    valid_dataset = STL10_Dataset_test()

    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        train_bs=TRAIN_BATCH_SIZE,
        valid_bs=VALID_BATCH_SIZE,
        device="cuda",
        epochs=EPOCHS,
        callbacks=[plot_callback],
        fp16=False,
        n_jobs=0,
#         valid_labels=valid_labels,
#         valid_folds=valid_folds,
        target_size=1,
        fold=0
    )
