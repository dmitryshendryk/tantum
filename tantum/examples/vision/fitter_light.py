import torch
import numpy as np 
import time 
from datetime import datetime

import sys

PROJECT_ROOT = '/Users/dmitry/Documents/Personal/study/kaggle/tantum'
sys.path.append(PROJECT_ROOT)



import torch.nn as nn 
import torch.nn.functional as F 
import pytorch_lightning as pl

from torchvision.datasets import MNIST
from torchvision import transforms
from pytorch_lightning.metrics.functional import accuracy

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from tantum.trainer.lightining.fitter import FitterLightining
from tantum.model.simple_net import Net

class CFG:
    device = 'cpu'
    criterion = 'CrossEntropyLoss'
    optim = 'adam'
    lr=0.001


# data
dataset = MNIST('../../../data', train=True, download=True, transform=transforms.ToTensor())
mnist_train, mnist_val = random_split(dataset, [55000, 5000])

train_loader = DataLoader(mnist_train, batch_size=32)
val_loader = DataLoader(mnist_val, batch_size=32)

# model
model = FitterLightining(Net(), CFG)

# training
trainer = pl.Trainer(gradient_clip_val=100, stochastic_weight_avg=True)
trainer.fit(model, train_loader, val_loader)