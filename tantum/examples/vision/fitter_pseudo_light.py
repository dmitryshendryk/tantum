import torch
import numpy as np 
import pandas as pd
import time 
from datetime import datetime

import os 
import sys

PROJECT_ROOT = '/Users/dmitry/Documents/Personal/study/kaggle/tantum'
sys.path.append(PROJECT_ROOT)



import pytorch_lightning as pl
from sklearn.preprocessing import Normalizer
from torchvision.datasets import MNIST
from torchvision import transforms
from pytorch_lightning.metrics.functional import accuracy

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from tantum.trainer.lightining.fitter import FitterLightining
from tantum.trainer.lightining.pseudo_label import FitterPseudoLightining
from tantum.model.simple_net import Net
from tantum.datasets.datasets import split_relabel_data

class CFG:
    device = 'cpu'
    criterion = 'CrossEntropyLoss'
    optim = 'adam'
    lr=0.001


# data
def mnist(n_labels, data_root='data'):

    x_train = pd.read_csv(os.path.join(data_root,'mnist_train.csv'))
    y_train = x_train['label']
    x_train.drop(['label'], inplace = True, axis = 1)

    x_test = pd.read_csv(os.path.join(data_root,'mnist_test.csv'))
    y_test = x_test['label']
    x_test.drop(['label'], inplace = True, axis = 1)

    x_train = x_train.values
    y_train = y_train.values
    x_test = x_test.values

    normalizer = Normalizer()
    x_train = normalizer.fit_transform(x_train)
    x_test = normalizer.transform(x_test)

    x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor) 

    x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
    y_test = torch.from_numpy(y_test.values).type(torch.LongTensor)


    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    evalset = torch.utils.data.TensorDataset(x_test, y_test)

    num_classes = 10
    label_per_class = n_labels // num_classes
    labeled_idxs, unlabed_idxs = split_relabel_data(
                                    np.array(y_train),
                                    y_train,
                                    label_per_class,
                                    num_classes)

    return {
        'trainset': trainset,
        'evalset': evalset,
        'label_idxs': labeled_idxs,
        'unlab_idxs': unlabed_idxs,
        'num_classes': num_classes
    }

def create_loaders_v2(trainset, evalset, label_idxs, unlab_idxs,
                      num_classes):
    ## supervised batch loader
    label_sampler = SubsetRandomSampler(label_idxs)
    label_batch_sampler = BatchSampler(label_sampler, 100,
                                       drop_last=True)
    label_loader = torch.utils.data.DataLoader(trainset,
                                          batch_sampler=label_batch_sampler,
                                          num_workers=0,
                                          pin_memory=True)
    ## unsupervised batch loader
    unlab_sampler = SubsetRandomSampler(unlab_idxs)
    unlab_batch_sampler = BatchSampler(unlab_sampler, 100,
                                       drop_last=True)
    unlab_loader = torch.utils.data.DataLoader(trainset,
                                          batch_sampler=unlab_batch_sampler,
                                          num_workers=0,
                                          pin_memory=True)
    ## test batch loader
    eval_loader = torch.utils.data.DataLoader(evalset,
                                           batch_size=100,
                                           shuffle=False,
                                           num_workers=0,
                                           pin_memory=True,
                                           drop_last=False)
    return label_loader, unlab_loader, eval_loader

data =   mnist(60000, data_root='../../../data')
loaders = create_loaders_v2(**data)

class MyFitter(FitterPseudoLightining):

    def __init__(self, net, cfg):
        super().__init__(net, cfg)
    
    def train_dataloader(self):
        return {'labeled': loaders[0], 'pseudo': loaders[1]}
    
    def val_dataloader(self):
        return loaders[2]
        
class CFG:
    device = 'cpu'
    criterion = 'CrossEntropyLoss'
    optim = 'adam'
    lr=0.001

#model
model = MyFitter(Net(), CFG)

# training
trainer = pl.Trainer(gradient_clip_val=100, stochastic_weight_avg=True)
trainer.fit(model)