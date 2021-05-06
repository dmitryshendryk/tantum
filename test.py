from __future__ import print_function
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torchvision import datasets, transforms

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn import model_selection

from sklearn.model_selection import StratifiedKFold
import efficientnet_pytorch
from torch.utils.data import Subset

import cv2
import os 
import sys
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import random

from torchvision.utils import make_grid
from torch.autograd import Variable

import pandas as pd
import time
from tqdm import tqdm
from torch.optim import Adam, SGD



from tantum.trainer.v1.fitter import Fitter
from tantum.trainer.v1.mean_teacher import MeanTeacher

from tantum.utils.loss import get_criterion
from tantum.scheduler.scheduler import get_scheduler
from tantum.scheduler.scheduler import GradualWarmupSchedulerV2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class CFG:
    debug = False
    weight_mean_teacher = 0.2
    alpha_mean_teacher = 0.99
    weight_rampup = 30
    apex = True
    seed=42
    criterion = 'CrossEntropyLoss' 
    device= 'GPU'
    fmix=False 
    target_size = 10
    cutmix=False
    lr = 0.001
    swa = False
    nprocs = 1
    swa_start = 5
    print_freq = 100
    scheduler = 'GradualWarmupSchedulerV2'
    optimizer = Adam
    batch_size = 100
    weight_decay=1e-6
    gradient_accumulation_steps=1
    max_grad_norm=1000
    n_fold=5
    T_0 = 10
    target_col = 'label'
    trn_fold=[0,1,2,3,4] #[0, 1, 2, 3, 4]
    num_workers = 0
    freeze_epo = 0 # GradualWarmupSchedulerV2
    warmup_epo = 1 # GradualWarmupSchedulerV2
    cosine_epo = 9 # GradualWarmupSchedulerV2  ## 19
    n_epochs = freeze_epo + warmup_epo + cosine_epo
    OUTPUT_DIR = './'
    model_name = 'simple_net'
    optimizer_params = dict(
        lr=lr, 
        weight_decay=weight_decay, 
        amsgrad=False
    )

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG.seed)

df = pd.read_csv('./train.csv')

if CFG.debug:
    CFG.n_epochs = 1
    df = df.sample(n=1000, random_state=CFG.seed).reset_index(drop=True)

df = df.sample(n=1000, random_state=CFG.seed).reset_index(drop=True)

folds = df.copy()
Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[CFG.target_col])):
    folds.loc[val_index, 'fold'] = int(n)
folds['fold'] = folds['fold'].astype(int)


class DatasetRetriever(Dataset):
    def __init__(self, X, y, transforms=None):
        super().__init__()
        self.X = X.reshape(-1, 28, 28).astype(np.float32)
        self.y = y
        self.transforms = transforms

    def __getitem__(self, index):
        image, target = self.X[index], self.y[index]
        image = np.stack([image] * 1, axis=-1)
        image /= 255.
        if self.transforms:
            image = self.transforms(image=image)['image']
            
        return image, torch.tensor(target, dtype=torch.long)

    def __len__(self):
        return self.y.shape[0]

def get_train_transforms():
    return A.Compose(
        [
            A.Rotate(limit=10, border_mode=cv2.BORDER_REPLICATE, p=0.5),
            A.Cutout(num_holes=8, max_h_size=2, max_w_size=2, fill_value=0, p=0.5),
            A.Cutout(num_holes=8, max_h_size=1, max_w_size=1, fill_value=1, p=0.5),
            ToTensorV2(p=1.0),
        ], 
        p=1.0)

def get_valid_transforms():
    return A.Compose(
        [
            ToTensorV2(p=1.0),
        ], 
        p=1.0
    )

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


num_indices = 100
indices = list(range(num_indices*8))




oof_df = pd.DataFrame()
for fold in range(CFG.n_fold):
    if fold in CFG.trn_fold:
        fitter = Fitter(
            cfg=CFG,
            model = Net(),
            device=device,
            optimizer = CFG.optimizer,
            n_epochs = CFG.n_epochs,
            sheduler = CFG.scheduler,
            optimizer_params = CFG.optimizer_params
        )
        trn_idx = folds[folds['fold'] != fold].index
        val_idx = folds[folds['fold'] == fold].index

        train_folds = folds.loc[trn_idx].reset_index(drop=True)
        valid_folds = folds.loc[val_idx].reset_index(drop=True)

        y = train_folds['label'].values
        X = train_folds.drop(['label','fold'], axis=1).values

        y_val = valid_folds['label'].values
        X_val = valid_folds.drop(['label','fold'], axis=1).values

        train_dataset = DatasetRetriever(
            X = X,
            y = y,
            transforms=get_train_transforms(),
        )

        valid_dataset = DatasetRetriever(
            X = X_val,
            y = y_val,
            transforms=get_valid_transforms(),
        )

        
        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
        )

        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
        )
        
        _oof_df = fitter.fit(CFG, fold, train_loader, valid_loader, valid_folds)
        oof_df = pd.concat([oof_df, _oof_df])

oof_df.to_csv('oof_df.csv', index=False)