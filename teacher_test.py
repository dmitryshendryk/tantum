
from __future__ import print_function
import argparse

import pandas as pd
from PIL import Image
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torch.utils.data import Subset
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import StratifiedKFold

from torch.optim import Adam, SGD

from tantum.trainer.v1.mean_teacher import MeanTeacher
from tantum.datasets.data_utils import DataSetWarpper, NO_LABEL, TransformTwice, create_loaders_v2
from tantum.datasets.datasets import encode_label, decode_label, split_relabel_data

parser = argparse.ArgumentParser()
parser.add_argument("--cons_weight", default=10, help="consistency weight")
args = parser.parse_args()

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
    cutmix=False
    lr = 0.001
    swa = False
    nprocs = 1
    swa_start = 5
    label_exclude = False
    print_freq = 100
    scheduler = 'GradualWarmupSchedulerV2'
    batch_size = 100
    weight_decay=1e-6
    gradient_accumulation_steps=1
    max_grad_norm=1000
    usp_batch_size=100
    n_fold=5
    T_0 = 10
    usp_weight=30.0
    optimizer = Adam
    sup_batch_size=100
    ema_decay = 0.97
    data_twice = True 
    data_idxs = False 
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
        return F.log_softmax(x, dim=1)



train_transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Pad(2, padding_mode='reflect'),
        # transforms.ColorJitter(brightness=0.4, contrast=0.4,
        #                        saturation=0.4, hue=0.1),
        # transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

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
            image = self.transforms(image)
            
        return image, torch.tensor(target, dtype=torch.long)

    def __len__(self):
        return self.y.shape[0]

def main():

    df = pd.read_csv('./train.csv')

    if CFG.debug:
        CFG.n_epochs = 1
        df = df.sample(n=1000, random_state=CFG.seed).reset_index(drop=True)


    folds = df.copy()
    Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[CFG.target_col])):
        folds.loc[val_index, 'fold'] = int(n)
    folds['fold'] = folds['fold'].astype(int)

    fold = 0 

    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    y = train_folds['label'].values
    X = train_folds.drop(['label','fold'], axis=1).values

    y_val = valid_folds['label'].values
    X_val = valid_folds.drop(['label','fold'], axis=1).values



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    step_counter = 0
    model = Net().to(device)
    ema_model = Net().to(device)
    for param in ema_model.parameters():
        param.detach_()


    train_dataset = DatasetRetriever(
            X = X,
            y = y,
            transforms=train_transform,
        )
    
    test_dataset = DatasetRetriever(
            X = X_val,
            y = y_val,
            transforms=eval_transform,
        )
    n_labels = 10000
    num_classes = 10

    label_per_class = n_labels // num_classes
    labeled_idxs, unlabed_idxs = split_relabel_data(
                                    np.array(train_dataset.y),
                                    train_dataset.y,
                                    label_per_class,
                                    num_classes)

    label_loader, unlab_loader, eval_loader = create_loaders_v2(
                                                                    train_dataset,
                                                                    test_dataset,
                                                                    labeled_idxs,
                                                                    unlabed_idxs,
                                                                    num_classes,
                                                                    CFG
    )

    
    fitter = MeanTeacher(
        cfg=CFG,
        model = Net(),
        mean_teacher=Net(),
        device=device,
        optimizer = CFG.optimizer,
        n_epochs = CFG.n_epochs,
        sheduler = CFG.scheduler,
        optimizer_params = CFG.optimizer_params
    )


    fitter.fit(CFG, fold, label_loader, unlab_loader, eval_loader, valid_folds)

if __name__ == '__main__':
    main()