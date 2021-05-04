
import os
import cv2
import math
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import timm 

class VIT_Attention(nn.Module):
    def __init__(self,
        arch_name,
        pretrained=False,
        img_size=256,
        multi_drop=False,
        multi_drop_rate=0.5,
        att_layer=False,
        att_pattern="A"
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.att_layer = att_layer
        self.multi_drop = multi_drop

        self.model = timm.create_model(
            arch_name, pretrained=pretrained
        )
        n_features = self.model.head.in_features
        self.model.head = nn.Identity()

        self.head = nn.Linear(n_features, 5)
        self.head_drops = nn.ModuleList()
        for i in range(5):
            self.head_drops.append(nn.Dropout(multi_drop_rate))

        if att_layer:
            if att_pattern == "A":
                self.att_layer = nn.Sequential(
                    nn.Linear(n_features, 256),
                    nn.Tanh(),
                    nn.Linear(256, 1),
                )
            elif att_pattern == "B":
                self.att_layer = nn.Linear(n_features, 1)
            else:
                raise ValueError("invalid att pattern")

    def forward(self, x):
        if self.att_layer:
            l = x.shape[2] // 2
            h1 = self.model(x[:, :, :l, :l])
            h2 = self.model(x[:, :, :l, l:])
            h3 = self.model(x[:, :, l:, :l])
            h4 = self.model(x[:, :, l:, l:])
            w = F.softmax(torch.cat([
                self.att_layer(h1),
                self.att_layer(h2),
                self.att_layer(h3),
                self.att_layer(h4),
            ], dim=1), dim=1)
            h = h1 * w[:, 0].unsqueeze(-1) + \
                h2 * w[:, 1].unsqueeze(-1) + \
                h3 * w[:, 2].unsqueeze(-1) + \
                h4 * w[:, 3].unsqueeze(-1)
        else:
            h = self.model(x)

        if self.multi_drop:
            for i, dropout in enumerate(self.head_drops):
                if i == 0:
                    output = self.head(dropout(h))
                else:
                    output += self.head(dropout(h))
            output /= len(self.head_drops)
        else:
            output = self.head(h)
        return output
