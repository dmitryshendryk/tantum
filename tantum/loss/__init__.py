import torch
from torch import nn
from torch.nn import functional as F


from tantum.loss.focal_cosine import FocalCosineLoss
from tantum.loss.label_smoothing import LabelSmoothing



def get_criterion(cfg):
    if cfg.criterion=='CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss
    elif cfg.criterion=='LabelSmoothing':
        criterion = LabelSmoothing
    elif cfg.criterion=='FocalCosineLoss':
        criterion = FocalCosineLoss
    # elif cfg.criterion=='SymmetricCrossEntropyLoss':
        # criterion = SymmetricCrossEntropy().to(device)
    # elif cfg.criterion=='BiTemperedLoss':
        # criterion = BiTemperedLogisticLoss(t1=cfg.t1, t2=cfg.t2, smoothing=cfg.smoothing).to(device)
    # elif cfg.criterion=='TaylorCrossEntropyLoss':
        # criterion = TaylorCrossEntropyLoss(smoothing=cfg.smoothing).to(device)
    elif cfg.criterion == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss
    return criterion