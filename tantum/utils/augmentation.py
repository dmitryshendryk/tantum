
import torch
from torch import nn

import numpy as np
from fmix import sample_mask


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha),0.3,0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (target, shuffled_target, lam)

    return new_data, targets



def fmix(data, targets, alpha, decay_power, shape, device, max_soft=0.0, reformulate=False):
    lam, mask = sample_mask(alpha, decay_power, shape, max_soft, reformulate)
    #mask =torch.tensor(mask, device=device).float()
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    x1 = torch.from_numpy(mask).to(device)*data
    x2 = torch.from_numpy(1-mask).to(device)*shuffled_data
    targets=(targets, shuffled_targets, lam)
    
    return (x1+x2), targets



def mixup(alpha, data, target):
    with torch.no_grad():
        bs = data.size(0)
        c = np.random.beta(alpha, alpha)

        perm = torch.randperm(bs).cuda()

        md = c * data + (1-c) * data[perm, :]
        mt = c * target + (1-c) * target[perm, :]
        return md, mt


class MixUpWrapper(object):
    def __init__(self, num_classes, alpha, dataloader):
        self.num_classes = num_classes
        self.alpha = alpha
        self.dataloader = dataloader

    def mixup_loader(self, loader):
        for input, target in loader:
            target = self.expand(self.num_classes, torch.float, target)
            i, t = mixup(self.alpha, input, target)
            yield i, t

    def __iter__(self):
        return self.mixup_loader(self.dataloader)

    def expand(self, num_classes, dtype, tensor):
        e = torch.zeros(tensor.size(0), num_classes, dtype=dtype)#, device=torch.device('cuda'))
        e = e.scatter(1, tensor.unsqueeze(1), 1.0)
        return e

class NLLMultiLabelSmooth(nn.Module):
    def __init__(self, smoothing = 0.0):
        super(NLLMultiLabelSmooth, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim = -1)
    
            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
    
            smooth_loss = -logprobs.mean(dim=-1)
    
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
    
            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)
            