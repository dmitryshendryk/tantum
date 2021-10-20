import torch
import torch.nn.functional as F 

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

from tantum.utils.rampup import exp_rampup
from tantum.datasets.data_utils import NO_LABEL
from tantum.datasets.datasets import decode_label

from tantum.utils.loss import get_criterion
from tantum.optimizer.optimizer import create_optimizer


class FitterPseudoLightining(pl.LightningModule):

    def __init__(self, net, cfg):
        super().__init__()
        self.net = net
        self.usp_weight = 1.0
        self.epoch = 10
        self.rampup = exp_rampup(self.usp_weight)
        self.cfg = cfg

    def forward(self, x):
        return self.net(x)
    
    def configure_optimizers(self):
        optimizer = create_optimizer(self.net.parameters(), self.cfg)
        return optimizer
    
    def decode_targets(self, targets):
        label_mask = targets.ge(0)
        unlab_mask = targets.le(NO_LABEL)
        targets[unlab_mask] = decode_label(targets[unlab_mask])
        return label_mask, unlab_mask

    def loss_fn(self,out,target):
        return get_criterion(self.cfg, self.cfg.device)(out, target)

    def training_step(self, train_batch, batch_idx):
        ## Labeled training
        x_labeled, y_labeled = train_batch['labeled']
        output_labeled = self(x_labeled)
        labeled_loss = self.loss_fn(output_labeled, y_labeled)

        ## Semi training
        x_unlabeled, y_unlabeled = train_batch['pseudo']

        self.decode_targets(y_unlabeled)

        unlab_outputs = self(x_unlabeled)
        with torch.no_grad():
            iter_unlab_pslab = unlab_outputs.max(1)[1]
            iter_unlab_pslab.detach_()

        uloss  = self.loss_fn(unlab_outputs, iter_unlab_pslab)
        uloss *= self.rampup(self.epoch)*self.usp_weight
        labeled_loss  += uloss

        self.log('train_loss', labeled_loss, prog_bar=True)
        return labeled_loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
