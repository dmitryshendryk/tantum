import torch

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy


from tantum.utils.loss import get_criterion
from tantum.optimizer.optimizer import create_optimizer

class FitterLightining(pl.LightningModule):

    def __init__(self, net, cfg):
        super().__init__()
        self.net = net
        self.cfg = cfg

    def forward(self, x):
        return self.net(x)
    
    def configure_optimizers(self):
        optimizer = create_optimizer(self.net.parameters(), self.cfg)
        return optimizer

    def loss_fn(self,out,target):
        return get_criterion(self.cfg, self.cfg.device)(out, target)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        output = self(x)
        labeled_loss = self.loss_fn(output, y)
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
