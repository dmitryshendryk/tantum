import time
import numpy as np

import torch
from torch.cuda.amp import autocast, GradScaler

from tantum.utils.metrics import AverageMeter, timeSince
from tantum.utils.augmentation import cutmix, fmix


class Trainer():

    def __init__(self, model, criterion, optimizer, scheduler, xm=None) -> None:

        self.model = model 
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.xm = xm


    def fit(self, cfg, train_loader,  epoch, device, fold):
        if cfg.device == 'GPU':
            scaler = GradScaler()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        scores = AverageMeter()
        # switch to train mode
        self.model.train()
        start = end = time.time()
        global_step = 0
        for step, (images, labels) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            images = images.to(device).float()
            labels = labels.to(device).long()
            batch_size = labels.size(0)
            
            mix_decision = np.random.rand()
            if mix_decision < 0.25 and cfg.cutmix:
                images, labels = cutmix(images, labels, 1.)
            elif mix_decision >= 0.25 and mix_decision < 0.5 and cfg.fmix:
                images, labels = fmix(images, labels, alpha=1., decay_power=5., shape=(512,512), device=device)

            if cfg.device == 'GPU':
                with autocast():
                    y_preds = self.model(images.float())
                    
                    if mix_decision < 0.50 and (cfg.fmix or cfg.cutmix):
                        loss = self.criterion(y_preds, labels[0]) * labels[2] + self.criterion(y_preds, labels[1]) * (1. - labels[2])
                    else:
                        loss = self.criterion(y_preds, labels)
                    # record loss
                    losses.update(loss.item(), batch_size)
                    if cfg.gradient_accumulation_steps > 1:
                        loss = loss / cfg.gradient_accumulation_steps
                    scaler.scale(loss).backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                    if (step + 1) % cfg.gradient_accumulation_steps == 0:
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.optimizer.zero_grad()
                        global_step += 1
                            
            elif cfg.device == 'TPU':
                y_preds = self.model(images)
                loss = self.criterion(y_preds, labels)
                # record loss
                losses.update(loss.item(), batch_size)
                if cfg.gradient_accumulation_steps > 1:
                    loss = loss / cfg.gradient_accumulation_steps
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    self.xm.optimizer_step(self.optimizer, barrier=True)
                    self.optimizer.zero_grad()
                    global_step += 1
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if cfg.device == 'GPU':
                if step % cfg.print_freq == 0 or step == (len(train_loader)-1):
                    print('Epoch: [{0}][{1}/{2}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Elapsed {remain:s} '
                        'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                        'Grad: {grad_norm:.4f}  '
                        'Fold: {fold:4f}'
                        #'LR: {lr:.6f}  '
                        .format(
                        epoch+1, step, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses,
                        remain=timeSince(start, float(step+1)/len(train_loader)),
                        grad_norm=grad_norm,
                        fold=fold
                        #lr=scheduler.get_lr()[0],
                        ))
            elif cfg.device == 'TPU':
                if step % cfg.print_freq == 0 or step == (len(train_loader)-1):
                    self.xm.master_print('Epoch: [{0}][{1}/{2}] '
                                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                                    'Elapsed {remain:s} '
                                    'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                                    'Grad: {grad_norm:.4f}  '
                                    'Fold: {fold:4f}'
                                    #'LR: {lr:.6f}  '
                                    .format(
                                    epoch+1, step, len(train_loader), batch_time=batch_time,
                                    data_time=data_time, loss=losses,
                                    remain=timeSince(start, float(step+1)/len(train_loader)),
                                    grad_norm=grad_norm,
                                    fold=fold
                                    #lr=scheduler.get_lr()[0],
                                    ))
        return losses.avg
