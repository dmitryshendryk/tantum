import torch
import numpy as np 
import time 
from datetime import datetime

from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel, SWALR

from tantum.utils.augmentation import cutmix, fmix
from tantum.scheduler.scheduler import  get_scheduler, GradualWarmupSchedulerV2
from tantum.utils.metrics import get_score
from tantum.utils.loss import get_criterion
from tantum.utils.logger import LOGGER
from tantum.utils.metrics import AverageMeter, timeSince


class Fitter():

    def __init__(
        self, cfg, model, device, optimizer, n_epochs,
        sheduler=None, optimizer_params=None, xm = None, pl = None, idist=None
    ) -> None:
        self.epoch = 0
        self.n_epochs = n_epochs
        self.base_dir = './'
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = np.inf

        self.model = model
        self.device = device
        self.cfg = cfg

        self.xm = xm 
        self.pl = pl 
        self.idist = idist

        self.optimizer = optimizer
        self.optimizer_params = optimizer_params

        self.sheduler = sheduler

        
    
    def fit(self, CFG, fold, train_loader, valid_loader, valid_folds):
        LOGGER.info(f"========== Base Fitter ==========")
        LOGGER.info(f"========== fold: {fold} training ==========")
        
        
        valid_labels = valid_folds[CFG.target_col].values

        if CFG.device == 'TPU':
            self.device = self.xm.xla_device()
            self.xm.set_rng_state(CFG.seed, self.device)


        self.model.to(self.device)
        # ====================================================
        # optimizer 
        # ====================================================
        ### Create optimizer
        # optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
        self.optimizer = self.optimizer(self.model.parameters(), **self.optimizer_params)
        
        # ====================================================
        # scheduler 
        # ====================================================
        ### Create scheduler
        if self.sheduler:
            self.scheduler = get_scheduler(CFG, self.optimizer)
        
        # ====================================================
        # criterion 
        # ====================================================
        ### Create criterion
        self.criterion = get_criterion(CFG, self.device)
        
        LOGGER.info(f'Fitter prepared. Device is {self.device}')
        
        
        best_score = 0.
        best_loss = np.inf
        
        for epoch in range(self.n_epochs):
            
            start_time = time.time()
            
            # ====================================================
            # train
            # ====================================================
            if CFG.device == 'TPU':
                if CFG.nprocs == 1:
                    avg_loss = self.train(train_loader, fold, epoch)
                elif CFG.nprocs > 1:
                    # para_train_loader = self.pl.ParallelLoader(train_loader, [self.device])
                    # avg_loss = self.train(para_train_loader.per_device_loader(self.device),  fold, epoch)
                    mp_device_loader = self.pl.MpDeviceLoader(train_loader, self.device)

                    avg_loss = self.train(mp_device_loader, fold, epoch)
            elif CFG.device == 'GPU':
                    avg_loss = self.train(train_loader, fold, epoch)
            
            # ====================================================
            # eval baseline
            # ====================================================
            if CFG.device == 'TPU':
                if CFG.nprocs == 1:
                    avg_val_loss, preds, _ = self.validation(valid_loader, fold)
                elif CFG.nprocs > 1:
                    # para_valid_loader = self.pl.ParallelLoader(valid_loader, [self.device])
                    # avg_val_loss, preds, valid_labels = self.validation(para_valid_loader.per_device_loader(self.device), fold)
                    # preds = self.idist.all_gather(torch.tensor(preds)).to('cpu').numpy()
                    # valid_labels = self.idist.all_gather(torch.tensor(valid_labels)).to('cpu').numpy()
                    mp_device_loader = self.pl.MpDeviceLoader(valid_loader, self.device)
                    avg_val_loss, preds, _ = self.validation(mp_device_loader, fold)
            elif CFG.device == 'GPU':
                    avg_val_loss, preds, _ = self.validation(valid_loader, fold)
            
            
            
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(avg_val_loss)
            elif isinstance(self.scheduler, CosineAnnealingLR):
                self.scheduler.step()
            elif isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                self.scheduler.step()
            elif isinstance(self.scheduler, GradualWarmupSchedulerV2):
                    self.scheduler.step(epoch)

            # ====================================================
            # scoring
            # ====================================================
            score = get_score(valid_labels, preds.argmax(1))

            elapsed = time.time() - start_time

            if CFG.device == 'GPU':
                LOGGER.info(f'Epoch {epoch+1} - Fold {fold} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
                LOGGER.info(f'Epoch {epoch+1} - Fold {fold} - Score: {score:.4f}')
            elif CFG.device == 'TPU':
                if CFG.nprocs == 1:
                    LOGGER.info(f'Epoch {epoch+1} - Fold {fold} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
                    LOGGER.info(f'Epoch {epoch+1} - Fold {fold} - Score: {score:.4f}')
                elif CFG.nprocs > 1:
                    LOGGER.info(f'Epoch {epoch+1} - Fold {fold} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
                    LOGGER.info(f'Epoch {epoch+1} - Fold {fold} - Score: {score:.4f}')
            
            if score > best_score:
                best_score = score
                if CFG.device == 'GPU':
                    LOGGER.info(f'Epoch {epoch+1} - Fold {fold} - Save Best Score: {best_score:.4f} Model')
                    torch.save({'model': self.model.state_dict(), 
                                'preds': preds},
                            CFG.OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_score.pth')
                elif CFG.device == 'TPU':
                    if CFG.nprocs == 1:
                        LOGGER.info(f'Epoch {epoch+1} - Fold {fold} - Save Best Score: {best_score:.4f} Model')
                    elif CFG.nprocs > 1:
                        LOGGER.info(f'Epoch {epoch+1} - Fold {fold} - Save Best Score: {best_score:.4f} Model')
                    self.xm.save({'model': self.model, 
                            'preds': preds}, 
                            CFG.OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_score.pth')

        
        if CFG.nprocs != 8:
            check_point = torch.load(CFG.OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_score.pth')
            valid_folds[[str(c) for c in range(CFG.target_size)]] = check_point['preds']
            valid_folds['preds'] = check_point['preds'].argmax(1)

        return valid_folds

    def train(self, train_loader, fold=0, epoch=0):
        if self.cfg.device == 'GPU':
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
            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = labels.size(0)
            
            mix_decision = np.random.rand()
            if mix_decision < 0.25 and self.cfg.cutmix:
                images, labels = cutmix(images, labels, 1.)
            elif mix_decision >= 0.25 and mix_decision < 0.5 and self.cfg.fmix:
                images, labels = fmix(images, labels, alpha=1., decay_power=5., shape=(self.cfg.size, self.cfg.size), device=self.device)

            if self.cfg.device == 'GPU':
                with autocast():
                    y_preds = self.model(images.float())
                    
                    if mix_decision < 0.50 and (self.cfg.fmix or self.cfg.cutmix):
                        loss = self.criterion(y_preds, labels[0]) * labels[2] + self.criterion(y_preds, labels[1]) * (1. - labels[2])
                    else:
                        loss = self.criterion(y_preds.view(-1), labels)
                
                # record loss
                if self.cfg.gradient_accumulation_steps > 1:
                    loss = loss / self.cfg.gradient_accumulation_steps
                
                losses.update(loss.item(), batch_size)
                scaler.scale(loss).backward()
                
                if (step + 1) % self.cfg.gradient_accumulation_steps == 0:

                    scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)

                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    global_step += 1
                            
            elif self.cfg.device == 'TPU':
                self.optimizer.zero_grad()

                y_preds = self.model(images.float())
                
                if mix_decision < 0.50 and (self.cfg.fmix or self.cfg.cutmix):
                    loss = self.criterion(y_preds, labels[0]) * labels[2] + self.criterion(y_preds, labels[1]) * (1. - labels[2])
                else:
                    loss = self.criterion(y_preds, labels)
                
                # record loss
                if self.cfg.gradient_accumulation_steps > 1:
                    loss = loss / self.cfg.gradient_accumulation_steps
                
                losses.update(loss.item(), batch_size)
                loss.backward()
                # scaler.scale(loss).backward()
                if (step + 1) % self.cfg.gradient_accumulation_steps == 0:
                    gradients = self.xm._fetch_gradients(self.optimizer)
                    self.xm.all_reduce('sum', gradients, scale=1.0 / self.xm.xrt_world_size())
                    self.xm.mark_step()

                # if (step + 1) % self.cfg.gradient_accumulation_steps == 0:

                #     # grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                #     self.xm.reduce_gradients(self.optimizer)
                #     self.xm.optimizer_step(self.optimizer, barrier=True)
                #     global_step += 1
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if self.cfg.device == 'GPU':
                if step % self.cfg.print_freq == 0 or step == (len(train_loader)-1):
                    LOGGER.info('Epoch: [{0}][{1}/{2}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Elapsed {remain:s} '
                        'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                        'Grad: {grad_norm:.4f}  '
                        'Fold: {fold} '
                        'Model: {model_name} '
                        #'LR: {lr:.6f}  '
                        .format(
                        epoch+1, step, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses,
                        remain=timeSince(start, float(step+1)/len(train_loader)),
                        grad_norm=grad_norm,
                        fold=fold,
                        model_name=self.cfg.model_name
                        #lr=scheduler.get_lr()[0],
                        ))
            elif self.cfg.device == 'TPU':
                if step % self.cfg.print_freq == 0 or step == (len(train_loader)-1):
                    LOGGER.info('Epoch: [{0}][{1}/{2}] '
                                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                                    'Elapsed {remain:s} '
                                    'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                                    # 'Grad: {grad_norm:.4f}  '
                                    'Fold: {fold} '
                                    'Model: {model_name} '
                                    #'LR: {lr:.6f}  '
                                    .format(
                                    epoch+1, step, len(train_loader), batch_time=batch_time,
                                    data_time=data_time, loss=losses,
                                    remain=timeSince(start, float(step+1)/len(train_loader)),
                                    # grad_norm=grad_norm,
                                    fold=fold,
                                    model_name=self.cfg.model_name
                                    #lr=scheduler.get_lr()[0],
                                    ))
        return losses.avg
    
    def validation(self, valid_loader, fold=0):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        scores = AverageMeter()
        # switch to evaluation mode
        self.model.eval()
        trues = []
        preds = []
        start = end = time.time()
        for step, (images, labels) in enumerate(valid_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = labels.size(0)
            
            # compute loss
            with torch.no_grad():
                y_preds = self.model(images)
            loss = self.criterion(y_preds.view(-1), labels)
            losses.update(loss.item(), batch_size)
            # record accuracy
            trues.append(labels.to('cpu').numpy())
            preds.append(y_preds.softmax(1).to('cpu').numpy())
            if self.cfg.gradient_accumulation_steps > 1:
                loss = loss / self.cfg.gradient_accumulation_steps
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if self.cfg.device == 'GPU':
                if step % self.cfg.print_freq == 0 or step == (len(valid_loader)-1):
                    LOGGER.info('EVAL: [{0}/{1}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Elapsed {remain:s} '
                        'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                        'Fold: {fold} '
                        'Model: {model_name} '
                        .format(
                        step, len(valid_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses,
                        remain=timeSince(start, float(step+1)/len(valid_loader)),
                        fold=fold,
                        model_name=self.cfg.model_name
                        ))
            elif self.cfg.device == 'TPU':
                if step % self.cfg.print_freq == 0 or step == (len(valid_loader)-1):
                    LOGGER.info('EVAL: [{0}/{1}] '
                                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                                    'Elapsed {remain:s} '
                                    'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                                    'Fold: {fold} '
                                    'Model: {model_name} '
                                    .format(
                                    step, len(valid_loader), batch_time=batch_time,
                                    data_time=data_time, loss=losses,
                                    remain=timeSince(start, float(step+1)/len(valid_loader)),
                                    fold=fold,
                                    model_name=self.cfg.model_name
                                    ))

        trues = np.concatenate(trues)
        predictions = np.concatenate(preds)
        return losses.avg, predictions, trues 


