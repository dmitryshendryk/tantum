import torch
import numpy as np 
import time 
from datetime import datetime
import pandas as pd

from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import torch.nn.functional as F 

from tantum.utils.augmentation import cutmix, fmix
from tantum.scheduler.scheduler import  get_scheduler, GradualWarmupSchedulerV2
from tantum.utils.metrics import get_score
from tantum.utils.loss import get_criterion
from tantum.utils.logger import LOGGER
from tantum.utils.metrics import AverageMeter, timeSince
from tantum.utils.loss import mse_with_softmax
from tantum.utils.rampup import exp_rampup
class MeanTeacher():

    def __init__(
        self, cfg, model, mean_teacher, device, optimizer, n_epochs,
        sheduler=None, optimizer_params=None, xm = None, pl = None, idist=None
    ) -> None:
        self.epoch = 0
        self.n_epochs = n_epochs
        self.base_dir = './'
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = np.inf

        self.model = model
        self.mean_teacher = mean_teacher
        self.device = device
        self.cfg = cfg

        self.xm = xm 
        self.pl = pl 
        self.iidist = idist
        self.rampup = exp_rampup(self.cfg.weight_rampup)

        # ====================================================
        # optimizer 
        # ====================================================
        ### Create optimizer
        # optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
        self.optimizer = optimizer(self.model.parameters(), **optimizer_params)
        
        # ====================================================
        # scheduler 
        # ====================================================
        ### Create scheduler
        if sheduler:
            self.scheduler = get_scheduler(cfg, self.optimizer)
        
        # ====================================================
        # criterion 
        # ====================================================
        ### Create criterion
        self.criterion = get_criterion(cfg, device)
        
        LOGGER.info(f'Fitter prepared. Device is {self.device}')
    
    def fit(self, CFG, fold, train_loader, valid_loader, valid_folds):
        LOGGER.info(f"========== Mean Teacher Trainig ==========")
        LOGGER.info(f"========== fold: {fold} training ==========")
        
        
        valid_labels = valid_folds[CFG.target_col].values
        
        self.model.to(self.device)
        self.mean_teacher.to(self.device)
        
        
        best_score_base = 0.
        best_score_mean = 0.
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
                    para_train_loader = self.pl.ParallelLoader(train_loader, [self.device])
                    avg_loss = self.train(para_train_loader.per_device_loader(self.device),  fold, epoch)
            elif CFG.device == 'GPU':
                    avg_loss = self.train(train_loader, fold, epoch)
            
            # ====================================================
            # eval baseline
            # ====================================================
            if CFG.device == 'TPU':
                if CFG.nprocs == 1:
                    avg_val_loss, preds, _ = self.validation(self.model, valid_loader, fold)
                elif CFG.nprocs > 1:
                    para_valid_loader = self.pl.ParallelLoader(valid_loader, [self.device])
                    avg_val_loss, preds, valid_labels = self.validation(self.model,para_valid_loader.per_device_loader(self.device), fold)
                    preds = self.idist.all_gather(torch.tensor(preds)).to('cpu').numpy()
                    valid_labels = self.idist.all_gather(torch.tensor(valid_labels)).to('cpu').numpy()
            elif CFG.device == 'GPU':
                    base_avg_val_loss, base_preds, _ = self.validation(self.model, valid_loader, fold)
                    mean_avg_val_loss, mean_preds, _ = self.validation(self.mean_teacher, valid_loader, fold)
            
            
            
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
            def print_scores(avg_val_loss, preds, model_name, best_score):
                LOGGER.info(f"===========  { model_name } ==============")
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
                        self.xm.master_print(f'Epoch {epoch+1} - Fold {fold} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
                        self.xm.master_print(f'Epoch {epoch+1} - Fold {fold} - Score: {score:.4f}')
                
                if score > best_score:
                    best_score = score
                    if CFG.device == 'GPU':
                        LOGGER.info(f'Epoch {epoch+1} - Fold {fold} - Save Best Score: {best_score:.4f} Model')
                        torch.save({'model': self.model.state_dict(), 
                                    'preds': preds},
                                CFG.OUTPUT_DIR+f'{CFG.model_name}_{model_name}_fold{fold}_best_score.pth')
                    elif CFG.device == 'TPU':
                        if CFG.nprocs == 1:
                            LOGGER.info(f'Epoch {epoch+1} - Fold {fold} - Save Best Score: {best_score:.4f} Model')
                        elif CFG.nprocs > 1:
                            self.xm.master_print(f'Epoch {epoch+1} - Fold {fold} - Save Best Score: {best_score:.4f} Model')
                        self.xm.save({'model': self.model, 
                                'preds': preds}, 
                                CFG.OUTPUT_DIR+f'{model_name}_fold{fold}_best_score.pth')
            
            print_scores(base_avg_val_loss, base_preds, 'base', best_score_base)
            print_scores(mean_avg_val_loss, mean_preds, 'mean', best_score_mean)
        
        if CFG.nprocs != 8:
            if best_score_base >= best_score_mean:
                check_point = torch.load(CFG.OUTPUT_DIR+f'{CFG.model_name}_base_fold{fold}_best_score.pth')
            else:
                check_point = torch.load(CFG.OUTPUT_DIR+f'{CFG.model_name}_mean_fold{fold}_best_score.pth')
            valid_folds['preds'] = check_point['preds'].argmax(1)
            valid_folds = pd.concat([valid_folds, pd.DataFrame(check_point['preds'])], axis=1)
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
            global_step += 1
            # measure data loading time
            data_time.update(time.time() - end)
            images = images.to(self.device).float()
            labels = labels.to(self.device).long()
            batch_size = labels.size(0)
            
            mix_decision = np.random.rand()
            if mix_decision < 0.25 and self.cfg.cutmix:
                images, labels = cutmix(images, labels, 1.)
            elif mix_decision >= 0.25 and mix_decision < 0.5 and self.cfg.fmix:
                images, labels = fmix(images, labels, alpha=1., decay_power=5., shape=(512,512), device=self.device)

            if self.cfg.device == 'GPU':
               
                with autocast(enabled=self.cfg.apex):
                    

                    ##=== forward ===
                    y_preds = self.model(images.float())
                    loss = self.criterion(y_preds, labels)

                    ##=== Semi-supervised Training ===
                    self.update_ema(self.model, self.mean_teacher, self.cfg.alpha_mean_teacher, global_step)
                    ## consistency loss
                    with torch.no_grad():
                        mean_t_output = self.mean_teacher(images.float())
                        mean_t_output = mean_t_output.detach()

                    cons_loss  = mse_with_softmax(y_preds, mean_t_output)
                    cons_loss *= self.rampup(epoch)*self.cfg.weight_mean_teacher
                    loss += cons_loss
                    

                    # record loss
                    losses.update(loss.item(), batch_size)
                    if self.cfg.gradient_accumulation_steps > 1:
                        loss = loss / self.cfg.gradient_accumulation_steps

                    scaler.scale(loss).backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    if (step + 1) % self.cfg.gradient_accumulation_steps == 0:
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.optimizer.zero_grad()
                        
                            
            elif self.cfg.device == 'TPU':
                y_preds = self.model(images)
                loss = self.criterion(y_preds, labels)
                # record loss
                losses.update(loss.item(), batch_size)
                if self.cfg.gradient_accumulation_steps > 1:
                    loss = loss / self.cfg.gradient_accumulation_steps
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                if (step + 1) % self.cfg.gradient_accumulation_steps == 0:
                    self.xm.optimizer_step(self.optimizer, barrier=True)
                    self.optimizer.zero_grad()
                    global_step += 1
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if self.cfg.device == 'GPU':
                if step % self.cfg.print_freq == 0 or step == (len(train_loader)-1):
                    print('Epoch: [{0}][{1}/{2}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Elapsed {remain:s} '
                        'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                        'Grad: {grad_norm:.4f}  '
                        'Fold: {fold}'
                        #'LR: {lr:.6f}  '
                        .format(
                        epoch+1, step, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses,
                        remain=timeSince(start, float(step+1)/len(train_loader)),
                        grad_norm=grad_norm,
                        fold=fold
                        #lr=scheduler.get_lr()[0],
                        ))
            elif self.cfg.device == 'TPU':
                if step % self.cfg.print_freq == 0 or step == (len(train_loader)-1):
                    self.xm.master_print('Epoch: [{0}][{1}/{2}] '
                                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                                    'Elapsed {remain:s} '
                                    'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                                    'Grad: {grad_norm:.4f}  '
                                    'Fold: {fold}'
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
    
    def validation(self, model, valid_loader, fold=0):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        scores = AverageMeter()
        # switch to evaluation mode
        model.eval()
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
                y_preds = model(images)
            loss = self.criterion(y_preds, labels)
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
                    print('EVAL: [{0}/{1}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Elapsed {remain:s} '
                        'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                        'Fold: {fold}'
                        .format(
                        step, len(valid_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses,
                        remain=timeSince(start, float(step+1)/len(valid_loader)),
                        fold=fold
                        ))
            elif self.cfg.device == 'TPU':
                if step % self.cfg.print_freq == 0 or step == (len(valid_loader)-1):
                    self.xm.master_print('EVAL: [{0}/{1}] '
                                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                                    'Elapsed {remain:s} '
                                    'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                                    'Fold: {fold}'
                                    .format(
                                    step, len(valid_loader), batch_time=batch_time,
                                    data_time=data_time, loss=losses,
                                    remain=timeSince(start, float(step+1)/len(valid_loader)),
                                    fold=fold
                                    ))

        trues = np.concatenate(trues)
        predictions = np.concatenate(preds)
        return losses.avg, predictions, trues 
    
    def update_ema(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step +1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1-alpha, param.data)