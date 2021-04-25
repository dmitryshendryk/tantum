import torch
import numpy as np 
import time 
from datetime import datetime

from tantum.utils.logger import LOGGER
from tantum.utils.metrics import AverageMeter, timeSince
from torch.cuda.amp import autocast, GradScaler

from torch.optim.swa_utils import AveragedModel, SWALR
from tantum.utils.augmentation import cutmix, fmix
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from tantum.scheduler.scheduler import  get_scheduler, GradualWarmupSchedulerV2

from tantum.utils.metrics import get_score

# class Fitter:
#     def __init__(
#         self, model, device, criterion, n_epochs, 
#         lr, sheduler=None, scheduler_params=None
#     ):
#         self.epoch = 0
#         self.n_epochs = n_epochs
#         self.base_dir = './'
#         self.log_path = f'{self.base_dir}/log.txt'
#         self.best_summary_loss = np.inf

#         self.model = model
#         self.device = device

#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
#         if sheduler:
#             self.scheduler = sheduler(self.optimizer, **scheduler_params)
            
#         self.criterion = criterion().to(self.device)
        
#         self.log(f'Fitter prepared. Device is {self.device}')

#     def fit(self, train_loader, valid_loader):
#         for e in range(self.n_epochs):
#             current_lr = self.optimizer.param_groups[0]['lr']
#             self.log(f'\n{datetime.datetime.utcnow().isoformat()}\nLR: {current_lr}')

#             t = int(time.time())
#             summary_loss, final_scores = self.train_one_epoch(train_loader)
#             self.log(
#                 f'[RESULT]: Train. Epoch: {self.epoch}, ' + \
#                 f'summary_loss: {summary_loss.avg:.5f}, ' + \
#                 f'final_score: {final_scores.avg:.5f}, ' + \
#                 f'time: {int(time.time()) - t} s'
#             )

#             t = int(time.time())
#             summary_loss, final_scores = self.validation(valid_loader)
#             self.log(
#                 f'[RESULT]: Valid. Epoch: {self.epoch}, ' + \
#                 f'summary_loss: {summary_loss.avg:.5f}, ' + \
#                 f'final_score: {final_scores.avg:.5f}, ' + \
#                 f'time: {int(time.time()) - t} s'
#             )
            
#             f_best = 0
#             if summary_loss.avg < self.best_summary_loss:
#                 self.best_summary_loss = summary_loss.avg
#                 f_best = 1

            
#             self.scheduler.step(metrics=summary_loss.avg)
                
#             self.save(f'{self.base_dir}/last-checkpoint.bin')
            
#             if f_best:
#                 self.save(f'{self.base_dir}/best-checkpoint.bin')
#                 print('New best checkpoint')

#             self.epoch += 1

#     def validation(self, val_loader):
#         self.model.eval()
#         summary_loss = LossMeter()
#         final_scores = AccMeter()
        
#         t = int(time.time())
#         for step, (images, targets) in enumerate(val_loader):
#             print(
#                 f'Valid Step {step}/{len(val_loader)}, ' + \
#                 f'summary_loss: {summary_loss.avg:.5f}, ' + \
#                 f'final_score: {final_scores.avg:.5f}, ' + \
#                 f'time: {int(time.time()) - t} s', end='\r'
#             )
            
#             with torch.no_grad():
#                 targets = targets.to(self.device)
#                 images = images.to(self.device)
#                 batch_size = images.shape[0]
                
#                 outputs = self.model(images)
#                 loss = self.criterion(outputs, targets)
                
#                 final_scores.update(targets, outputs)
#                 summary_loss.update(loss.detach().item(), batch_size)

#         return summary_loss, final_scores

#     def train_one_epoch(self, train_loader):
#         self.model.train()
#         summary_loss = LossMeter()
#         final_scores = AccMeter()
        
#         t = int(time.time())
#         for step, (images, targets) in enumerate(train_loader):
#             print(
#                 f'Train Step {step}/{len(train_loader)}, ' + \
#                 f'summary_loss: {summary_loss.avg:.5f}, ' + \
#                 f'final_score: {final_scores.avg:.5f}, ' + \
#                 f'time: {int(time.time()) - t} s', end='\r'
#             )
            
#             targets = targets.to(self.device)
#             images = images.to(self.device)
#             batch_size = images.shape[0]

#             self.optimizer.zero_grad()
#             outputs = self.model(images)
            
#             loss = self.criterion(outputs, targets)
#             loss.backward()

#             final_scores.update(targets, outputs.detach())
#             summary_loss.update(loss.detach().item(), batch_size)
            
#             self.optimizer.step()

#         return summary_loss, final_scores
    
#     def save(self, path):
#         self.model.eval()
#         torch.save({
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'scheduler_state_dict': self.scheduler.state_dict(),
#             'best_summary_loss': self.best_summary_loss,
#             'epoch': self.epoch,
#         }, path)

#     def load(self, path):
#         checkpoint = torch.load(path)
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#         self.best_summary_loss = checkpoint['best_summary_loss']
#         self.epoch = checkpoint['epoch'] + 1
        
#     def log(self, message):
#         print(message)
#         with open(self.log_path, 'a+') as logger:
#             logger.write(f'{message}\n')



class Fitter():

    def __init__(
        self, cfg, model, device, optimizer, criterion, n_epochs, 
        lr, sheduler=None, scheduler_params=None, xm = None, pl = None, idist=None
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
        self.iidist = idist

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        
        if sheduler:
            self.scheduler = sheduler(self.optimizer, **scheduler_params)
            
        self.criterion = criterion().to(self.device)
        
        LOGGER(f'Fitter prepared. Device is {self.device}')
    
    def fit(self, CFG, fold, train_loader, valid_loader, valid_folds):
        LOGGER.info(f"========== fold: {fold} training ==========")
        
        # # ====================================================
        # # loader
        # # ====================================================
        # trn_idx = folds[folds['fold'] != fold].index
        # val_idx = folds[folds['fold'] == fold].index

        # train_folds = folds.loc[trn_idx].reset_index(drop=True)
        # valid_folds = folds.loc[val_idx].reset_index(drop=True)
        
        # train_dataset = TrainDataset(train_folds, CFG.TRAIN_PATH, 'image_id', 'label', transform=self.transformer(data='train'))
        # valid_dataset = TrainDataset(valid_folds, CFG.TRAIN_PATH, 'image_id', 'label', transform=self.transformer(data='valid'))
        
        # train_loader = DataLoader(train_dataset, 
        #                         batch_size=CFG.batch_size, 
        #                         shuffle=True, 
        #                         num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
        # valid_loader = DataLoader(valid_dataset, 
        #                         batch_size=CFG.batch_size, 
        #                         shuffle=False, 
        #                         num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
        
        # valid_labels = valid_folds[CFG.target_col].values
        
        # # ====================================================
        # # device
        # # ====================================================
        # if CFG.device == 'TPU':
        #     device = xm.xla_device(fold+1)
        # elif CFG.device == 'GPU':
        #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        # ====================================================
        # model
        # ====================================================
        ### Create model
        # model = self.model(CFG.model_name, pretrained=True)
        self.model.to(self.device)
        
            
        # ====================================================
        # optimizer 
        # ====================================================
        ### Create optimizer
        # optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
        
        # ====================================================
        # scheduler 
        # ====================================================
        ### Create scheduler
        # scheduler = get_scheduler(CFG, optimizer)
        
        # ====================================================
        # criterion 
        # ====================================================
        ### Create criterion
        # criterion = get_criterion(CFG, device)
        
        # ====================================================
        # Stochastik Weighted Average
        # ====================================================
        
        if CFG.swa:
            swa_model = AveragedModel(self.model)
            swa_scheduler = SWALR(self.optimizer, swa_lr=0.05)
        else:
            swa_model = None
        
        # ====================================================
        # Trainer and Evaluator
        # ====================================================
        # if CFG.device == 'GPU':
        #     trainer = Trainer(model, criterion, optimizer, scheduler)
        #     evaluator = Evaluator(model, criterion)
        # elif CFG.device == 'TPU':
        #     trainer = Trainer(model, criterion, optimizer, scheduler, xm=xm)
        #     evaluator = Evaluator(model, criterion, xm=xm)
        
        
        best_score = 0.
        best_loss = np.inf
        
        for epoch in range(self.epochs):
            
            start_time = time.time()
            
            # ====================================================
            # train
            # ====================================================
            if self.device == 'TPU':
                if CFG.nprocs == 1:
                    avg_loss = self.train(train_loader, fold)
                elif CFG.nprocs > 1:
                    para_train_loader = self.pl.ParallelLoader(train_loader, [self.device])
                    avg_loss = self.train(para_train_loader.per_device_loader(self.device),  fold)
            elif CFG.device == 'GPU':
                    avg_loss = self.train(train_loader, fold)
            
            # ====================================================
            # eval baseline
            # ====================================================
            if CFG.device == 'TPU':
                if CFG.nprocs == 1:
                    avg_val_loss, preds, _ = self.validation(valid_loader, fold)
                elif CFG.nprocs > 1:
                    para_valid_loader = self.pl.ParallelLoader(valid_loader, [self.device])
                    avg_val_loss, preds, valid_labels = self.validation(para_valid_loader.per_device_loader(self.device), fold)
                    preds = self.idist.all_gather(torch.tensor(preds)).to('cpu').numpy()
                    valid_labels = self.idist.all_gather(torch.tensor(valid_labels)).to('cpu').numpy()
            elif CFG.device == 'GPU':
                    avg_val_loss, preds, _ = self.validation(valid_loader, fold)
            
            
            
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(avg_val_loss)
            elif isinstance(self.scheduler, CosineAnnealingLR):
                self.scheduler.step()
            elif isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                self.scheduler.step()
            elif isinstance(self.scheduler, GradualWarmupSchedulerV2):
                if epoch > CFG.swa_start and CFG.swa:
                    swa_model.update_parameters(self.model)
                    swa_scheduler.step()
                else:
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
                    self.xm.master_print(f'Epoch {epoch+1} - Fold {fold} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
                    self.xm.master_print(f'Epoch {epoch+1} - Fold {fold} - Score: {score:.4f}')
            
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
                        self.xm.master_print(f'Epoch {epoch+1} - Fold {fold} - Save Best Score: {best_score:.4f} Model')
                    self.xm.save({'model': self.model, 
                            'preds': preds}, 
                            CFG.OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_score.pth')
        
        # ====================================================
        # Update bn statistics for the swa_model
        # ====================================================
        if CFG.swa:
            torch.cuda.empty_cache()
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=self.device)
        
        #==========================
        # Compare SWA and Baseline
        #==========================
        
        if CFG.swa:
            avg_val_loss_baseline, preds_baseline, _ = self.validation(valid_loader, fold)
            score_baseline = get_score(valid_labels, preds_baseline.argmax(1))

            self.model = swa_model

            avg_val_loss_swa, preds_swa, _ = self.validation(valid_loader, fold)
            score_swa = get_score(valid_labels, preds_swa.argmax(1))

            if CFG.device == 'GPU':
                LOGGER.info(f'Fold {fold} - avg_val_loss_baseline: {avg_val_loss_baseline:.4f} ')
                LOGGER.info(f'Fold {fold} - Score_Baseline: {score_baseline:.4f}')
                LOGGER.info('=====================================')
                LOGGER.info(f'Fold {fold} - avg_val_loss_swa: {avg_val_loss_swa:.4f}')
                LOGGER.info(f'Fold {fold} - Score_SWA: {score_swa:.4f}')
        
        
        if CFG.nprocs != 8:
            check_point = torch.load(CFG.OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_score.pth')
            valid_folds['preds'] = check_point['preds'].argmax(1)

        return valid_folds

    def train(self, train_loader, fold=0):
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
            images = images.to(self.device).float()
            labels = labels.to(self.device).long()
            batch_size = labels.size(0)
            
            mix_decision = np.random.rand()
            if mix_decision < 0.25 and self.cfg.cutmix:
                images, labels = cutmix(images, labels, 1.)
            elif mix_decision >= 0.25 and mix_decision < 0.5 and self.cfg.fmix:
                images, labels = fmix(images, labels, alpha=1., decay_power=5., shape=(512,512), device=self.device)

            if self.device == 'GPU':
                with autocast():
                    y_preds = self.model(images.float())
                    
                    if mix_decision < 0.50 and (self.cfg.fmix or self.cfg.cutmix):
                        loss = self.criterion(y_preds, labels[0]) * labels[2] + self.criterion(y_preds, labels[1]) * (1. - labels[2])
                    else:
                        loss = self.criterion(y_preds, labels)
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
                        global_step += 1
                            
            elif self.device == 'TPU':
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
            if self.device == 'GPU':
                if step % self.cfg.print_freq == 0 or step == (len(train_loader)-1):
                    print('Epoch: [{0}][{1}/{2}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Elapsed {remain:s} '
                        'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                        'Grad: {grad_norm:.4f}  '
                        'Fold: {fold:4f}'
                        #'LR: {lr:.6f}  '
                        .format(
                        self.epoch+1, step, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses,
                        remain=timeSince(start, float(step+1)/len(train_loader)),
                        grad_norm=grad_norm,
                        fold=fold
                        #lr=scheduler.get_lr()[0],
                        ))
            elif self.device == 'TPU':
                if step % self.cfg.print_freq == 0 or step == (len(train_loader)-1):
                    self.xm.master_print('Epoch: [{0}][{1}/{2}] '
                                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                                    'Elapsed {remain:s} '
                                    'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                                    'Grad: {grad_norm:.4f}  '
                                    'Fold: {fold:4f}'
                                    #'LR: {lr:.6f}  '
                                    .format(
                                    self.epoch+1, step, len(train_loader), batch_time=batch_time,
                                    data_time=data_time, loss=losses,
                                    remain=timeSince(start, float(step+1)/len(train_loader)),
                                    grad_norm=grad_norm,
                                    fold=fold
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
            if self.device == 'GPU':
                if step % self.cfg.print_freq == 0 or step == (len(valid_loader)-1):
                    print('EVAL: [{0}/{1}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Elapsed {remain:s} '
                        'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                        'Fold: {fold:4f}'
                        .format(
                        step, len(valid_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses,
                        remain=timeSince(start, float(step+1)/len(valid_loader)),
                        fold=fold
                        ))
            elif self.device == 'TPU':
                if step % self.cfg.print_freq == 0 or step == (len(valid_loader)-1):
                    self.xm.master_print('EVAL: [{0}/{1}] '
                                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                                    'Elapsed {remain:s} '
                                    'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                                    'Fold: {fold:4f}'
                                    .format(
                                    step, len(valid_loader), batch_time=batch_time,
                                    data_time=data_time, loss=losses,
                                    remain=timeSince(start, float(step+1)/len(valid_loader)),
                                    fold=fold
                                    ))

        trues = np.concatenate(trues)
        predictions = np.concatenate(preds)
        return losses.avg, predictions, trues 