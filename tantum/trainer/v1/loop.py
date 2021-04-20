
import time 
import numpy as np


import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim import Adam

from tantum.trainer.v1.trainer import Trainer
from tantum.trainer.v1.evaluator import Evaluator
from tantum.datasets.datasets import TrainDataset
from tantum.scheduler.scheduler import  get_scheduler, GradualWarmupSchedulerV2
from tantum.utils.loss import get_criterion
from tantum.utils.logger import LOGGER
from tantum.utils.metrics import get_score

class MainLoop():
    def __init__(self, net, transformer = None) -> None:
        self.transformer = transformer
        self.net = net

        

    def fit(self, CFG, folds, fold, xm=None, pl=None, idist=None):
        
        LOGGER.info(f"========== fold: {fold} training ==========")
        
        # ====================================================
        # loader
        # ====================================================
        trn_idx = folds[folds['fold'] != fold].index
        val_idx = folds[folds['fold'] == fold].index

        train_folds = folds.loc[trn_idx].reset_index(drop=True)
        valid_folds = folds.loc[val_idx].reset_index(drop=True)
        
        train_dataset = TrainDataset(train_folds, CFG.TRAIN_PATH, 'image_id', 'label', transform=self.transformer(data='train'))
        valid_dataset = TrainDataset(valid_folds, CFG.TRAIN_PATH, 'image_id', 'label', transform=self.transformer(data='valid'))
        
        train_loader = DataLoader(train_dataset, 
                                batch_size=CFG.batch_size, 
                                shuffle=True, 
                                num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, 
                                batch_size=CFG.batch_size, 
                                shuffle=False, 
                                num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
        
        valid_labels = valid_folds[CFG.target_col].values
        
        # ====================================================
        # device
        # ====================================================
        if CFG.device == 'TPU':
            device = xm.xla_device(fold+1)
        elif CFG.device == 'GPU':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        # ====================================================
        # model
        # ====================================================
        ### Create model
        model = self.net(CFG.model_name, pretrained=True)
        model.to(device)
        
            
        # ====================================================
        # optimizer 
        # ====================================================
        ### Create optimizer
        optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
        
        # ====================================================
        # scheduler 
        # ====================================================
        ### Create scheduler
        scheduler = get_scheduler(CFG, optimizer)
        
        # ====================================================
        # criterion 
        # ====================================================
        ### Create criterion
        criterion = get_criterion(CFG, device)
        
        # ====================================================
        # Stochastik Weighted Average
        # ====================================================
        
        if CFG.swa:
            swa_model = AveragedModel(model)
            swa_scheduler = SWALR(optimizer, swa_lr=0.05)
        else:
            swa_model = None
        
        # ====================================================
        # Trainer and Evaluator
        # ====================================================
        trainer = Trainer(model, criterion, optimizer, scheduler)
        evaluator = Evaluator(model, criterion)
        
        
        best_score = 0.
        best_loss = np.inf
        
        for epoch in range(CFG.epochs):
            
            start_time = time.time()
            
            # ====================================================
            # train
            # ====================================================
            if CFG.device == 'TPU':
                if CFG.nprocs == 1:
                    avg_loss = trainer.fit(CFG, train_loader, epoch, device, fold)
                elif CFG.nprocs > 1:
                    para_train_loader = pl.ParallelLoader(train_loader, [device])
                    avg_loss = trainer.fit(CFG, para_train_loader.per_device_loader(device), epoch, device, fold)
            elif CFG.device == 'GPU':
                    avg_loss = trainer.fit(CFG, train_loader, epoch, device, fold)
            
            # ====================================================
            # eval baseline
            # ====================================================
            if CFG.device == 'TPU':
                if CFG.nprocs == 1:
                    avg_val_loss, preds, _ = evaluator.fit(CFG, valid_loader, device, fold)
                elif CFG.nprocs > 1:
                    para_valid_loader = pl.ParallelLoader(valid_loader, [device])
                    avg_val_loss, preds, valid_labels = evaluator.fit(CFG, para_valid_loader.per_device_loader(device),  device, fold)
                    preds = idist.all_gather(torch.tensor(preds)).to('cpu').numpy()
                    valid_labels = idist.all_gather(torch.tensor(valid_labels)).to('cpu').numpy()
            elif CFG.device == 'GPU':
                    avg_val_loss, preds, _ = evaluator.fit(CFG, valid_loader, device, fold)
            
            
            
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            elif isinstance(scheduler, CosineAnnealingLR):
                scheduler.step()
            elif isinstance(scheduler, CosineAnnealingWarmRestarts):
                scheduler.step()
            elif isinstance(scheduler, GradualWarmupSchedulerV2):
                if epoch > CFG.swa_start and CFG.swa:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()
                else:
                    scheduler.step(epoch)

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
                    xm.master_print(f'Epoch {epoch+1} - Fold {fold} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
                    xm.master_print(f'Epoch {epoch+1} - Fold {fold} - Score: {score:.4f}')
            
            if score > best_score:
                best_score = score
                if CFG.device == 'GPU':
                    LOGGER.info(f'Epoch {epoch+1} - Fold {fold} - Save Best Score: {best_score:.4f} Model')
                    torch.save({'model': model.state_dict(), 
                                'preds': preds},
                            CFG.OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_score.pth')
                elif CFG.device == 'TPU':
                    if CFG.nprocs == 1:
                        LOGGER.info(f'Epoch {epoch+1} - Fold {fold} - Save Best Score: {best_score:.4f} Model')
                    elif CFG.nprocs > 1:
                        xm.master_print(f'Epoch {epoch+1} - Fold {fold} - Save Best Score: {best_score:.4f} Model')
                    xm.save({'model': model, 
                            'preds': preds}, 
                            CFG.OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_score.pth')
        
        # ====================================================
        # Update bn statistics for the swa_model
        # ====================================================
        if CFG.swa:
            torch.cuda.empty_cache()
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        
        #==========================
        # Compare SWA and Baseline
        #==========================
        
        if CFG.swa:
            avg_val_loss_baseline, preds_baseline, _ = evaluator.fit(CFG, valid_loader, device, fold)
            score_baseline = get_score(valid_labels, preds_baseline.argmax(1))

            evaluator_swa = Evaluator(swa_model, criterion)
            avg_val_loss_swa, preds_swa, _ = evaluator_swa.fit(CFG, valid_loader, device, fold)
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