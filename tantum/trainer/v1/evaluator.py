import time
import numpy as np

import torch
import torch_xla.core.xla_model as xm

from tantum.utils.metrics import AverageMeter, timeSince




class Evaluator():

    def __init__(self, model, criterion) -> None:
        
        self.model = model 
        self.criterion = criterion


    def fit(self, cfg, valid_loader, device, fold):
        
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
            images = images.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)
            
            # compute loss
            with torch.no_grad():
                y_preds = self.model(images)
            loss = self.criterion(y_preds, labels)
            losses.update(loss.item(), batch_size)
            # record accuracy
            trues.append(labels.to('cpu').numpy())
            preds.append(y_preds.softmax(1).to('cpu').numpy())
            if cfg.gradient_accumulation_steps > 1:
                loss = loss / cfg.gradient_accumulation_steps
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if cfg.device == 'GPU':
                if step % cfg.print_freq == 0 or step == (len(valid_loader)-1):
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
            elif cfg.device == 'TPU':
                if step % cfg.print_freq == 0 or step == (len(valid_loader)-1):
                    xm.master_print('EVAL: [{0}/{1}] '
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