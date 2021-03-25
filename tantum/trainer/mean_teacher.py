from numpy.lib.function_base import select
import torch
from torch._C import set_flush_denormal 
import torch.nn.functional as F

import os 
import datetime
from pathlib import Path
from itertools import cycle
from collections import defaultdict
from tqdm.contrib import tzip
from tqdm import tqdm



from tantum.utils.loss import mse_with_softmax
from tantum.utils.ramps import exp_rampup
from tantum.datasets.datasets import decode_label
from tantum.utils.data_utils import NO_LABELS


class Trainer:

    def __init__(self, model, ema_model, optimizer, device, config) -> None:
        print("Mean Teacher")

        self.model      = model
        self.ema_model  = ema_model
        self.optimizer  = optimizer
        self.ce_loss    = torch.nn.CrossEntropyLoss(ignore_index=NO_LABELS)
        self.cons_loss  = mse_with_softmax #F.mse_loss
        self.save_dir  = '{}-{}_{}-{}_{}'.format(config.arch, config.model,
                          config.dataset, config.num_labels,
                          datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
        self.save_dir  = os.path.join(config.save_dir, self.save_dir)
        self.usp_weight  = config.usp_weight
        self.ema_decay   = config.ema_decay
        self.rampup      = exp_rampup(config.weight_rampup)
        self.save_freq   = config.save_freq
        self.print_freq  = config.print_freq
        self.device      = device
        self.global_step = 0
        self.epoch       = 0
    
    def train_iteration(self, label_loader, unlab_loader, print_freq):
        loop_info = defaultdict(list)
        batch_idx, label_n, unlab_n = 0, 0, 0
        for ((x1,_), label_y), ((u1,u2), unlab_y) in tqdm(zip(cycle(label_loader), unlab_loader), total=len(unlab_loader), desc ="Training" ):
            self.global_step += 1
            label_x, unlab_x1, unlab_x2 = x1.to(self.device), u1.to(self.device), u2.to(self.device)
            label_y, unlab_y = label_y.to(self.device), unlab_y.to(self.device)
            ##=== decode targets ===
            self.decode_targets(unlab_y)
            lbs, ubs = x1.size(0), u1.size(0)

            ##=== forward ===
            outputs = self.model(label_x)
            loss = self.ce_loss(outputs, label_y)
            loop_info['lloss'].append(loss.item())

           
            ##=== Semi-supervised Training ===
            ## update mean-teacher
            self.update_ema(self.model, self.ema_model, self.ema_decay, self.global_step)
            ## consistency loss
            unlab_outputs = self.model(unlab_x1)
            with torch.no_grad():
                ema_outputs = self.ema_model(unlab_x2)
                ema_outputs = ema_outputs.detach()
            cons_loss  = self.cons_loss(unlab_outputs, ema_outputs)
            cons_loss *= self.rampup(self.epoch)*self.usp_weight
            loss += cons_loss; loop_info['uloss'].append(cons_loss.item())

            ## backwark
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            ##=== log info ===
            batch_idx, label_n, unlab_n = batch_idx+1, label_n+lbs, unlab_n+ubs
            loop_info['lacc'].append(label_y.eq(outputs.max(1)[1]).float().sum().item())
            loop_info['uacc'].append(unlab_y.eq(unlab_outputs.max(1)[1]).float().sum().item())
            loop_info['u2acc'].append(unlab_y.eq(ema_outputs.max(1)[1]).float().sum().item())
            if print_freq>0 and (batch_idx%print_freq)==0:
                print(f"[train][{batch_idx:<3}]", self.gen_info(loop_info, lbs, ubs))
        print(f">>>[train]", self.gen_info(loop_info, label_n, unlab_n, False))
        return loop_info, label_n
    
    def test_iteration(self, data_loader, print_freq):
        loop_info = defaultdict(list)
        label_n, unlab_n = 0, 0 

        for batch_idx, (data, targets) in enumerate(data_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            lbs, ubs = data.size(0), -1 

            outputs = self.model(data)
            ema_outputs = self.ema_model(data)
            loss = self.ce_loss(outputs, ema_outputs)
            loop_info['lloss'].append(loss.item())

            ##=== log info ===
            label_n, unlab_n = label_n+lbs, unlab_n+ubs
            loop_info['lacc'].append(targets.eq(outputs.max(1)[1]).float().sum().item())
            loop_info['l2acc'].append(targets.eq(ema_outputs.max(1)[1]).float().sum().item())
            if print_freq>0 and (batch_idx%print_freq)==0:
                print(f"[test][{batch_idx:<3}]", self.gen_info(loop_info, lbs, ubs))
        print(f">>>[test]", self.gen_info(loop_info, label_n, unlab_n, False))
        return loop_info, label_n
    

    def train(self, label_loader, unlab_loader, print_freq=20):
        self.model.train()
        self.ema_model.train()
        with torch.enable_grad():
            return self.train_iteration(label_loader, unlab_loader, print_freq)
    
    def test(self, data_loader, print_freq=10):
        self.model.eval()
        self.ema_model.eval()
        with torch.no_grad():
            return self.test_iteration(data_loader, print_freq)
    
    def loop(self, epochs, label_data, unlab_data, test_data, scheduler=None):
        best_acc, n, best_info = 0., 0., None
        epochs_bar = tqdm(range(epochs), total=epochs, desc ="Epoch")
        for ep in epochs_bar:
            self.epoch = ep
            if scheduler is not None: scheduler.step()
            print("------ Training epochs: {} ------".format(ep))
            self.train(label_data, unlab_data, self.print_freq)
            print("------ Testing epochs: {} ------".format(ep))
            info, n = self.test(test_data, self.print_freq)
            acc     = sum(info['lacc'])/n
            if acc>best_acc: best_acc, best_info = acc, info
            ## save model
            if self.save_freq!=0 and (ep+1)%self.save_freq == 0:
                self.save(ep)
        print(f">>>[best]", self.gen_info(best_info, n, n, False))


    def update_ema(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step +1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1-alpha, param.data)

    
    def decode_targets(self, targets):
        label_mask = targets.ge(0)
        unlab_mask = targets.le(NO_LABELS)
        targets[unlab_mask] = decode_label(targets[unlab_mask])
        return label_mask, unlab_mask
    
    def gen_info(self, info, lbs, ubs, iteration=True):
        ret = []
        nums = {'l': lbs, 'u':ubs, 'a': lbs+ubs}
        for k, val in info.items():
            n = nums[k[0]]
            v = val[-1] if iteration else sum(val)
            s = f'{k}: {v/n:.3%}' if k[-1]=='c' else f'{k}: {v:.5f}'
            ret.append(s)
        return '\t'.join(ret)

    def save(self, epoch, **kwargs):
        if self.save_dir is not None:
            model_out_path = Path(self.save_dir)
            state = {"epoch": epoch,
                    "weight": self.model.state_dict()}
            if not model_out_path.exists():
                model_out_path.mkdir()
            save_target = model_out_path / "model_epoch_{}.pth".format(epoch)
            torch.save(state, save_target)
            print('==> save model to {}'.format(save_target))