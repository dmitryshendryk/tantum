import torch
import torch.nn as nn
from tqdm.auto import tqdm

from tantum.utils import AverageMeter




import torch
import numpy as np

class GenericDataset:
    def __init__(self, data, targets, dtypes, transform):
        self.data = data
        self.targets = targets
        self.dtypes = dtypes
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = np.asarray(data).astype(np.uint8).reshape(28, 28, 1)

        if self.transform:
            data = self.transform(data)
            
        targets = self.targets[idx]
        return {
            "x": data.view(-1,28*28),
            "y": torch.tensor([targets], dtype=self.dtypes[1]),
        }

class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.test_loader = None
        self.current_train_step = 0
        self.current_valid_step = 0
        self.metrics = {}
        self.metrics["train"] = {}
        self.metrics["valid"] = {}
        self.metrics["test"] = {}

    def forward(self, *args, **kwargs):
        super().forward(*args, **kwargs)

    def fetch_optimizer(self, *args, **kwargs):
        return

    def fetch_scheduler(self, *args, **kwargs):
        return

    def monitor_metrics(self, *args, **kwargs):
        return

    def loss(self, *args, **kwargs):
        return

    def update_metrics(self, losses, monitor):
        self.metrics[self._model_state.value].update(monitor)
        self.metrics[self._model_state.value]["loss"] = losses.avg

    def train_one_step(self, data, device):
        self.optimizer.zero_grad()

        for k, v in data.items():
            data[k] = v.to(device)
        _, loss, metrics = self(**data)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss, metrics

    def train_one_epoch(self, data_loader, device):
        self.train()
        epoch_loss = 0
        losses = AverageMeter()
        tk0 = tqdm(data_loader, total=len(data_loader))
        for b_idx, data in enumerate(tk0):
            loss, metrics = self.train_one_step(data, device)
            epoch_loss += loss
            losses.update(loss.item(), data_loader.batch_size)
            if b_idx == 0:
                metrics_meter = {k: AverageMeter() for k in metrics}
            monitor = {}
            for m_m in metrics_meter:
                metrics_meter[m_m].update(metrics[m_m], data_loader.batch_size)
                monitor[m_m] = metrics_meter[m_m].avg
            self.current_train_step += 1
            tk0.set_postfix(loss=losses.avg, stage="train", **monitor)
        return losses.avg

    def fit(self,
            train_dataset,
            valid_dataset,
            train_bs,
            valid_bs,
            epochs,
            device='cpu'
            ):

        if self.train_loader is None:
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=train_bs,
                shuffle=True
            )
        if self.test_loader is None:
            self.test_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=valid_bs,
                shuffle=True
            )

        if next(self.parameters()).device != device:
            self.to(device)

        self.optimizer = self.fetch_optimizer()
        self.scheduler = self.fetch_scheduler()

        for _ in range(epochs):
            train_loss = self.train_one_epoch(self.train_loader, device)
