

import torch
import numpy as np

class MnistDataset:
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