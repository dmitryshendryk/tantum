import torch


class GenericDataset():
    
    def __init__(self, data, targets, dtypes, transform):
        self.data = data 
        self.targets = targets
        self.dtypes = dtypes 
        self.transform = transform 
    
    def __len__(self):
        return len(self.data) 
    
    def __getitem__(self, idx):
        data = self.data[idx]
        targets = self.targets[idx]
        return {
            'x': torch.tensor(data, dtype=self.dtypes[0]),
            'y': torch.tensor(targets, dtype=self.dtypes[1])
        }