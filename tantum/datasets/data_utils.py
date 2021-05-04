
import torch 
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

NO_LABEL = -1

class DataSetWarpper(Dataset):
    """Enable dataset to output index of sample
    """
    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes

    def __getitem__(self, index):
        sample, label = self.dataset[index]
        return sample, label, index

    def __len__(self):
        return len(self.dataset)

class TransformTwice:

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

class TransformWeakStrong:

    def __init__(self, trans1, trans2):
        self.transform1 = trans1
        self.transform2 = trans2

    def __call__(self, inp):
        out1 = self.transform1(inp)
        out2 = self.transform2(inp)
        return out1, out2



def create_loaders_v2(trainset, evalset, label_idxs, unlab_idxs,
                      num_classes,
                      config):
    if config.data_twice: trainset.transforms = TransformTwice(trainset.transforms)
    if config.data_idxs: trainset = DataSetWarpper(trainset, num_classes)
    ## supervised batch loader
    label_sampler = SubsetRandomSampler(label_idxs)
    label_batch_sampler = BatchSampler(label_sampler, config.sup_batch_size,
                                       drop_last=True)
    label_loader = torch.utils.data.DataLoader(trainset,
                                          batch_sampler=label_batch_sampler,
                                          num_workers=config.num_workers,
                                          pin_memory=True)
    ## unsupervised batch loader
    if not config.label_exclude: unlab_idxs += label_idxs
    unlab_sampler = SubsetRandomSampler(unlab_idxs)
    unlab_batch_sampler = BatchSampler(unlab_sampler, config.usp_batch_size,
                                       drop_last=True)
    unlab_loader = torch.utils.data.DataLoader(trainset,
                                          batch_sampler=unlab_batch_sampler,
                                          num_workers=config.num_workers,
                                          pin_memory=True)
    ## test batch loader
    eval_loader = torch.utils.data.DataLoader(evalset,
                                           batch_size=config.sup_batch_size,
                                           shuffle=False,
                                           num_workers=2*config.num_workers,
                                           pin_memory=True,
                                           drop_last=False)
    return label_loader, unlab_loader, eval_loader
