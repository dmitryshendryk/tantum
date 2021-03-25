
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.data import DataLoader 


from tantum.utils.data_utils import DataSetWrapper
from tantum.utils.data_utils import TwoStreamBatchSampler
from tantum.utils.data_utils import TransformTwice as twice

def create_loaders_v2(trainset, evalset, label_idxs, unlab_idxs, num_classes, config):
     
    if config.data_twice: trainset.transform = twice(trainset.transform)
    if config.data_idxs: trainset = DataSetWrapper(trainset, num_classes)

    label_sampler = SubsetRandomSampler(label_idxs)
    label_batch_sampler = BatchSampler(label_sampler, config.sup_batch_size, drop_last=True)

    label_loader = DataLoader(trainset,
                            batch_sampler=label_batch_sampler,
                            num_workers=config.workers,
                            pin_memory=True)
    
    if not config.label_exclude: unlab_idxs += label_idxs
    unlab_sampler = SubsetRandomSampler(unlab_idxs)
    unlab_batch_sampler = BatchSampler(unlab_sampler, config.usp_batch_size, drop_last=True)

    unlab_loader = DataLoader(trainset,
                              batch_sampler=unlab_batch_sampler,
                              num_workers=config.workers,
                              pin_memory=True)
    
    eval_loader = DataLoader(evalset,
                            batch_size=config.sup_batch_size,
                            shuffle=False,
                            num_workers=2*config.workers,
                            pin_memory=True,
                            drop_last=False)
    
    return label_loader, unlab_loader, eval_loader
                
    