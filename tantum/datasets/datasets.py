import numpy as np
from torch.utils import data 
import torchvision as tv 
import torchvision.transforms as transforms


from tantum.utils.data_utils import NO_LABELS
from tantum.utils.data_utils import TransformWeakStrong as wstwice
import albumentations


load = {}

def register_dataset(dataset):
    def wrapper(f):
        load[dataset] = f
        return f 
    return wrapper

def encode_label(label):
    return NO_LABELS * (label + 1)

def decode_label(label):
    return NO_LABELS * label - 1 

def split_relabel_data(np_labels, labels, label_per_class, num_classes):

    labeled_idxs = []
    unlabeled_idxs = []

    for id in range(num_classes):
        indexes = np.where(np_labels==id)[0]
        np.random.shuffle(indexes)
        labeled_idxs.extend(indexes[:label_per_class])
        unlabeled_idxs.extend(indexes[label_per_class:])
    np.random.shuffle(labeled_idxs)
    np.random.shuffle(unlabeled_idxs)

    for idx in unlabeled_idxs:
        labels[idx] = encode_label(labels[idx])
    
    return labeled_idxs, unlabeled_idxs


@register_dataset('cifar10')
def wscifar10(n_labels, data_root='./data-local/cifar10/'):
    channels_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                          std=[0.2023, 0.1994, 0.2010])
    
    train_transform = transforms.Compose([
        transforms.Pad(2, padding_mode='reflect'),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.4, hue=0.1),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channels_stats)
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channels_stats)
    ])

    trainset = tv.datasets.CIFAR10(data_root, train=True, download=False, transform=train_transform)
    evalset = tv.datasets.CIFAR10(data_root, train=False, download=False, transform=eval_transform)

    num_classes = 10
    label_per_class = n_labels // num_classes
    labeled_idxs, unlabed_idxs = split_relabel_data(
                                    np.array(trainset.targets),
                                    trainset.targets,
                                    label_per_class,
                                    num_classes)
    return {
        'trainset': trainset,
        'evalset': evalset,
        'label_idxs': labeled_idxs,
        'unlab_idxs': unlabed_idxs,
        'num_classes': num_classes
    }