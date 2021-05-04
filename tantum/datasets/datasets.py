import torch 
import cv2 
import numpy as np

from torch.utils.data import Dataset
from tantum.datasets.data_utils import NO_LABEL


def encode_label(label):
    return NO_LABEL* (label +1)

def decode_label(label):
    return NO_LABEL * label -1

def split_relabel_data(np_labs, labels, label_per_class,
                        num_classes):
    """ Return the labeled indexes and unlabeled_indexes
    """
    labeled_idxs = []
    unlabed_idxs = []
    for id in range(num_classes):
        indexes = np.where(np_labs==id)[0]
        np.random.shuffle(indexes)
        labeled_idxs.extend(indexes[:label_per_class])
        unlabed_idxs.extend(indexes[label_per_class:])
    np.random.shuffle(labeled_idxs)
    np.random.shuffle(unlabed_idxs)
    ## relabel dataset
    for idx in unlabed_idxs:
        labels[idx] = encode_label(labels[idx])

    return labeled_idxs, unlabed_idxs



class TrainDataset(Dataset):
    
    def __init__(self, 
                 df, 
                 train_path, 
                 image_id,
                 label,
                 transform=None):

        self.df = df
        self.file_names = df[image_id].values
        self.labels = df[label].values
        self.transform = transform 
        self.train_path = train_path
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{self.train_path}/{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        label = torch.tensor(self.labels[idx]).long()
        return image, label