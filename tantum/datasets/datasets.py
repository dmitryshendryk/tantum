import torch 
import cv2 
import numpy as np

from torch.utils.data import Dataset
from tantum.datasets.data_utils import NO_LABEL
from sklearn.model_selection import StratifiedKFold
import cv2
import torch
import torch.nn.functional as F

import numpy as np

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


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

def create_folds(train_df, n_folds):
    folds = train_df.copy()
    fold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for n, (train_index, val_index) in enumerate(fold.split(folds, folds['label'])):
        folds.loc[val_index, 'fold'] = int(n)
    folds['fold'] = folds['fold'].astype(int)
    return folds

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

class MnistDataset(Dataset):
    def __init__(self, X, y, transforms=None):
        super().__init__()
        self.X = X.reshape(-1, 28, 28).astype(np.float32)
        self.y = y
        self.transforms = transforms

    def __getitem__(self, index):
        image, target = self.X[index], self.y[index]
        image = np.stack([image] * 1, axis=-1)
        image /= 255.
        if self.transforms:
            image = self.transforms(image=image)['image']
        return {
            "image": image,
            "targets": torch.tensor(target, dtype=torch.long),
        }

    def __len__(self):
        return self.y.shape[0]


class ImageDataset:
    def __init__(
        self,
        image_paths,
        targets,
        augmentations=None,
        backend="pil",
        channel_first=True,
        grayscale=False,
    ):
        """
        :param image_paths: list of paths to images
        :param targets: numpy array
        :param augmentations: albumentations augmentations
        """
        self.image_paths = image_paths
        self.targets = targets
        self.augmentations = augmentations
        self.backend = backend
        self.channel_first = channel_first
        self.grayscale = grayscale

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        targets = self.targets[item]
        if self.backend == "pil":
            image = Image.open(self.image_paths[item])
            image = np.array(image)
            if self.augmentations is not None:
                augmented = self.augmentations(image=image)
                image = augmented["image"]
        elif self.backend == "cv2":
            if self.grayscale is False:
                image = cv2.imread(self.image_paths[item])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.imread(self.image_paths[item], cv2.IMREAD_GRAYSCALE)
            if self.augmentations is not None:
                augmented = self.augmentations(image=image)
                image = augmented["image"]
        else:
            raise Exception("Backend not implemented")
        if self.channel_first is True and self.grayscale is False:
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        image_tensor = torch.tensor(image)
        if self.grayscale:
            image_tensor = image_tensor.unsqueeze(0)
        return {
            "image": image_tensor,
            "targets": torch.tensor(targets),
        }