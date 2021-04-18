import torch 
import cv2 

from torch.utils.data import Dataset
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