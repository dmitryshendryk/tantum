
import torch.nn as nn
from torchvision import datasets, models, transforms




class ResNet(nn.Module):
    
    def __init__(self, model_name='resnet18', target_size=10, pretrained=False) -> None:
        super().__init__()
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            features = self.model.fc.in_features
            self.model.fc = nn.Linear(features, target_size)

    def forward(self, x):
        x = self.model(x)
        return x

