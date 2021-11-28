import timm 
import torch.nn as nn



class CustomResNext(nn.Module):
    
    def __init__(self, model_name='resnext50_32x4d', target_size=1, pretrained=False):
        super().__init__()
        
        self.model = timm.create_model(model_name=model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, target_size)
    
    def forward(self, x):
        x = self.model(x)
        return x 