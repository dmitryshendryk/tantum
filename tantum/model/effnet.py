import timm 
import torch.nn as nn



class EffNet(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b5_ns', target_size=1, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, target_size)

    def forward(self, x):
        x = self.model(x)
        return x