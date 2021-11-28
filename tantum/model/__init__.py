from tantum.model.simple_net import Net
from tantum.model.vit_attention import VIT_Attention
from tantum.model.resnext import CustomResNext
from tantum.model.effnet import EffNet



def get_model(config):
    if config.model_type == 'Net':
        model = Net
    elif config.model_type == 'VitAttention':
        model = VIT_Attention
    elif config.model_type == 'ResNext':
        model = CustomResNext
    elif config.model_type == 'EffNet':
        model = EffNet
    
    return model 
        
