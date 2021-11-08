from tantum.model.simple_net import Net
from tantum.model.vit_attention import VIT_Attention



def get_model(config):
    if config.model_type == 'Net':
        model = Net
    elif config.model_type == 'VitAttention':
        model = VIT_Attention
    
    return model 
        
