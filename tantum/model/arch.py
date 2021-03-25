from functools import wraps

from tantum.model.resnet import ResNet


arch = {
    'resnet18': ResNet
}



def RegisterArch(arch_name):
    """Register a model
    you must import the file where using this decorator
    for register the model function
    """
    def warpper(f):
        arch[arch_name] = f
        return f
    return warpper