from .AlexNet import *
from .ConvNet import *
from .VGG import *
from .ResNet import *
from .LeNet import *
from .DenseNet import *
from .GoogleNet import *
from .WideResNet import *


def create(name, feat_dim=32):
    if name == 'AlexNet':
        net = AlexNet(feat_dim=feat_dim)
    elif name == 'ConvNet':
        net = ConvNet(feat_dim=feat_dim)
    else:
        raise ValueError(f'model {name} is not available!')
    return net