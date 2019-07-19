from .NetFc import *
from .ConvNet import *
from .VGG import *
from .ResNet import *
from .LeNet import *
from .DenseNet import *
from .GoogleNet import *
from .WideResNet import *


def create(name, feat_dim=32):
    if 'AlexNet' in name:
        return AlexNetFc(feat_dim)
    elif 'ResNet' in name:
        return ResNetFc(name, feat_dim)
    elif 'VGG' in name:
        return VGGFc(name, feat_dim)
    else:
        raise ValueError(f'model {name} is not available!')
