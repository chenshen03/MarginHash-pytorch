from .AlexNet import *
from .VGG import *
from .ResNet import *
from .LeNet import *
from .DenseNet import *
from .GoogleNet import *
from .WideResNet import *


__factory = {
    'cnn': ConvNet,
    'alexnet': AlexNet,
}


def create(name, num_classes, feat_dim=32):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](num_classes, feat_dim)