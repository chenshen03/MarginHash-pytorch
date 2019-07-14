from .cifar10 import CIFAR, CIFARS1, CIFARS2
from .mnist import MNIST


__factory = {
    'mnist': MNIST,
    'cifar': CIFAR,
    'cifar-s1': CIFARS1,
    'cifar-s2': CIFARS2,
}


def create(name, batch_size, use_gpu, num_workers):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](batch_size, use_gpu, num_workers)
