from .mnist import MNIST
from .cifar10 import CIFAR, CIFARS1, CIFARS2
from .nuswide import *


def create(name, batch_size):
    if name == 'mnist':
        dataset = MNIST(batch_size)
    elif name == 'cifar':
        dataset = CIFAR(batch_size)
    elif name == 'cifar_s1':
        dataset = CIFARS1(batch_size)
    elif name == 'cifar_s2':
        dataset = CIFARS2(batch_size)
    elif name == 'nuswide_21':
        dataset = NUSWIDE21(batch_size)
    elif name == 'nuswide_81':
        dataset = NUSWIDE81(batch_size)
    else:
        raise ValueError(f'dataset {name} is not available!')
    return dataset
