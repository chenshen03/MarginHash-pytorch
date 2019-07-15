from .mnist import MNIST
from .cifar10 import CIFAR, CIFARS1, CIFARS2


def create(name, batch_size):
    if name == 'cifar':
        dataset = CIFAR(batch_size)
    elif name == 'cifar-s1':
        dataset = CIFARS1(batch_size)
    elif name == 'cifar-s2':
        dataset = CIFARS2(batch_size)
    elif name == 'mnist':
        dataset = MNIST(batch_size)
    else:
        raise ValueError(f'dataset {name} is not available!')
    return dataset
