import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from .datalist import ImageDataset


normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

class CIFAR(object):
    def __init__(self, batch_size, use_gpu, num_workers):
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        pin_memory = True if use_gpu else False

        trainset = torchvision.datasets.CIFAR10(root='./data/cifar', train=True, download=True, transform=train_transform)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        
        testset = torchvision.datasets.CIFAR10(root='./data/cifar', train=False, download=True, transform=test_transform)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.trainloader = trainloader
        self.testloader = testloader
        self.databaseloader = testloader
        self.num_classes = 10
        self.R = 50000


class CIFARS1(object):
    def __init__(self, batch_size, use_gpu, num_workers, tencrop=False):
        pin_memory = True if use_gpu else False

        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        trainset = ImageDataset('data/cifar-s1/train.txt', transform=train_transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        if tencrop:
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.TenCrop(224),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
            ])
        else:
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        testset = ImageDataset('data/cifar-s1/test.txt', transform=test_transform)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        databaseset = ImageDataset('data/cifar-s1/database.txt', transform=test_transform)
        databaseloader = torch.utils.data.DataLoader(
            databaseset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.trainloader = trainloader
        self.testloader = testloader
        self.databaseloader = databaseloader
        self.num_classes = 10
        self.R = 54000


class CIFARS2(object):
    def __init__(self, batch_size, use_gpu, num_workers, tencrop=False):
        pin_memory = True if use_gpu else False

        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        trainset = ImageDataset('data/cifar-s2/train.txt', transform=train_transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        if tencrop:
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.TenCrop(224),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
            ])
        else:
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
    
        testset = ImageDataset('data/cifar-s2/test.txt', transform=test_transform)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        databaseset = ImageDataset('data/cifar-s2/database.txt', transform=test_transform)
        databaseloader = torch.utils.data.DataLoader(
            databaseset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.trainloader = trainloader
        self.testloader = testloader
        self.databaseloader = databaseloader
        self.num_classes = 10
        self.R = 50000
