import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from .datalist import ImageDataset


normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class ImageNet(object):
    def __init__(self, batch_size, bit=32, tencrop=False):
        root_path = '/media/disk1/chenshen/datasets/ImageNet'
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        trainset = ImageDataset('data/imagenet/train.txt', transform=train_transform, root_path=root_path)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True,
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
        testset = ImageDataset('data/imagenet/test.txt', transform=test_transform, root_path=root_path)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )

        databaseset = ImageDataset('data/imagenet/database.txt', transform=test_transform, root_path=root_path)
        databaseloader = torch.utils.data.DataLoader(
            databaseset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )

        self.trainloader = trainloader
        self.testloader = testloader
        self.databaseloader = databaseloader
        self.num_classes = 100
        self.R = 1000
