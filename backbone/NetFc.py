import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import math


class AlexNetFc(nn.Module):
    def __init__(self, feat_dim):
        super(AlexNetFc, self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        self.features = model_alexnet.features
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier"+str(i), model_alexnet.classifier[i])
        self.feature_layers = nn.Sequential(self.features, self.classifier)
        self.hash_layer = nn.Linear(4096, feat_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x = self.hash_layer(x)
        x = self.activation(x)
        return x


class AlexNetFc2(nn.Module):
    def __init__(self, feat_dim):
        super(AlexNetFc2, self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        self.features = model_alexnet.features
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.dp1 = model_alexnet.classifier[0]
        self.fc1 = model_alexnet.classifier[1]
        self.relu1 = model_alexnet.classifier[2]
        self.dp2 = model_alexnet.classifier[3]
        self.fc2 = model_alexnet.classifier[4]
        self.relu2 = model_alexnet.classifier[5]

        self.feature_layers = nn.Sequential(self.features, self.fc1, self.fc2)

        self.hash_layer = nn.Linear(8192, feat_dim)
        self.activation = nn.Tanh()

        self.iter_num = 0
        self.step_size = 1000
        self.gamma = 0.005
        self.power = 0.5
        self.init_scale = 0.5
        self.scale = self.init_scale

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        fc1 = self.relu1(self.fc1(self.dp1(x)))
        fc2 = self.relu2(self.fc2(self.dp2(fc1)))
        fc_cat = torch.cat((fc1, fc2), dim=1)

        if self.training:
            self.iter_num += 1
        if self.iter_num % self.step_size==0:
            self.scale = self.init_scale * (math.pow((1.+self.gamma*self.iter_num), self.power))
            if self.training:
                print(f"tanh scale change to {self.scale:.5f}")
            
        y = self.activation(self.hash_layer(fc_cat))
        return y


class AlexNetFc3(nn.Module):
    def __init__(self, feat_dim):
        super(AlexNetFc3, self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        self.features = model_alexnet.features
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier"+str(i), model_alexnet.classifier[i])
        self.feature_layers = nn.Sequential(self.features, self.classifier)
        self.hash_layer = nn.Linear(4096, feat_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        fc1 = self.classifier(x)
        y = self.activation(self.hash_layer(fc1))
        return y, fc1


resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152} 
class ResNetFc(nn.Module):
    def __init__(self, name, feat_dim):
        super(ResNetFc, self).__init__()
        model_resnet = resnet_dict[name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        self.hash_layer = nn.Linear(model_resnet.fc.in_features, feat_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        x = self.hash_layer(x)
        x = self.activation(x)
        return x


vgg_dict = {"VGG11":models.vgg11, "VGG13":models.vgg13, "VGG16":models.vgg16, "VGG19":models.vgg19, "VGG11BN":models.vgg11_bn, "VGG13BN":models.vgg13_bn, "VGG16BN":models.vgg16_bn, "VGG19BN":models.vgg19_bn} 
class VGGFc(nn.Module):
    def __init__(self, name, feat_dim):
        super(VGGFc, self).__init__()
        model_vgg = vgg_dict[name](pretrained=True)
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
        self.feature_layers = nn.Sequential(self.features, self.classifier)
        self.hash_layer = nn.Linear(model_vgg.classifier[6].in_features, feat_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 25088)
        x = self.classifier(x)
        x = self.hash_layer(x)
        x = self.activation(x)
        return x

    def output_num(self):
        return self.__in_features
