import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class AlexNet(nn.Module):
    def __init__(self, feat_dim):
        super(AlexNet, self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        self.features = model_alexnet.features
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier"+str(i), model_alexnet.classifier[i])
        self.feature_layers = nn.Sequential(self.features, self.classifier)
        self.fc = nn.Linear(4096, feat_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x = self.fc(x)
        # x = self.activation(x)
        return x
