import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np


class AirFace(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
        """

    def __init__(self, in_features, out_features, s=10.0, m=0.2):
        super(AirFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.Li = lambda x: (math.pi - 2 * x) / math.pi

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        theta = cosine.data.acos()
        logit1 = self.Li(theta + self.m)
        logit2 = self.Li(theta)
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * logit1) + ((1.0 - one_hot) * logit2)
        output *= self.s
        return output
