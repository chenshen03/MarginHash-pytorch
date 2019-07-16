import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x, label):
        return x
