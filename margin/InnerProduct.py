import torch
import torch.nn as nn
import torch.nn.functional as F


class InnerProduct(nn.Module):
    def __init__(self, in_feature, out_feature, bias=True):
        super(InnerProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.fc = torch.nn.Linear(in_feature, out_feature, bias=bias)

    def forward(self, x, label):
        # label not used
        output = self.fc(x)
        return output


if __name__ == '__main__':
    pass