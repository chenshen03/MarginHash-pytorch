from .center_loss import CenterLoss

import torch


def quantization_loss(output):
    loss = torch.mean((torch.abs(output) - 1) ** 2)
    return loss


