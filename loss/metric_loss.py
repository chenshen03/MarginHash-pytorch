import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.distance import distance
import torch.nn.functional as F

from utils.evaluation import get_RAMAP, get_mAP
from utils.tool import sign


def map_loss(output, label, k=100):
    output = output.data.cpu().numpy()
    label = label.data.cpu().numpy()
    batch_size, bit = output.shape
    RAMAP = 0
    for i in range(batch_size):
        q_output = output[i].reshape(1, -1)
        q_label = label[i].reshape(1, -1)
        db_output = np.delete(output, i, 0)
        db_label = np.delete(label, i, 0)
        RAMAP += get_mAP(sign(db_output), db_label, sign(q_output), q_label, R=batch_size-1)
    RAMAP /= batch_size
    loss = 1-RAMAP
    return loss
