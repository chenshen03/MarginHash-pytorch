import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.distance import distance
import torch.nn.functional as F


def pairwise_loss(output, label, alpha=10.0, class_num=5.0, l_threshold=15.0):
    '''https://github.com/thuml/HashNet/issues/27#issuecomment-494265209'''
    bits = output.shape[1]
    similarity = Variable(torch.mm(label.data.float(), label.data.float().t()) > 0).float()
    dot_product = alpha * torch.mm(output, output.t()) / bits
    mask_dot = dot_product.data > l_threshold
    mask_exp = dot_product.data <= l_threshold
    mask_positive = similarity.data > 0
    mask_negative = similarity.data <= 0
    mask_dp = mask_dot & mask_positive
    mask_dn = mask_dot & mask_negative
    mask_ep = mask_exp & mask_positive
    mask_en = mask_exp & mask_negative

    dot_loss = dot_product * (1-similarity)
    exp_loss = torch.log(1+torch.exp(dot_product)) - similarity * dot_product
    loss = (torch.sum(torch.masked_select(exp_loss, Variable(mask_ep))) + torch.sum(torch.masked_select(dot_loss, Variable(mask_dp)))) * class_num + \
            torch.sum(torch.masked_select(exp_loss, Variable(mask_en))) + torch.sum(torch.masked_select(dot_loss, Variable(mask_dn)))

    return loss / (torch.sum(mask_positive.float()) * class_num + torch.sum(mask_negative.float()))


def contrastive_loss(output, label, margin=16):
    '''contrastive loss
    - Deep Supervised Hashing for Fast Image Retrieval
    '''
    batch_size = output.shape[0]
    S =  torch.mm(label.float(), label.float().t())
    dist = distance(output, dist_type='euclidean2')
    loss_1 = S * dist + (1 - S) * torch.max(margin - dist, torch.zeros_like(dist))
    loss = torch.sum(loss_1) / (batch_size*(batch_size-1))
    return loss


def exp_loss(output, label, wordvec=None, alpha=5.0, balanced=False):
    '''exponential loss
    '''
    batch_size, bit = output.shape
    mask = (torch.eye(batch_size) == 0).to(torch.device("cuda"))

    S =  torch.mm(label.float(), label.float().t())
    S_m = torch.masked_select(S, mask)

    wordvec_u = torch.mm(label.float(), wordvec)
    W = distance(wordvec_u, dist_type='cosine')
    W_m = torch.masked_select(W, mask)


    ## inner product
    # balance = True
    # ip = torch.mm(output, output.t()) / 32
    # ip = F.linear(F.normalize(output), F.normalize(output))
    # ip_m = torch.masked_select(ip, mask)
    # loss_1 = (S_m - ip_m) ** 2


    ## sigmoid
    # D = distance(output, dist_type='cosine')
    # E = torch.log(1 + torch.exp(-alpha * (1-2*D)))
    # E_m = torch.masked_select(E, mask)
    # loss_1 = 10 * S_m * E_m + (1 - S_m) * (E_m - torch.log((torch.exp(E_m) - 1).clamp(1e-6)))


    ## baseline
    balanced = True
    alpha_1 = 8
    alpha_2 = 8
    m1 = 0
    m2 = 0
    scale = 1

    dot_product = torch.mm(output, output.t()) / 32

    E1 = torch.log(1 + torch.exp(-alpha_1 * (dot_product-m1)))
    E1_m = torch.masked_select(E1, mask)
    loss_s1 = scale * S_m * E1_m

    E2 = torch.log(1 + torch.exp(-alpha_2 * (dot_product-m2)))
    E2_m = torch.masked_select(E2, mask)
    loss_s0 = (1 - S_m) * (E2_m - torch.log((torch.exp(E2_m) - 1).clamp(1e-6)))

    loss_1 = loss_s1 + loss_s0
    # print(f'max:{dot_product.max().item():.4f} min:{dot_product.min().item():.4f}')
    print('loss_s1:{:.4f} loss_s0:{:.4f}'.format(loss_s1.sum().item(), loss_s0.sum().item()))
    

    ## hyper sigmoid
    # alpha = 9
    # belta = 20
    # gamma = 1.5
    # margin = 0.25
    # D = distance(output, dist_type='cosine')
    # E1 = torch.log(1 + torch.exp(-alpha * (1-gamma*2*D)))
    # E1_m = torch.masked_select(E1, mask)
    # loss_s1 = S_m * E1_m
    # E2 = torch.log(1 + torch.exp(-alpha * (1-gamma*2*(D-margin))))
    # E2_m = torch.masked_select(E2, mask)
    # loss_s0 = (1 - S_m) * (E2_m - torch.log((torch.exp(E2_m) - 1)).clamp(1e-6))
    # loss_1 = belta * loss_s1 + loss_s0


    ## margin hash
    # D = distance(output, dist_type='cosine')
    # E1 = torch.exp(2* D) - 1
    # E2 = torch.exp(2 * (1 - D)) - 1
    # E1_m = torch.masked_select(E1, mask)
    # E2_m = torch.masked_select(E2, mask)
    # loss_1 = S_m * E1_m + (1 - S_m) * E2_m


    if balanced:
        S_all = batch_size * (batch_size - 1)
        S_1 = torch.sum(S)
        balance_param = (S_all / S_1) * S + (1 - S)
        B_m= torch.masked_select(balance_param, mask)
        loss_1 = B_m * loss_1

    loss = torch.mean(loss_1)
    return loss


weights = torch.tensor([0.50857143, 0.61952381, 0.89038095, 0.70780952, 0.89171429,
                        0.85942857, 0.89714286, 0.9067619 , 0.8847619 , 0.85714286,
                        0.87914286, 0.9187619 , 0.92685714, 0.90457143, 0.904     ,
                        0.91561905, 0.92561905, 0.9272381 , 0.92457143, 0.91742857,
                        0.90780952]).cuda()

def hadamard_loss(output, label, hadamard):
    def rand_num():
        r = torch.round(torch.rand(1))
        return r if r == 1 else torch.tensor([-1])

    batch_size, bit = output.shape
    label = label.float()
    if torch.sum(label) == batch_size:
        hc = torch.mm(label, hadamard)
    else:
        # label *= weights
        hc = torch.mm(label, hadamard)
        # hc[hc>0] = 1
        # hc[hc<0] = -1
        # hc[hc==0] = rand_num()
    loss = (output - hc) ** 2
    # import pdb; pdb.set_trace()
    return loss.mean()


def quantization_loss(output):
    loss = torch.mean((torch.abs(output) - 1) ** 2)
    return loss


def balance_loss(output):
    '''balance loss

    Each bit should be half 0 and half 1.
    - Supervised Learning of Semantics-preserving Hashing via Deep Neural Networks for Large-scale Image Search
    '''
    H = torch.sign(output)
    H_mean = torch.mean(H, dim=0)
    loss = torch.mean(H_mean ** 2)
    return loss


def independence_loss(output):
    '''independence loss
    - Deep Triplet Quantization
    '''
    batch_size, bit = output.shape
    H = torch.sign(output)
    I = torch.eye(bit).cuda()
    loss = torch.mean((torch.mm(H.t(), H) / batch_size - I) ** 2)
    return loss
