import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes, feat_dim, device='cuda'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        # self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))
        self.centers = nn.Parameter(torch.Tensor(self.num_classes, self.feat_dim).to(self.device))
        nn.init.xavier_uniform_(self.centers)


    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


class AgentCenterLoss(nn.Module):
    '''The variety of center loss, which use the class weight as the class center and normalize both the weight and feature,
       in this way, the cos distance of weight and feature can be used as the supervised signal.
       It's similar with torch.nn.CosineEmbeddingLoss, x_1 means weight_i, x_2 means feature_i.
    '''

    def __init__(self, num_classes, feat_dim, scale, use_gpu=True):
        super(AgentCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.scale = scale

        if use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        '''
        Parameters:
            x: input tensor with shape (batch_size, feat_dim)
            labels: ground truth label with shape (batch_size)
        Return:
            loss of centers
        '''
        cos_dis = F.linear(F.normalize(x), F.normalize(self.centers)) * self.scale

        one_hot = torch.zeros_like(cos_dis)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # loss = 1 - cosine(i)
        loss = one_hot * self.scale - (one_hot * cos_dis)

        return loss.mean()
