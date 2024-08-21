import torch.nn as nn
import torch

class RankNetLoss(nn.Module):
    """
        https://github.com/szdr/RankNet/
        From RankNet to LambdaRank to LambdaMART: An Overview
        (https://www.microsoft.com/en-us/research/uploads/prod/2016/02/MSR-TR-2010-82.pdf)
    """
    def __init__(self, ):
        super(RankNetLoss, self).__init__()

    def __call__(self, s_i, s_j, t_i, t_j):
        s_diff = s_i - s_j
        if t_i > t_j:
            S_ij = 1
        elif t_i < t_j:
            S_ij = -1
        else:
            S_ij = 0
        loss = (1 - S_ij) / 2. * torch.sigmoid(s_diff)  + torch.log(1 + torch.exp(-torch.sigmoid(s_diff)))
        return loss