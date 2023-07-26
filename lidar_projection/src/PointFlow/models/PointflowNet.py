#!/usr/bin/env python3

import torch
import torch.nn as nn
from .OrientationNet import ONET
from .TranslationNet import TNET

class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
    
    def _normalize_quaternion(self, q):
        magnitude = torch.norm(q)
        q = torch.div(q, magnitude)

        return q

    def forward(self, pred, gt):
        p_hat, p = pred[:3], gt[:3] # translation
        q_hat, q = pred[3:], gt[3:] # orientation(quaternion)
        q = self._normalize_quaternion(q)
        
        p_error = torch.pow(torch.norm(p - p_hat), 2)
        q_error = torch.pow(torch.norm(q - q_hat), 2)

        loss = (p_error) + 100 * (q_error)
        
        return loss
    
class PointflowNet(nn.Module):
    def __init__(self, init_st=-5.0, init_sq=5.0):
        super(PointflowNet, self).__init__()
        # self.st = nn.Parameter(torch.Tensor([init_st]))
        # self.sq = nn.Parameter(torch.Tensor([init_sq]))

        self.onet = ONET(fc_size=131072)
        self.tnet = TNET(fc_size=24192)
    
    def forward(self, x):
        translation = self.tnet(x)
        orientation = self.onet(x)

        pose = torch.cat((translation, orientation), dim=1) # (x, y, z, qw, qx, qy, qz)
        return pose
