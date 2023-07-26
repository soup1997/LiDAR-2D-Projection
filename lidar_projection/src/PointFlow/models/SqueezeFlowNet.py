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

    def forward(self, pred, gt, t_coeff, o_coeff):
        p_hat, p = pred[:3], gt[:3] # translation
        q_hat, q = pred[3:], gt[3:] # orientation(quaternion)
        q = self._normalize_quaternion(q)
        
        p_error = torch.pow(torch.norm(p - p_hat), 2)
        q_error = torch.pow(torch.norm(q - q_hat), 2)

        loss = (t_coeff * p_error) + (o_coeff * q_error)
        
        return loss
    
class SqueezeFlowNet(nn.Module):
    def __init__(self, init_t_coeff=20.0, init_o_coeff=50.0):
        super(SqueezeFlowNet, self).__init__()

        self.t_coeff = nn.Parameter(torch.Tensor([init_t_coeff]))
        self.o_coeff = nn.Parameter(torch.Tensor([init_o_coeff]))

        self.onet = ONET()
        self.tnet = TNET(fc_size=131072)
    
    def forward(self, x):
        translation = self.tnet(x)
        orientation = self.onet(x)

        pose = torch.cat((translation, orientation), dim=1) # (x, y, z, roll, pitch, yaw)
        return pose
