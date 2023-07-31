#!/usr/bin/env python3

import torch
import torch.nn as nn
from .OrientationNet import ONET
from .TranslationNet import TNET

class Criterion(nn.Module):
    def __init__(self, orientation='euler'):
        super(Criterion, self).__init__()
        self.orientation = orientation
    
    def _normalize_quaternion(self, q):
        magnitude = torch.norm(q)
        q = torch.div(q, magnitude)

        return q

    def forward(self, pred, gt):
        p_hat, p = pred[:3], gt[:3] # translation
        q_hat, q = pred[3:], gt[3:] # orientation(euler)
        
        if self.orientation == 'quaternion':
            q_hat = self._normalize_quaternion(q_hat)
        
        p_error = nn.SmoothL1Loss()(p, p_hat)
        q_error = nn.SmoothL1Loss()(q, q_hat)

        loss = (10.0 * p_error) + (100.0 * q_error)
        
        return loss
    
class FlowFlowNet(nn.Module):
    def __init__(self):
        super(FlowFlowNet, self).__init__()
        self.onet = ONET(fc_size=16384)
        self.tnet = TNET(fc_size=16384)
    
    def forward(self, x):
        translation = self.tnet(x)
        orientation = self.onet(x)

        pose = torch.cat((translation, orientation), dim=1) # (x, y, z, roll, pitch, yaw)
        return pose
