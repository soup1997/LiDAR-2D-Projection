import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
import numpy as np
from typing import Optional, Tuple

class Criterion(nn.Module):
    def __init__(self, k=100.0):
        super(Criterion, self).__init__()
        self.mse = nn.MSELoss()
        self.k = k

    def forward(self, pred, gt):
        r_hat, r_gt = pred[:, :, :3], gt[:, :, :3]
        p_hat, p_gt = pred[:, :, 3:], gt[:, :, 3:]

        r_error = self.mse(r_gt, r_hat)
        p_error = self.mse(p_gt, p_hat)

        loss = p_error + (self.k * r_error)

        return loss


class DeepVO(nn.Module):
    def __init__(self, batchNorm=True):
        super(DeepVO,self).__init__()
        
        self.batchNorm = batchNorm
        self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
        self.rnn_hidden_size = 1000
        self.rnn_dropout_ratio = 0.2
        self.rnn_dropout_between = 0.2 

        self.conv1   = self.conv(self.batchNorm,   6,   64, kernel_size=7, stride=2, dropout=self.conv_dropout[0])
        self.conv2   = self.conv(self.batchNorm,  64,  128, kernel_size=5, stride=2, dropout=self.conv_dropout[1])
        self.conv3   = self.conv(self.batchNorm, 128,  256, kernel_size=5, stride=2, dropout=self.conv_dropout[2])
        self.conv3_1 = self.conv(self.batchNorm, 256,  256, kernel_size=3, stride=1, dropout=self.conv_dropout[3])
        self.conv4   = self.conv(self.batchNorm, 256,  512, kernel_size=3, stride=2, dropout=self.conv_dropout[4])
        self.conv4_1 = self.conv(self.batchNorm, 512,  512, kernel_size=3, stride=1, dropout=self.conv_dropout[5])
        self.conv5   = self.conv(self.batchNorm, 512,  512, kernel_size=3, stride=2, dropout=self.conv_dropout[6])
        self.conv5_1 = self.conv(self.batchNorm, 512,  512, kernel_size=3, stride=1, dropout=self.conv_dropout[7])
        self.conv6   = self.conv(self.batchNorm, 512,  1024, kernel_size=3, stride=2, dropout=self.conv_dropout[8])
        
        # Comput the shape based on diff image size
        __tmp = torch.zeros(1, 6, 64, 1200)
        __tmp = self.encode_image(__tmp)
        
        # RNN
        self.rnn = nn.LSTM(
                    input_size=int(np.prod(__tmp.size())), 
                    hidden_size=self.rnn_hidden_size, 
                    num_layers=2, 
                    dropout=self.rnn_dropout_between, 
                    batch_first=True)
        self.rnn_dropout = nn.Dropout(self.rnn_dropout_ratio)
        self.linear = nn.Linear(in_features=self.rnn_hidden_size, out_features=6)

        # Initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LSTM):
                # layer 1
                kaiming_normal_(m.weight_ih_l0)  #orthogonal_(m.weight_ih_l0)
                kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                # Set forget gate bias to 1 (remember)
                n = m.bias_hh_l0.size(0)
                start, end = n//4, n//2
                m.bias_hh_l0.data[start:end].fill_(1.)

                # layer 2
                kaiming_normal_(m.weight_ih_l1)  #orthogonal_(m.weight_ih_l1)
                kaiming_normal_(m.weight_hh_l1)
                m.bias_ih_l1.data.zero_()
                m.bias_hh_l1.data.zero_()
                n = m.bias_hh_l1.size(0)
                start, end = n//4, n//2
                m.bias_hh_l1.data[start:end].fill_(1.)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def conv(self, batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
        if batchNorm:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout)#, inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout)#, inplace=True)
            )
        
    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6
    
    def load_Flownet(self):
        pretrained_flownet = '/home/smeet/catkin_ws/src/LiDAR-Inertial-Odometry/lidar_projection/src/DeepVO/trained/flownets_EPE1.951.pth'
        pretrained_w = torch.load(pretrained_flownet, map_location='cpu')

        # Load FlowNet weights pretrained with FlyingChairs
        # NOTE: the pretrained model assumes image rgb values in range [-0.5, 0.5]
        # Use only conv-layer of FlowNet as CNN for DeepVO
        model_dict = self.state_dict()
        update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
        model_dict.update(update_dict)
        self.load_state_dict(model_dict)

    def weight_selfameters(self):
        return [selfam for name, selfam in self.named_selfameters() if 'weight' in name]

    def bias_selfameters(self):
        return [selfam for name, selfam in self.named_selfameters() if 'bias' in name]

    def forward(self, x, prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        # x: torch.Size([8, 7, 3, 64, 1200]) which is (batch, seq_len, channel, width, height)

        x = torch.cat((x[:, :-1, :, :, :], x[:, 1:, :, :, :]), dim=2) # torch.Size([8, 6, 6, 64, 1200])
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # CNN
        x = x.view(batch_size*seq_len, x.size(2), x.size(3), x.size(4))
        x = self.encode_image(x)
        x = x.view(batch_size, seq_len, -1)

        # RNN
        if prev is None:
            out, hc = self.rnn(x)
        else:
            out, hc = self.rnn(x, prev)

        out = self.rnn_dropout(out)
        pose = self.linear(out) # torch.Size([8, 6, 6])

        return pose