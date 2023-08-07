import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np

class DeepVO(nn.Module):
    def __init__(self, batchNorm=True):
        super(DeepVO, self).__init__()
        # CNN
        self.batchNorm = batchNorm
        self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)

        self.conv1   = self.conv(self.batchNorm,   6,   64, kernel_size=7, stride=2, dropout=self.conv_dropout[0])
        self.conv2   = self.conv(self.batchNorm,  64,  128, kernel_size=5, stride=2, dropout=self.conv_dropout[1])
        self.conv3   = self.conv(self.batchNorm, 128,  256, kernel_size=5, stride=2, dropout=self.conv_dropout[2])
        self.conv3_1 = self.conv(self.batchNorm, 256,  256, kernel_size=3, stride=1, dropout=self.conv_dropout[3])
        self.conv4   = self.conv(self.batchNorm, 256,  512, kernel_size=3, stride=2, dropout=self.conv_dropout[4])
        self.conv4_1 = self.conv(self.batchNorm, 512,  512, kernel_size=3, stride=1, dropout=self.conv_dropout[5])
        self.conv5   = self.conv(self.batchNorm, 512,  512, kernel_size=3, stride=2, dropout=self.conv_dropout[6])
        self.conv5_1 = self.conv(self.batchNorm, 512,  512, kernel_size=3, stride=1, dropout=self.conv_dropout[7])
        self.conv6   = self.conv(self.batchNorm, 512, 1024, kernel_size=3, stride=2, dropout=self.conv_dropout[8])

        # Comput the shape based on diff image size
        __tmp = torch.zeros(1, 6, 64, 1800)
        __tmp = self.encode_image(__tmp)

        # RNN
        self.rnn = nn.LSTM(input_size=int(np.prod(__tmp.size())), hidden_size=1000, num_layers=2, dropout=0, batch_first=True)
        self.rnn_drop_out = nn.Dropout(0.5)
        self.linear = nn.Linear(in_features=1000, out_features=6)

        # Initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.LSTM):
                # layer 1
                kaiming_normal_(m.weight_ih_l0)  # orthogonal_(m.weight_ih_l0)
                kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                # Set forget gate bias to 1 (remember)
                n = m.bias_hh_l0.size(0)
                start, end = n//4, n//2
                m.bias_hh_l0.data[start:end].fill_(1.)

                # layer 2
                kaiming_normal_(m.weight_ih_l1)  # orthogonal_(m.weight_ih_l1)
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
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                          stride=stride, padding=(kernel_size-1)//2, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout)  # , inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                          stride=stride, padding=(kernel_size-1)//2, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout)  # , inplace=True)
            )

    def forward(self, x):
        split_idx = x.size(2) // 2
        img1 = x[:, :, :split_idx, :]
        img2 = x[:, :, split_idx:, :]

        x = torch.cat((img1, img2), dim=1)
        x = self.encode_image(x)  # CNN
        x = x.view(x.size(0), 1, -1)

        # RNN
        x, _ = self.rnn(x)
        x = self.rnn_drop_out(x)
        x = self.linear(x)
        x = x.view(x.size(0), -1)

        return x

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))

        return out_conv5

    def weight_selfameters(self):
        return [selfam for name, selfam in self.named_selfameters() if 'weight' in name]

    def bias_selfameters(self):
        return [selfam for name, selfam in self.named_selfameters() if 'bias' in name]
