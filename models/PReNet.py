import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn import Parameter
 

class PReNet(nn.Module):
    
    def __init__(self, recurrent_iter = 6, use_gpu = True, num_channels = 32):

        super(PReNet, self).__init__()
        
        self.iteration = recurrent_iter
        self.use_gpu = use_gpu
        self.ncs = num_channels

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, self.ncs, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(self.ncs, self.ncs, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.ncs, self.ncs, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(self.ncs, self.ncs, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.ncs, self.ncs, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(self.ncs, self.ncs, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.ncs, self.ncs, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(self.ncs, self.ncs, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.ncs, self.ncs, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(self.ncs, self.ncs, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.ncs, self.ncs, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv6 = nn.Sequential(
            nn.Conv2d(self.ncs, self.ncs, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.ncs, self.ncs, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv7 = nn.Sequential(
            nn.Conv2d(self.ncs, self.ncs, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.ncs, self.ncs, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv8 = nn.Sequential(
            nn.Conv2d(self.ncs, self.ncs, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.ncs, self.ncs, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv9 = nn.Sequential(
            nn.Conv2d(self.ncs, self.ncs, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.ncs, self.ncs, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv10 = nn.Sequential(
            nn.Conv2d(self.ncs, self.ncs, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.ncs, self.ncs, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(self.ncs + self.ncs, self.ncs, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(self.ncs + self.ncs, self.ncs, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(self.ncs + self.ncs, self.ncs, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(self.ncs + self.ncs, self.ncs, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(self.ncs, 3, 3, 1, 1),
            )
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 1, 3, 1, 1),
            )
    def forward(self, inputs):

        batch_size, row, col = inputs.size(0), inputs.size(2), inputs.size(3)

        x = inputs
        h = Variable(torch.zeros(batch_size, self.ncs, row, col))
        c = Variable(torch.zeros(batch_size, self.ncs, row, col))

        if self.use_gpu:
            h = h.cuda()
            c = c.cuda()

        for i in range(self.iteration):
            x = torch.cat((inputs, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            resx = x
            x = F.relu(self.res_conv6(x) + resx)
            resx = x
            x = F.relu(self.res_conv7(x) + resx)
            resx = x
            x = F.relu(self.res_conv8(x) + resx)
            resx = x
            x = F.relu(self.res_conv9(x) + resx)
            resx = x
            x = F.relu(self.res_conv10(x) + resx)
            x = self.conv(x)

            x = x + inputs

        x = self.conv1(x)
        return x