import math
import torch
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t

class Kerv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        gamma: int = 1,
        balance: int = 1, 
        power: int = 3,
        bias: bool = True,
        learnable_kernel=False,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ): 
        super(Kerv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        assert self.kernel_size[0] == 1 and self.kernel_size[1] == 1
        self.gamma = gamma
        self.balance = balance
        self.power = power
        if learnable_kernel == True:
            #self.gamma = nn.Parameter(torch.Tensor([gamma] * out_channels), requires_grad=True)
            self.balance = nn.Parameter(torch.Tensor([balance] * out_channels), requires_grad=True)

    def forward(self, input):
        batch_size, in_channels, height, width = input.size()
        input_unfold = input.view(batch_size, 1, self.kernel_size[0]*self.kernel_size[1]*self.in_channels, -1)
        weight_flat  = self.weight.view(self.out_channels, -1, 1)

        #gamma = self.gamma.view(-1, 1)
        #output = (-gamma*((input_unfold - weight_flat)**2).sum(dim=2)).exp()

        balance = self.balance.view(-1, 1)
        #output = ((input_unfold * weight_flat).sum(dim=2) + balance)**self.power
        output = F.conv2d(input, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups).view(batch_size, self.out_channels, -1)
        output = (output + balance)**self.power

        if self.bias is not None:
            output += self.bias.view(self.out_channels, -1)

        output = output.view(batch_size, self.out_channels, height, width)
        return output