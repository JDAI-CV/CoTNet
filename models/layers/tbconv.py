import math
import torch
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t

class TBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        blocks: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        use_weight = True
    ):
        super(TBConv, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        if in_channels % blocks != 0:
            raise ValueError('in_channels must be divisible by blocks')
        if out_channels % blocks != 0:
            raise ValueError('out_channels must be divisible by blocks')

        self.in_channels = in_channels // blocks
        self.out_channels = out_channels // blocks
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.blocks = blocks
        self.use_weight = use_weight

        if self.use_weight:
            self.weight = nn.Parameter(torch.Tensor(
                self.out_channels, self.in_channels // groups, *kernel_size))

            if bias:
                self.bias = nn.Parameter(torch.Tensor(self.out_channels))
            else:
                self.register_parameter('bias', None)

            self.reset_parameters()
        else:
            self.weight = None
            self.bias = None

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        if self.blocks != 1:
            s +=  ', blocks={blocks}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(TBConv, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

    def forward(self, input: Tensor, weight=None) -> Tensor:
        #if self.use_weight:
        #    conv_weight = self.weight
        #else:
        #    conv_weight = weight
        b, c, h, w = input.size()
        input = input.view(b*self.blocks, -1, h, w)
        x = F.conv2d(input, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups)
        b, c, h, w = x.size()
        x = x.view(-1, c * self.blocks, h, w)
        return x