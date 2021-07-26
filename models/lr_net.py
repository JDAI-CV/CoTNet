import math
import numpy as np
import torch
from torch import nn as nn

from config import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg
from .layers import SelectiveKernelConv, ConvBnAct, create_attn
from .registry import register_model
from .resnet import ResNet
from .layers import Shiftlution
from cupy_layers.aggregation_zeropad import LocalConvolution
from cupy_layers.aggregation_zeropad_mix import LocalConvolutionMix
from cupy_layers.aggregation_zeropad_mix_merge import LocalConvolutionMixMerge
from cupy_layers.aggregation_zeropad_dilate import LocalConvolutionDilate
from .layers import create_act_layer
import torch.nn.functional as F
from torch import einsum
from .layers.activations import Swish
from .layers.tbconv import TBConv
from .layers.kerv2d import Kerv2d
from .layers import get_act_layer

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }

default_cfgs = {
    'lrnet_basic': _cfg(
        url=''),
}

class SelfAttLayer(nn.Module):
    def __init__(self, dim, kernel_size, key_ks):
        super(SelfAttLayer, self).__init__()

        rel_factor = 1
        in_planes = dim
        rel_planes = dim // rel_factor
        out_planes = dim
        self.head_num = dim // rel_factor // 8
        self.kernel_size = kernel_size

        self.conv_q = nn.Sequential(
            nn.Conv2d(in_planes, rel_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(rel_planes),
            nn.ReLU(inplace=True)
        )
            
        if key_ks > 1:
            self.conv_k = nn.Sequential(
                nn.Conv2d(in_planes, rel_planes, kernel_size=key_ks, padding=key_ks//2, bias=False),
                nn.BatchNorm2d(rel_planes),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv_k = nn.Sequential(
                nn.Conv2d(in_planes, rel_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(rel_planes),
                nn.ReLU(inplace=True)
            )
        self.conv_v = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )
        
        self.pos_h = nn.Parameter(torch.randn(rel_planes, self.kernel_size, 1))
        self.pos_w = nn.Parameter(torch.randn(rel_planes, 1, self.kernel_size))
        self.unfold = torch.nn.Unfold(kernel_size, 1, kernel_size//2, 1)
        self.softmax = nn.Softmax(dim=2)

        self.local_conv = LocalConvolution(dim, dim, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2, dilation=1)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.conv_q(x), self.conv_k(x), self.conv_v(x)
        unfold_k = self.unfold(k)
        unfold_k = unfold_k.view(B, -1, self.kernel_size*self.kernel_size, H, W)
        pos = self.pos_h + self.pos_w
        pos = pos.view(1, -1, self.kernel_size*self.kernel_size, 1, 1)
        kp = unfold_k + pos
        
        q = q.view(B, self.head_num, -1, 1, H, W)
        kp = kp.view(B, self.head_num, -1, self.kernel_size*self.kernel_size, H, W)
        attn = (q * kp).sum(2)
        attn = self.softmax(attn)
        w = attn.view(B, 1, -1, self.kernel_size*self.kernel_size, H, W)
        x = self.local_conv(v, w)
        x = self.bn(x)
        x = self.act(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        if stride > 1:
            self.avd = nn.AvgPool2d(3, 2, padding=1)
        else:
            self.avd = None
        
        self.conv2 = SelfAttLayer(width, kernel_size=3, key_ks=1)

        #self.conv2 = nn.Conv2d(
        #    first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
        #    padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        #self.bn2 = norm_layer(width)
        #self.act2 = act_layer(inplace=True)
        #self.aa = aa_layer(channels=width, stride=stride) if use_aa else None

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        #if self.avd is not None:
        #    x = self.avd(x)

        x = self.conv2(x)
        #x = self.bn2(x)
        #if self.drop_block is not None:
        #    x = self.drop_block(x)
        #x = self.act2(x)
        #if self.aa is not None:
        #    x = self.aa(x)

        if self.avd is not None:
            x = self.avd(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act3(x)

        return x

class Bottleneck_Ks3(Bottleneck):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(Bottleneck_Ks3, self).__init__(inplanes, planes, stride, downsample, cardinality, base_width,
                 reduce_first, dilation, first_dilation, act_layer, norm_layer,
                 attn_layer, aa_layer, drop_block, drop_path)
        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        self.conv2 = SelfAttLayer(width, kernel_size=3, key_ks=3)

def _create_lrnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        ResNet, variant, default_cfg=default_cfgs[variant], pretrained=pretrained, **kwargs)

@register_model
def lrnet50(pretrained=False, **kwargs):
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return _create_lrnet('lrnet_basic', pretrained, **model_args)

@register_model
def lrnet50_ks3(pretrained=False, **kwargs):
    model_args = dict(block=Bottleneck_Ks3, layers=[3, 4, 6, 3],  **kwargs)
    return _create_lrnet('lrnet_basic', pretrained, **model_args)