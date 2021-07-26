import math
import numpy as np
import torch
from torch import nn as nn
from torch import einsum
import torch.nn.functional as F
from config import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg
from .layers import SelectiveKernelConv, ConvBnAct, create_attn
from .registry import register_model
from .resnet import ResNet
from einops import rearrange


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
    'resnet': _cfg(
        url=''),
}

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_rel_pos(n):
    pos = torch.meshgrid(torch.arange(n), torch.arange(n))
    pos = rearrange(torch.stack(pos), 'n i j -> (i j) n')  # [n*n, 2] pos[n] = (i, j)
    rel_pos = pos[None, :] - pos[:, None]                  # [n*n, n*n, 2] rel_pos[n, m] = (rel_i, rel_j)
    rel_pos += n - 1                                       # shift value range from [-n+1, n-1] to [0, 2n-2]
    return rel_pos

class LambdaLayer(nn.Module):
    def __init__(
        self,
        dim,
        dim_k,
        r = 15,
        heads = 4):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.dim_k = dim_k
        self.dim_v = dim // heads
        self.r = r

        self.to_q = nn.Sequential(
            nn.Conv2d(dim, dim_k * heads, 1, bias = False),
            nn.BatchNorm2d(dim_k * heads)
        )
        self.to_k = nn.Conv2d(dim, dim_k, 1, bias = False)
        self.to_v = nn.Sequential(
            nn.Conv2d(dim, self.dim_v, 1, bias = False),
            nn.BatchNorm2d(self.dim_v)
        )
        self.softmax = nn.Softmax(dim=1)
        self.embeddings = nn.Parameter(torch.randn(dim_k, 1, 1, self.r, self.r))
        self.padding = (self.r - 1) // 2
          
    def compute_position_lambdas(self, embeddings, values):
        b, v, w, h = values.shape
        values = values.view(b, 1, v, w, h)
        position_lambdas = F.conv3d(values, embeddings, padding=(0, self.padding, self.padding))
        position_lambdas = position_lambdas.view(b, self.dim_k, v, w*h)
        return position_lambdas
    
    def lambda_layer(self, queries, keys, embeddings, values):
        position_lambdas = self.compute_position_lambdas(embeddings, values)
        keys = self.softmax(keys)
        
        b, v, w, h = values.shape
        queries = queries.view(b, self.heads, self.dim_k, w*h)
        keys = keys.view(b, self.dim_k, w*h)
        values = values.view(b, self.dim_v, w*h)
        content_lambda = einsum('bkm,bvm->bkv', keys, values)
        content_output = einsum('bhkn,bkv->bhvn', queries, content_lambda)
        position_output = einsum('bhkn,bkvn->bhvn', queries, position_lambdas)
        output = content_output + position_output
        output = output.reshape(b, -1, w, h)
        return output

    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        output = self.lambda_layer(q, k, self.embeddings, v)
        return output

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
            self.conv2_down = nn.AvgPool2d(3, 2, padding=1)
        else:
            self.conv2_down = None
        self.conv2 = LambdaLayer(
            width,
            dim_k=16,
            r = 15,
            heads = 4,
        )

        #self.conv2 = nn.Conv2d(
        #    first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
        #    padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.aa = aa_layer(channels=width, stride=stride) if use_aa else None

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

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x)

        if self.conv2_down is not None:
            x = self.conv2_down(x)

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

def _create_lambdanet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        ResNet, variant, default_cfg=default_cfgs[variant], pretrained=pretrained, **kwargs)

@register_model
def lambdanet50(pretrained=False, **kwargs):
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return _create_lambdanet('resnet', pretrained, **model_args)