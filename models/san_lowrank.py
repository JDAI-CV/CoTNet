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

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'fc',
        **kwargs
    }

default_cfgs = {
    'san19': _cfg(
        url='',),
}

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SAM(nn.Module):
    def __init__(self, in_planes, rel_planes, out_planes, share_planes, kernel_size=3, stride=1, dilation=1):
        super(SAM, self).__init__()
        self.kernel_size, self.stride = kernel_size, stride
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)

        self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes * (pow(kernel_size, 2) + 1)), nn.ReLU(inplace=True),
                                    nn.Conv2d(rel_planes * (pow(kernel_size, 2) + 1), out_planes // share_planes, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(out_planes // share_planes), nn.ReLU(inplace=True),
                                    nn.Conv2d(out_planes // share_planes, pow(kernel_size, 2) * out_planes // share_planes, kernel_size=1))

        self.unfold_j = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=0, stride=stride)
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        #self.aggregation = Aggregation(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
        self.local_conv = LocalConvolution(out_planes, out_planes, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2, dilation=1)

    def forward(self, x):
        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)
        x2 = self.unfold_j(self.pad(x2)).view(x.shape[0], -1, x1.shape[2], x1.shape[3])
        w = self.conv_w(torch.cat([x1, x2], 1))
        w = w.view(x1.shape[0], -1, self.kernel_size*self.kernel_size, x1.shape[2], x1.shape[3])
        w = w.unsqueeze(1)
        #x = self.aggregation(x3, w)
        x = self.local_conv(x3, w)
        return x

class SAM_lowRank(nn.Module):
    def __init__(self, in_planes, rel_planes, out_planes, share_planes, kernel_size=3, stride=1, dilation=1):
        super(SAM_lowRank, self).__init__()
        self.rel_planes = rel_planes
        self.out_planes = out_planes
        self.kernel_size, self.stride = kernel_size, stride

        self.pool_size = min(512 // out_planes, 4)
        self.down = nn.AvgPool2d(self.pool_size, self.pool_size, padding=0) if self.pool_size > 1 else None

        self.unfold_j = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=0, stride=stride)
        self.pad = nn.ReflectionPad2d(kernel_size // 2) 

        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes + 2*rel_planes, kernel_size=1, bias=False),
            #nn.BatchNorm2d(out_planes + rel_planes),
            #nn.ReLU(inplace=True),
        )

        self.key_embed = nn.Sequential(
            nn.BatchNorm2d(rel_planes * self.kernel_size * self.kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(rel_planes * self.kernel_size * self.kernel_size, rel_planes, 1, bias=False),
        )
        self.conv_w = nn.Sequential(
            nn.BatchNorm2d(rel_planes * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(rel_planes * 2, out_planes * self.kernel_size * 2, kernel_size=1, bias=False)
        )

        self.local_conv = LocalConvolution(out_planes, out_planes, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2, dilation=1)


    def forward(self, x):
        x = self.conv(x)
        q, k, x = torch.split(x, [self.rel_planes, self.rel_planes, self.out_planes], 1)
        x2 = self.unfold_j(self.pad(k))
        x2 = x2.view(x.shape[0], -1, x.shape[2], x.shape[3])
        x2 = self.key_embed(x2)
        
        qk = torch.cat([q, x2], 1)
        if self.pool_size > 1:
            qk = self.down(qk)

        b, c, qk_hh, qk_ww = qk.size()
        
        embed = self.conv_w(qk)
        embed_h, embed_w = torch.split(embed, embed.shape[1] // 2, dim=1)
        embed_h = embed_h.view(b, -1, self.kernel_size, 1, qk_hh, qk_ww)
        embed_w = embed_w.view(b, -1, 1, self.kernel_size, qk_hh, qk_ww)
        w = embed_h * embed_w
        w = w.view(x.shape[0], -1, self.kernel_size*self.kernel_size, qk_hh, qk_ww)

        if self.pool_size > 1:
            w = w.view(b, -1, self.kernel_size*self.kernel_size, qk_hh, 1, qk_ww, 1)
            w = w.expand(b, -1, self.kernel_size*self.kernel_size, qk_hh, self.pool_size, qk_ww, self.pool_size).contiguous()
            w = w.view(b, -1, self.kernel_size*self.kernel_size, x.shape[2], x.shape[3])

        w = w.unsqueeze(1)
        x = self.local_conv(x, w)
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, kernel_size=7, stride=1):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.sam = SAM(in_planes, rel_planes, mid_planes, share_planes, kernel_size, stride)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv = nn.Conv2d(mid_planes, out_planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(x))
        out = self.relu(self.bn2(self.sam(out)))
        out = self.conv(out)
        out += identity
        return out

class SAN(nn.Module):
    def __init__(self, in_chans, block, layers, kernels, num_classes, **kwargs):
        super(SAN, self).__init__()
        c = 64
        self.conv_in, self.bn_in = conv1x1(3, c), nn.BatchNorm2d(c)
        self.conv0, self.bn0 = conv1x1(c, c), nn.BatchNorm2d(c)
        self.layer0 = self._make_layer(block, c, layers[0], kernels[0])

        c *= 4
        self.conv1, self.bn1 = conv1x1(c // 4, c), nn.BatchNorm2d(c)
        self.layer1 = self._make_layer(block, c, layers[1], kernels[1])

        c *= 2
        self.conv2, self.bn2 = conv1x1(c // 2, c), nn.BatchNorm2d(c)
        self.layer2 = self._make_layer(block, c, layers[2], kernels[2])

        c *= 2
        self.conv3, self.bn3 = conv1x1(c // 2, c), nn.BatchNorm2d(c)
        self.layer3 = self._make_layer(block, c, layers[3], kernels[3])

        c *= 2
        self.conv4, self.bn4 = conv1x1(c // 2, c), nn.BatchNorm2d(c)
        self.layer4 = self._make_layer(block, c, layers[4], kernels[4])

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c, num_classes)

    def _make_layer(self, block, planes, blocks, kernel_size=7, stride=1):
        layers = []
        for _ in range(0, blocks):
            layers.append(block(planes, planes // 16, planes // 4, planes, 8, kernel_size, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn_in(self.conv_in(x)))
        x = self.relu(self.bn0(self.layer0(self.conv0(self.pool(x)))))
        x = self.relu(self.bn1(self.layer1(self.conv1(self.pool(x)))))
        x = self.relu(self.bn2(self.layer2(self.conv2(self.pool(x)))))
        x = self.relu(self.bn3(self.layer3(self.conv3(self.pool(x)))))
        x = self.relu(self.bn4(self.layer4(self.conv4(self.pool(x)))))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def _create_san(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        SAN, variant, default_cfg=default_cfgs[variant], pretrained=pretrained, **kwargs)

@register_model
def san19(pretrained=False, **kwargs):
    #model_args = dict(block=Bottleneck, layers=[3, 3, 4, 6, 3], kernels = [3, 3, 3, 3, 3],  **kwargs)
    #model_args = dict(block=Bottleneck, layers=[3, 3, 4, 6, 3], kernels = [3, 5, 5, 5, 5],  **kwargs)
    model_args = dict(block=Bottleneck, layers=[3, 3, 4, 6, 3], kernels=[3, 7, 7, 7, 7],  **kwargs)
    return _create_san('san19', pretrained, **model_args)