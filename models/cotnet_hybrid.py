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
from .layers import create_act_layer
import torch.nn.functional as F
from torch import einsum
from .layers.activations import Swish
from .layers.tbconv import TBConv
from .layers.kerv2d import Kerv2d
from .layers import get_act_layer
from .layers import SplitAttnConv2d
from .layers import DropBlock2d, DropPath, AvgPool2dSame, BlurPool2d, create_classifier

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
    'cot_basic': _cfg(
        url=''),

    'cot_s': _cfg(
        url='',
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=0.888, interpolation='bicubic'),
    'cot_m': _cfg(
        url='',
        input_size=(3, 288, 288), pool_size=(9, 9), crop_pct=0.9, interpolation='bicubic'),
    'cot_l': _cfg(
        url='',
        input_size=(3, 320, 320), pool_size=(10, 10), crop_pct=0.909, interpolation='bicubic'),
}

class CoTLayer(nn.Module):
    def __init__(self, dim, kernel_size):
        super(CoTLayer, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size//2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv2d(2*dim, dim//factor, 1, bias=False),
            nn.BatchNorm2d(dim//factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//factor, pow(kernel_size, 2) * dim // share_planes, kernel_size=1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=pow(kernel_size, 2) * dim // share_planes)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.local_conv = LocalConvolution(dim, dim, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2, dilation=1)
        self.bn = nn.BatchNorm2d(dim)
        act = get_act_layer('swish')
        self.act = act(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv2d(dim, attn_chs, 1),
            nn.BatchNorm2d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_chs, self.radix*dim, 1)
        )

    def forward(self, x):
        k = self.key_embed(x)
        qk = torch.cat([x, k], dim=1)
        b, c, qk_hh, qk_ww = qk.size()

        w = self.embed(qk)
        w = w.view(b, 1, -1, self.kernel_size*self.kernel_size, qk_hh, qk_ww)
        
        x = self.conv1x1(x)
        x = self.local_conv(x, w)
        x = self.bn(x)
        x = self.act(x)

        B, C, H, W = x.shape
        x = x.view(B, C, 1, H, W)
        k = k.view(B, C, 1, H, W)
        x = torch.cat([x, k], dim=2)

        x_gap = x.sum(dim=2)
        x_gap = x_gap.mean((2, 3), keepdim=True)
        x_attn = self.se(x_gap)
        x_attn = x_attn.view(B, C, self.radix)
        x_attn = F.softmax(x_attn, dim=2)
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1))).sum(dim=2)
        
        return out.contiguous()

class CoTBottleneck(nn.Module):
    expansion = 4

    def __init__(self, block_idx, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None, radix=1, avd=False,  avd_first=True, conv_dim={},
                 c4_dim=-1, c4_idx={}):
        super(CoTBottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        self.avd_first = avd_first
        self.avd = None

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = nn.ReLU(inplace=True) #act_layer(inplace=True)

        if (width in conv_dim) or (width == c4_dim and block_idx not in c4_idx):
            if stride > 1 and avd:
                self.avd = nn.AvgPool2d(3, stride, padding=1) if aa_layer is None else aa_layer(channels=width, stride=stride)
                stride = 1
            
            if radix >= 1:
                self.conv2 = SplitAttnConv2d(
                    first_planes, width, kernel_size=3, stride=stride, padding=first_dilation, reduction_factor=4, 
                    dilation=first_dilation, groups=cardinality, radix=radix, norm_layer=norm_layer, drop_block=drop_block, act_layer=get_act_layer('swish'))
            else:
                self.conv2 = nn.Sequential(
                   nn.Conv2d(first_planes, width, kernel_size=3, stride=stride, padding=first_dilation, 
                       dilation=first_dilation, groups=cardinality, bias=False),
                   norm_layer(width),
                   act_layer(inplace=True)
                )
        else:
            self.conv2 = CoTLayer(width, kernel_size=3)
            if stride > 1:
                self.avd = nn.AvgPool2d(3, stride, padding=1) if aa_layer is None else aa_layer(channels=width, stride=stride)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.act3 = nn.ReLU(inplace=True)  #act_layer(inplace=True)
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

        if (self.avd is not None) and self.avd_first:
            x = self.avd(x)

        x = self.conv2(x)

        if (self.avd is not None) and (not self.avd_first):
            x = self.avd(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act3(x)

        return x

def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

def downsample_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])

def downsample_avg(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])

def drop_blocks(drop_block_rate=0.):
    return [
        None, None,
        DropBlock2d(drop_block_rate, 5, 0.25) if drop_block_rate else None,
        DropBlock2d(drop_block_rate, 3, 1.00) if drop_block_rate else None]

def make_blocks(
        block_fn, channels, block_repeats, inplanes, reduce_first=1, output_stride=32,
        down_kernel_size=1, avg_down=False, drop_block_rate=0., drop_path_rate=0., **kwargs):
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    #net_stride = 4
    net_stride = 2
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        #stride = 1 if stage_idx == 0 else 2
        stride = 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes, out_channels=planes * block_fn.expansion, kernel_size=down_kernel_size,
                stride=stride, dilation=dilation, first_dilation=prev_dilation, norm_layer=kwargs.get('norm_layer'))
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            blocks.append(block_fn(block_idx,
                inplanes, planes, stride, downsample, first_dilation=prev_dilation,
                drop_path=DropPath(block_dpr) if block_dpr > 0. else None, **block_kwargs))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info

def make_blocks_arr(
        block_fn_arr, channels, block_repeats, inplanes, reduce_first=1, output_stride=32,
        down_kernel_size=1, avg_down=False, drop_block_rate=0., drop_path_rate=0., **kwargs):
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    #net_stride = 4
    net_stride = 2
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        #stride = 1 if stage_idx == 0 else 2
        stride = 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        if stage_idx == 0 or stage_idx == 1:
            block_fn = block_fn_arr[0]
        else:
            block_fn = block_fn_arr[1]

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes, out_channels=planes * block_fn.expansion, kernel_size=down_kernel_size,
                stride=stride, dilation=dilation, first_dilation=prev_dilation, norm_layer=kwargs.get('norm_layer'))
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            blocks.append(block_fn(block_idx,
                inplanes, planes, stride, downsample, first_dilation=prev_dilation,
                drop_path=DropPath(block_dpr) if block_dpr > 0. else None, **block_kwargs))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info

class CoTHybridNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, in_chans=3,
                 cardinality=1, base_width=64, stem_width=64, stem_type='',
                 output_stride=32, block_reduce_first=1, down_kernel_size=1, avg_down=False,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None, drop_rate=0.0, drop_path_rate=0.,
                 drop_block_rate=0., global_pool='avg', zero_init_last_bn=True, block_args=None):

        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        super(CoTHybridNet, self).__init__()

        # Stem
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs_1 = stem_chs_2 = stem_width
            if 'tiered' in stem_type:
                stem_chs_1 = 3 * (stem_width // 4)
                stem_chs_2 = stem_width if 'narrow' in stem_type else 6 * (stem_width // 4)
            self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_chs_1, 3, stride=2, padding=1, bias=False),
                norm_layer(stem_chs_1),
                #act_layer(inplace=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_chs_1, stem_chs_2, 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs_2),
                #act_layer(inplace=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_chs_2, inplanes, 3, stride=1, padding=1, bias=False)])
        else:
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.act1 = nn.ReLU(inplace=True) #act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        ## Stem Pooling
        #if aa_layer is not None:
        #    self.maxpool = nn.Sequential(*[
        #        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #        aa_layer(channels=inplanes, stride=2)])
        #else:
        #    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        channels = [64, 128, 256, 512]

        if type(block) == list:
            stage_modules, stage_feature_info = make_blocks_arr(
                block, channels, layers, inplanes, cardinality=cardinality, base_width=base_width,
                output_stride=output_stride, reduce_first=block_reduce_first, avg_down=avg_down,
                down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer,
                drop_block_rate=drop_block_rate, drop_path_rate=drop_path_rate, **block_args)
        else:
            stage_modules, stage_feature_info = make_blocks(
                block, channels, layers, inplanes, cardinality=cardinality, base_width=base_width,
                output_stride=output_stride, reduce_first=block_reduce_first, avg_down=avg_down,
                down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer,
                drop_block_rate=drop_block_rate, drop_path_rate=drop_path_rate, **block_args)
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        if type(block) == list:
            self.num_features = 512 * block[1].expansion
        else:
            self.num_features = 512 * block.expansion
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, 'zero_init_last_bn'):
                    m.zero_init_last_bn()

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x


def _create_se_cotnetd(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        CoTHybridNet, variant, default_cfg=default_cfgs[variant], pretrained=pretrained, **kwargs)

@register_model
def se_cotnetd_50(pretrained=False, **kwargs):
    model_args = dict(block=CoTBottleneck, layers=[3, 4, 6, 3],  
    act_layer=get_act_layer('swish'),
    stem_type='deep', stem_width=32, avg_down=True, base_width=64, cardinality=1, aa_layer=None, #BlurPool2d
    block_args=dict(radix=1, avd=False, avd_first=True, conv_dim={64,128}, c4_dim=256, c4_idx=set(range(0,6,2))), **kwargs)
    return _create_se_cotnetd('cot_basic', pretrained, **model_args)

@register_model
def se_cotnetd_101(pretrained=False, **kwargs):
    model_args = dict(block=CoTBottleneck, layers=[3, 4, 23, 3],  
    act_layer=get_act_layer('swish'),
    stem_type='deep', stem_width=64, avg_down=True, base_width=64, cardinality=1, aa_layer=None,  #BlurPool2d
    block_args=dict(radix=1, avd=False, avd_first=True, conv_dim={64,128}, c4_dim=256, c4_idx=set(range(0,23,2))), **kwargs)
    return _create_se_cotnetd('cot_basic', pretrained, **model_args)

@register_model
def se_cotnetd_152(pretrained=False, **kwargs):
    model_args = dict(block=CoTBottleneck, layers=[3, 8, 36, 3],  
    act_layer=get_act_layer('swish'),
    stem_type='deep', stem_width=64, avg_down=True, base_width=64, cardinality=1, aa_layer=BlurPool2d,
    block_args=dict(radix=1, avd=True, avd_first=False, conv_dim={64,128}, c4_dim=256, c4_idx=set(range(0,36,2))), **kwargs)
    return _create_se_cotnetd('cot_s', pretrained, **model_args)

@register_model
def se_cotnetd_152_L(pretrained=False, **kwargs):
    model_args = dict(block=CoTBottleneck, layers=[3, 8, 36, 3],  
    act_layer=get_act_layer('swish'),
    stem_type='deep', stem_width=64, avg_down=True, base_width=64, cardinality=1, aa_layer=BlurPool2d,
    block_args=dict(radix=1, avd=True, avd_first=False, conv_dim={64,128}, c4_dim=256, c4_idx=set(range(0,36,2))), **kwargs)
    return _create_se_cotnetd('cot_l', pretrained, **model_args)

@register_model
def se_cotnetd_200(pretrained=False, **kwargs):
    model_args = dict(block=CoTBottleneck, layers=[3, 24, 36, 3],  
    act_layer=get_act_layer('swish'),
    stem_type='deep', stem_width=64, avg_down=True, base_width=64, cardinality=1, aa_layer=BlurPool2d,
    block_args=dict(radix=1, avd=True, avd_first=False, conv_dim={64,128}, c4_dim=256, c4_idx=set(range(0,36,2))), **kwargs)
    return _create_se_cotnetd('cot_s', pretrained, **model_args)

@register_model
def se_cotnetd_270(pretrained=False, **kwargs):
    model_args = dict(block=CoTBottleneck, layers=[4, 29, 53, 4],  
    act_layer=get_act_layer('swish'),
    stem_type='deep', stem_width=64, avg_down=True, base_width=64, cardinality=1, aa_layer=BlurPool2d,
    block_args=dict(radix=1, avd=True, avd_first=False, conv_dim={64,128}, c4_dim=256, c4_idx=set(range(0,53,2))), **kwargs)
    return _create_se_cotnetd('cot_s', pretrained, **model_args)