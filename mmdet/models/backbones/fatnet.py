# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# ------------------------------------------------------------------------------
import logging

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.plugins import GeneralizedAttention
from mmdet.ops import ContextBlock, DeformConv, ModulatedDeformConv
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer
from mmdet.models.utils import (build_norm_layer, bias_init_with_prob, ConvModule)
import torch.nn.functional as F

BN_MOMENTUM = 0.1


class Pang_unit(nn.Module):  #### basic unit
    def __init__(self, cin, cout, dilation=1, conv_cfg=None, norm_cfg=None):
        super(Pang_unit, self).__init__()
        self.branch0 = ConvModule(cin, cout, kernel_size=3, stride=1, padding=1, dilation=dilation, conv_cfg=conv_cfg,
                                  norm_cfg=norm_cfg)
        self.branch1 = ConvModule(cin, cout, kernel_size=1, stride=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x0 = x0 + x1
        return x0


class Pang_unit_stride(nn.Module):  #### basic unit
    def __init__(self, cin, cout, dilation=1, conv_cfg=None, norm_cfg=None):
        super(Pang_unit_stride, self).__init__()
        self.branch0 = ConvModule(cin, cout, kernel_size=3, stride=2, padding=dilation, dilation=dilation,
                                  conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.branch1 = ConvModule(cin, cout, kernel_size=1, stride=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg)

    def forward(self, x):
        x0 = self.branch0(x)
        x0 = F.upsample_nearest(x0, scale_factor=2)
        x1 = self.branch1(x)
        x0 = x1 + x0
        return x0


@BACKBONES.register_module
class FatNet(nn.Module):
    def __init__(self,
                 stage_index=(3, 6, 9, 12),
                 conv_cfg=None,
                 norm_cfg=None,
                 ):
        super(FatNet, self).__init__()

        # self.conv1 = ConvModule(3, 16, kernel_size=7, stride=2, padding=3, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.stage_index = stage_index
        self.features = self._make_layers_pangnet(conv_cfg=conv_cfg, norm_cfg=norm_cfg)

    def _make_layers_pangnet(self, conv_cfg=None, norm_cfg=None):
        layers = nn.ModuleList()
        in_channels = 3
        cfg = [16, 16, 32, 32, 32, 32, 32, 64, 64, 64, 128, 128, 128]
        dilation = [1, 1, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16]
        for ic, v in enumerate(cfg):
            v = v * 1
            if ic <= 1:
                layers.append(Pang_unit(in_channels, v, dilation=dilation[ic], conv_cfg=conv_cfg, norm_cfg=norm_cfg))
            else:
                layers.append(Pang_unit_stride(in_channels, v, dilation=dilation[ic], conv_cfg=conv_cfg,
                                               norm_cfg=norm_cfg))
            in_channels = v
        return layers

    def forward(self, x):
        output = []
        for id, layer in enumerate(self.features):
            x = layer(x)
            if id in self.stage_index:
                output.append(x)
        return tuple(output)

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

