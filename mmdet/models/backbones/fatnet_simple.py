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


class conv_duc(nn.Module):
    def __init__(self, cin, cout, stride=1, conv_cfg=None, norm_cfg=None):
        super(conv_duc, self).__init__()
        self.branch0 = ConvModule(cin, cout, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg)

        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride)
        down_ch = cout * stride * stride
        self.branch1_1 = ConvModule(cin, cin, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.branch1_2 = ConvModule(cin, down_ch, kernel_size=1, stride=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.up = nn.PixelShuffle(upscale_factor=stride)
        self.branch1 = nn.Sequential(
            self.pool,
            self.branch1_1,
            self.branch1_2,
            self.up,
        )

    def forward(self, x):
        x0 = self.branch0(x)

        x1 = self.branch1(x)

        x2 = x0+x1
        return x2


@BACKBONES.register_module
class FatNetSimple(nn.Module):
    def __init__(self,
                 conv_cfg=None,
                 norm_cfg=None,
                 ):
        super(FatNetSimple, self).__init__()

        self.conv1 = ConvModule(3, 64, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.conv2 = ConvModule(64, 64, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg)

        self.conv3 = conv_duc(64, 32, stride=2, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.conv4 = conv_duc(32, 32, stride=2, conv_cfg=conv_cfg, norm_cfg=norm_cfg)

        self.conv5 = conv_duc(32, 16, stride=4, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.conv6 = conv_duc(16, 16, stride=4, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.conv7 = conv_duc(16, 16, stride=4, conv_cfg=conv_cfg, norm_cfg=norm_cfg)

        self.conv8 = conv_duc(16, 16, stride=8, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.conv9 = conv_duc(16, 16, stride=8, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.conv10 = conv_duc(16, 16, stride=8, conv_cfg=conv_cfg, norm_cfg=norm_cfg)

        self.conv11 = conv_duc(16, 16, stride=8, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.conv12 = conv_duc(16, 16, stride=8, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.conv13 = conv_duc(16, 16, stride=8, conv_cfg=conv_cfg, norm_cfg=norm_cfg)






    def forward(self, x):
        output = []
        x = self.conv1(x)
        x = self.conv2(x)
        # output.append(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # output.append(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        output.append(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        output.append(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        output.append(x)


        return tuple(output)

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
