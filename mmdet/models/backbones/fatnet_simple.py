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


@BACKBONES.register_module
class FatNetSimple(nn.Module):
    def __init__(self,
                 conv_cfg=None,
                 norm_cfg=None,
                 ):
        super(FatNetSimple, self).__init__()

        self.conv1 = ConvModule(3, 64, kernel_size=3, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.conv2 = ConvModule(64, 64, kernel_size=3, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.conv3 = ConvModule(64, 64, kernel_size=3, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.conv4 = ConvModule(64, 64, kernel_size=3, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.conv5 = ConvModule(64, 64, kernel_size=3, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg)


    def forward(self, x):
        output = []
        x = self.conv1(x)
        x = self.conv2(x)
        output.append(x)
        x = self.conv3(x)
        output.append(x)
        x = self.conv4(x)
        output.append(x)
        x = self.conv5(x)
        output.append(x)
        return tuple(output)

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

