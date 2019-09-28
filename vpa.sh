#!/usr/bin/env bash
./tools/dist_train.sh cfgs_fatdet/pascal/baseline/ttfnet_r18_3x_lr16_no_pretrain_bn.py 2
./tools/dist_train.sh cfgs_fatdet/pascal/baseline/ttfnet_r18_3x_lr016_no_pretrain_bn.py 2
./tools/dist_train.sh cfgs_fatdet/pascal/baseline/ttfnet_r18_3x_lr016_no_pretrain_gn.py 2