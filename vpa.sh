#!/usr/bin/env bash
./tools/dist_train.sh cfgs_fatdet/pascal/baseline/ttfnet_r101_1x_lr004.py 2
./tools/dist_train.sh cfgs_fatdet/pascal/baseline/ttfnet_r18_1x_lr004_ep12.py 2
./tools/dist_train.sh cfgs_fatdet/pascal/baseline/ttfnet_r18_1x_lr008.py 2
