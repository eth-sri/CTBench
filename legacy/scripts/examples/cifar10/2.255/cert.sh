#!/bin/bash


CUDA_VISIBLE_DEVICES=1 python3 mnbab_certify.py \
                        --dataset cifar10 --disable-mnbab --use-autoattack \
                        --net cnn_7layer_bn  \
                        --load-model ../cert-base-master/benchmark_models/cifar10/eps0.0078431/STAPS_trained_new/cnn_7layer_bn/init_fast/fast_reg_0.5/eps_shrink_0.1/relu_shrink_0.8/TAPS_block_4_scale_5.0/L1_2e-06/pop_bn_stats/model.ckpt \
                        --test-eps 0.00784313725 --test-steps 200  \
                        --train-batch 1 --test-batch 10 \
                        --mnbab-config ./MNBAB_configs/cifar10_eps2.255.json