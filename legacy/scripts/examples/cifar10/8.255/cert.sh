#!/bin/bash


CUDA_VISIBLE_DEVICES=5 python3 mnbab_certify.py \
                        --dataset cifar10 \
                        --net cnn_7layer_bn  \
                        --load-model ../cert-base-master/benchmark_models_more_steps/cifar10/eps0.031373/STAPS_trained/cnn_7layer_bn/init_fast/fast_reg_0.5/eps_shrink_0.9/TAPS_block_4_scale_2.0/pop_bn_stats/model.ckpt \
                        --test-eps 0.03137254901 --test-steps 200  \
                        --train-batch 1 --test-batch 200 \
                        --mnbab-config ./MNBAB_configs/cifar10_eps8.255.json