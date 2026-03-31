#!/bin/bash


CUDA_VISIBLE_DEVICES=1  python3 mnbab_certify.py \
                        --dataset mnist \
                        --net cnn_7layer_bn  \
                        --load-model ../cert-base-master/benchmark_models/mnist/eps0.3/IBP_trained/cnn_7layer_bn/init_fast/fast_reg_0.5/L1_1e-06/pop_bn_stats/model.ckpt \
                        --test-eps 0.3 --test-steps 200  \
                        --train-batch 1 --test-batch 200 \
                        --mnbab-config ./MNBAB_configs/mnist_eps0.3.json