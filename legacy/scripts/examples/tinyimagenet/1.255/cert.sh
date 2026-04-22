#!/bin/bash


CUDA_VISIBLE_DEVICES=7 python3 mnbab_certify.py \
                        --dataset tinyimagenet \
                        --net cnn_7layer_bn_tinyimagenet  \
                        --load-model ../../CTBench/cert-base-master/benchmark_models/tinyimagenet/eps0.0039216/DPBox_trained/cnn_7layer_bn_tinyimagenet/init_fast/fast_reg_0.2/L1_2e-05/optim_swa/pop_bn_stats/model.ckpt \
                        --test-eps 0.00392156863 --test-steps 200  \
                        --train-batch 1 --test-batch 2 \
                        --mnbab-config ./MNBAB_configs/tin_eps1.255.json