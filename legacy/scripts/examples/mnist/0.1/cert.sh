#!/bin/bash


CUDA_VISIBLE_DEVICES=6 python3 mnbab_certify.py \
                        --dataset mnist --load-certify-file cert.json \
                        --net cnn_7layer_bn  \
                        --load-model ../cert-base-master/benchmark_models/mnist/eps0.1/train_eps0.2/PGD_trained/cnn_7layer_bn/init_default/EDAC_step_0.3/L1_1e-05/pop_bn_stats/model.ckpt \
                        --test-eps 0.1 --test-steps 200  \
                        --train-batch 1 --test-batch 200 \
                        --mnbab-config ./MNBAB_configs/mnist_eps0.1.json