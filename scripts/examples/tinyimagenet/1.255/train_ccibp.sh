#!/bin/bash

gpu_idx=0
dataset=tinyimagenet
net=cnn_7layer_bn_tinyimagenet
L1=5e-5
robust_weight=0.7

train_eps=0.00392156863
test_eps=0.00392156863


# --- MTL-IBP ---

init=fast
fast_reg=0.2
ibp_coef=5e-2
train_steps=1
test_steps=10
restarts=3
attack_range_scale=1

CUDA_VISIBLE_DEVICES=$gpu_idx python mix_train.py --use-pop-bn-stats --use-ccibp-training --ibp-coef $ibp_coef --attack-range-scale $attack_range_scale --model-selection None  --fast-reg $fast_reg --init $init --dataset $dataset --net $net --lr 0.0005 --lr-milestones 120 140 --train-eps $train_eps --test-eps $test_eps --train-steps $train_steps --test-steps $test_steps  --restarts $restarts --train-batch 128 --test-batch 128 --grad-clip 10 --n-epochs 160 --L1-reg $L1 --start-value-robust-weight $robust_weight --end-value-robust-weight $robust_weight --start-epoch-eps 1 --end-epoch-eps 81  --save-dir ./benchmark_models --use-swa --swa-start 150
