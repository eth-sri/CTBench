#!/bin/bash

gpu_idx=3
dataset=mnist
net=cnn_7layer_bn
L1=2e-6
robust_weight=1

train_eps=0.3
test_eps=0.3
train_steps=5
test_steps=5
restarts=3

# --- SABR ---

init=fast
fast_reg=0.5
eps_shrinkage=0.9
relu_shrinkage=None

CUDA_VISIBLE_DEVICES=$gpu_idx python mix_train.py --use-pop-bn-stats --use-ibp-training --use-small-box --eps-shrinkage $eps_shrinkage --relu-shrinkage $relu_shrinkage --fast-reg $fast_reg --init $init --dataset $dataset --net $net --lr 0.0005 --lr-milestones 50 60 --train-eps $train_eps --test-eps $test_eps --train-steps $train_steps --test-steps $test_steps  --restarts $restarts --train-batch 256 --test-batch 256 --grad-clip 10 --n-epochs 70 --L1-reg $L1 --start-value-robust-weight $robust_weight --end-value-robust-weight $robust_weight --start-epoch-eps 0 --end-epoch-eps 20  --save-dir ./benchmark_models