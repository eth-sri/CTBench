#!/bin/bash

gpu_idx=6
dataset=mnist
net=cnn_7layer_bn
L1=1e-6
robust_weight=1

train_eps=0.3
test_eps=0.3


# --- CC-DIST ---

init=fast
fast_reg=0.5
ibp_coef=0.5
train_steps=1
test_steps=10
restarts=3
attack_range_scale=1
teacher_path=./AD_CERT_RELEASE/models/mnist/eps0.3/TEACHER/model.ckpt

CUDA_VISIBLE_DEVICES=$gpu_idx python mix_train.py --use-pop-bn-stats --use-ccdist-training --teacher-model $teacher_path --ibp-coef $ibp_coef --attack-range-scale $attack_range_scale --model-selection None  --fast-reg $fast_reg --init $init --dataset $dataset --net $net --lr 0.0005 --lr-milestones 50 60 --train-eps $train_eps --test-eps $test_eps --train-steps $train_steps --test-steps $test_steps  --restarts $restarts --train-batch 256 --test-batch 256 --grad-clip 10 --n-epochs 70 --L1-reg $L1 --start-value-robust-weight $robust_weight --end-value-robust-weight $robust_weight --start-epoch-eps 0 --end-epoch-eps 20  --save-dir ./ablations --use-swa --swa-start 65
