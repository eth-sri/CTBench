#!/bin/bash

gpu_idx=5
dataset=mnist
net=cnn_7layer_bn
L1=1e-6
robust_weight=0.7

train_eps=0.2
test_eps=0.1


# --- AD-CERT ---

init=fast
fast_reg=0.5
ibp_coef=0.0075
train_steps=10
test_steps=10
restarts=3
attack_range_scale=1
teacher_path=./AD_CERT_RELEASE/models/mnist/eps0.1/TEACHER/model.ckpt

CUDA_VISIBLE_DEVICES=$gpu_idx python mix_train.py --use-pop-bn-stats --use-adcert-training --teacher-model $teacher_path --ibp-coef $ibp_coef --attack-range-scale $attack_range_scale --model-selection None  --fast-reg $fast_reg --init $init --dataset $dataset --net $net --lr 0.0005 --lr-milestones 50 60 --train-eps $train_eps --test-eps $test_eps --train-steps $train_steps --test-steps $test_steps  --restarts $restarts --train-batch 256 --test-batch 256 --grad-clip 10 --n-epochs 70 --L1-reg $L1 --start-value-robust-weight $robust_weight --end-value-robust-weight $robust_weight --start-epoch-eps 0 --end-epoch-eps 20  --save-dir ./ablations --use-swa --swa-start 65
