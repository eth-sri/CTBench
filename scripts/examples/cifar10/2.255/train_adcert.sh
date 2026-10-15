#!/bin/bash

gpu_idx=4
dataset=cifar10
net=cnn_7layer_bn
L1=3e-6
robust_weight=1

train_eps=0.007843137255
test_eps=0.007843137255


# --- AD-CERT ---

init=fast
fast_reg=0.5
ibp_coef=0.01
train_steps=8
test_steps=10
restarts=3
attack_range_scale=2
teacher_path=./AD_CERT_RELEASE/models/cifar10/2.255/TEACHER/model.ckpt

CUDA_VISIBLE_DEVICES=$gpu_idx python mix_train.py --use-pop-bn-stats --use-adcert-training --teacher-model $teacher_path --ibp-coef $ibp_coef --attack-range-scale $attack_range_scale --model-selection None  --fast-reg $fast_reg --init $init --dataset $dataset --net $net --lr 0.0005 --lr-milestones 120 140 --train-eps $train_eps --test-eps $test_eps --train-steps $train_steps --test-steps $test_steps  --restarts $restarts --train-batch 128 --test-batch 128 --grad-clip 10 --n-epochs 160 --L1-reg $L1 --start-value-robust-weight $robust_weight --end-value-robust-weight $robust_weight --start-epoch-eps 1 --end-epoch-eps 81  --save-dir ./ablations --use-swa --swa-start 150
