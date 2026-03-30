#!/bin/bash

# Configuration
GPUS=(0 1 2 3)
CHUNKS=${#GPUS[@]}
TOTAL_SAMPLES=10000
CHUNK_SIZE=$((TOTAL_SAMPLES / CHUNKS))

# Args
DATASET=$1
EPS_DIR=$2   # e.g., 2.255, 8.255, 0.1, 0.3
NET=$3
METHOD=$4
CONFIG=$5
BATCH=$6

if [ -z "$CONFIG" ]; then
    echo "Usage: ./run_parallel_abcrown.sh <dataset> <eps_dir> <net> <method> <config_yaml> <batch_size>"
    echo "Example: ./run_parallel_abcrown.sh cifar10 2.255 cnn_7layer_bn IBP abCROWN_configs/cifar10_eps2.255.yaml 16"
    exit 1
fi

SAVE_DIR="./results/$DATASET/$EPS_DIR/$METHOD"
mkdir -p "$SAVE_DIR"

echo "Running $METHOD on $DATASET (eps_dir $EPS_DIR) across GPUs ${GPUS[*]}..."

# Pre-download dataset to avoid race conditions when multiple GPU processes start simultaneously
echo "Pre-downloading dataset '$DATASET' if needed..."
python3 -c "
import torchvision, torchvision.transforms as T
ds = '$DATASET'
if ds == 'cifar10':
    torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=T.ToTensor())
    torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=T.ToTensor())
elif ds == 'mnist':
    torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=T.ToTensor())
    torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=T.ToTensor())
print(f'{ds} dataset ready.')
"
echo "Dataset pre-download complete."

for i in "${!GPUS[@]}"; do
    START=$((i * CHUNK_SIZE))
    END=$(((i + 1) * CHUNK_SIZE))
    # Ensure the last chunk covers everything
    if [ $i -eq $((CHUNKS - 1)) ]; then END=$TOTAL_SAMPLES; fi
    
    GPU=${GPUS[$i]}
    
    echo "  [GPU $GPU] Samples $START to $END"
    
    CUDA_VISIBLE_DEVICES=$GPU python3 abcrown_certify.py \
        --dataset "$DATASET" \
        --net "$NET" \
        --load-model "./CTBenchRelease/$DATASET/$EPS_DIR/$METHOD/model.ckpt" \
        --abcrown-config "$CONFIG" \
        --save-dir "$SAVE_DIR" \
        --test-batch "$BATCH" \
        --start-idx "$START" \
        --end-idx "$END" \
        --tolerate-error > "$SAVE_DIR/log_${START}_${END}.txt" 2>&1 &
done

echo "Launched parallel processes."
wait
echo "Done! You can now run: python summarize_results.py $SAVE_DIR"
