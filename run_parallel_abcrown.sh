#!/bin/bash

# Dynamic GPU allocation
# Uses GPUs specified by CUDA_VISIBLE_DEVICES if set, otherwise auto-detects all available GPUs.
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    GPUS=($(seq 0 $((NUM_GPUS - 1))))
else
    IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
fi

CHUNKS=${#GPUS[@]}
if [ "$CHUNKS" -eq 0 ]; then
    echo "Error: No GPUs detected. For CPU-only runs (e.g., with --dp-only), invoke abcrown_certify.py directly."
    exit 1
fi

DATASET=""
SAVE_DIR=""
LOAD_MODEL=""
args=("$@")

for ((i=0; i<${#args[@]}; i++)); do
    if [[ "${args[$i]}" == "--dataset" ]]; then
        DATASET="${args[$i+1]}"
    fi
    if [[ "${args[$i]}" == "--save-dir" ]]; then
        SAVE_DIR="${args[$i+1]}"
    fi
    if [[ "${args[$i]}" == "--load-model" ]]; then
        LOAD_MODEL="${args[$i+1]}"
    fi
done

# Default log directory: --save-dir if given, otherwise the model checkpoint's directory
if [ -z "$SAVE_DIR" ]; then
    if [ -n "$LOAD_MODEL" ]; then
        SAVE_DIR="$(dirname "$LOAD_MODEL")"
    else
        echo "Error: --load-model must be provided."
        exit 1
    fi
fi

if [ -z "$DATASET" ]; then
    echo "Error: --dataset flag is required. (e.g., --dataset cifar10)"
    exit 1
fi

echo "Running certification across $CHUNKS GPUs (${GPUS[*]})..."

# Pre-download dataset to avoid race conditions when multiple GPU processes start simultaneously
echo "Pre-downloading dataset '$DATASET' if needed..."
TOTAL_SAMPLES=$(python3 -c "
import os, sys, torchvision, torchvision.transforms as T
ds = '$DATASET'
if ds == 'cifar10':
    torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=T.ToTensor())
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=T.ToTensor())
elif ds == 'mnist':
    torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=T.ToTensor())
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=T.ToTensor())
elif ds == 'tinyimagenet':
    if not os.path.isdir('./data/tiny-imagenet-200'):
        import subprocess
        print('TinyImageNet not found. Downloading via scripts/examples/tinyimagenet/download_tinyimagenet.sh ...', file=sys.stderr)
        subprocess.run(['bash', 'scripts/examples/tinyimagenet/download_tinyimagenet.sh'], check=True)
    else:
        print('TinyImageNet already exists.', file=sys.stderr)
    test_set = torchvision.datasets.ImageFolder('./data/tiny-imagenet-200/val', transform=T.ToTensor())
else:
    print(f'Unknown dataset: {ds}', file=sys.stderr)
    sys.exit(1)
print(f'{ds} dataset ready (test size: {len(test_set)})', file=sys.stderr)
print(len(test_set))
")
echo "Dataset pre-download complete. Test set size: $TOTAL_SAMPLES"
CHUNK_SIZE=$((TOTAL_SAMPLES / CHUNKS))

PIDS=()
for i in "${!GPUS[@]}"; do
    START=$((i * CHUNK_SIZE))
    END=$(((i + 1) * CHUNK_SIZE))
    # Ensure the last chunk covers everything
    if [ $i -eq $((CHUNKS - 1)) ]; then END=$TOTAL_SAMPLES; fi
    
    GPU=${GPUS[$i]}
    
    echo "  [GPU $GPU] Samples $START to $END"
    
    CUDA_VISIBLE_DEVICES=$GPU python3 abcrown_certify.py \
        "$@" \
        --start-idx "$START" \
        --end-idx "$END" > "$SAVE_DIR/log_${START}_${END}.txt" 2>&1 &
    PIDS+=($!)
done

echo "Launched parallel processes."

FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAILED=$((FAILED + 1))
done

if [ $FAILED -ne 0 ]; then
    echo "Error: $FAILED process(es) exited with errors. Check logs in $SAVE_DIR/log_*.txt for details."
    exit 1
else
    echo "Done! You can now run: python summarize_results.py $SAVE_DIR"
fi