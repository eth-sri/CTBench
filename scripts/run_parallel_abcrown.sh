#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

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
USER_START_IDX=""
USER_END_IDX=""
args=("$@")
PASSTHROUGH_ARGS=()

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
    if [[ "${args[$i]}" == "--start-idx" ]]; then
        USER_START_IDX="${args[$i+1]}"
        ((i++))
        continue
    fi
    if [[ "${args[$i]}" == "--end-idx" ]]; then
        USER_END_IDX="${args[$i+1]}"
        ((i++))
        continue
    fi
    PASSTHROUGH_ARGS+=("${args[$i]}")
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

mkdir -p "$SAVE_DIR"

echo "Running certification across $CHUNKS GPUs (${GPUS[*]})..."

# Pre-download dataset to avoid race conditions when multiple GPU processes start simultaneously
echo "Pre-downloading dataset '$DATASET' if needed..."
TOTAL_SAMPLES=$(cd "$REPO_ROOT" && python3 -c "
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

if [ -z "$USER_START_IDX" ]; then
    EFFECTIVE_START=0
else
    EFFECTIVE_START=$USER_START_IDX
fi

if [ -z "$USER_END_IDX" ] || [ "$USER_END_IDX" -eq -1 ]; then
    EFFECTIVE_END=$TOTAL_SAMPLES
else
    EFFECTIVE_END=$USER_END_IDX
fi

if [ "$EFFECTIVE_START" -lt 0 ]; then
    echo "Error: --start-idx must be non-negative."
    exit 1
fi

if [ "$EFFECTIVE_END" -le "$EFFECTIVE_START" ] || [ "$EFFECTIVE_END" -gt "$TOTAL_SAMPLES" ]; then
    echo "Error: effective range [$EFFECTIVE_START, $EFFECTIVE_END) is invalid for dataset size $TOTAL_SAMPLES."
    exit 1
fi

RANGE_SIZE=$((EFFECTIVE_END - EFFECTIVE_START))
CHUNK_SIZE=$(((RANGE_SIZE + CHUNKS - 1) / CHUNKS))

PIDS=()
LAUNCHED=0
for i in "${!GPUS[@]}"; do
    START=$((EFFECTIVE_START + i * CHUNK_SIZE))
    END=$((START + CHUNK_SIZE))
    if [ "$END" -gt "$EFFECTIVE_END" ]; then END=$EFFECTIVE_END; fi
    if [ "$START" -ge "$EFFECTIVE_END" ]; then break; fi

    GPU=${GPUS[$i]}

    echo "  [GPU $GPU] Samples $START to $END"

    (
        cd "$REPO_ROOT" && \
        CUDA_VISIBLE_DEVICES=$GPU python3 abcrown_certify.py \
            "${PASSTHROUGH_ARGS[@]}" \
            --start-idx "$START" \
            --end-idx "$END"
    ) > "$SAVE_DIR/log_${START}_${END}.txt" 2>&1 &
    PIDS+=($!)
    LAUNCHED=$((LAUNCHED + 1))
done

echo "Launched $LAUNCHED parallel process(es) over [$EFFECTIVE_START, $EFFECTIVE_END)."

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
