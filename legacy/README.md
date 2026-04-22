# Legacy MN-BaB Pipeline

> **Note:** This is a legacy, archived directory. The main CTBench pipeline has migrated to **alpha-beta-CROWN**. The code here is **not directly runnable** from this subdirectory — all files and directories in `legacy/` must be moved to the project root before use, since the scripts import modules (e.g., `loaders`, `networks`, `model_wrapper`) from the project root.

## Overview

In the original CTBench pipeline, model certification was done via a combination of IBP (fastest), PGD attack (fast) / autoattack (slow), CROWN-IBP (fast) and DeepPoly (medium) / MN-BaB (complete verifier, very slow). This is implemented in `mnbab_certify.py` in this directory.

## Environment

The original CTBench environment required Python 3.9 due to MN-BaB's dependency constraints (e.g., `gurobipy==9.1.2`):

```console
conda create --name CTBench python=3.9
conda activate CTBench
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt
```

This environment was used for both training and MN-BaB certification. If you are only using MN-BaB for certification (with a separate training environment), the same Python 3.9 constraint applies.

## Installation

Install MN-BaB according to the instructions at `https://github.com/eth-sri/mn-bab`.

## Usage

After moving the legacy files to the project root, run from the project root:

```bash
CUDA_VISIBLE_DEVICES=0 python3 mnbab_certify.py \
    --dataset cifar10 \
    --net cnn_7layer_bn \
    --load-model ./CTBenchRelease/cifar10/2.255/IBP/model.ckpt \
    --mnbab-config ./MNBAB_configs/cifar10_eps2.255.json \
    --test-batch 16
```

### Options

- `--use-autoattack`: Use AutoAttack for stronger attack strength. Requires `pip install git+https://github.com/fra31/auto-attack`. In most cases (when no gradient masking is expected), the default PGD attack is faster and provides similar numbers.
- `--disable-mnbab`: Skip the complete certification provided by MN-BaB, relying only on faster incomplete methods.
- `--tolerate-error`: Ignore MN-BaB errors (typically GPU/CPU memory overflows), marking failed samples as undecidable.

## Example Scripts

Legacy certification scripts are in `./scripts/examples/` within this directory. Each `cert.sh` uses the MN-BaB pipeline.
