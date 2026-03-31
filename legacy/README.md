# Legacy MN-BaB Pipeline

> **Note:** This is a legacy directory. The main CTBench pipeline has migrated to **abCROWN**. This folder is maintained solely for backward compatibility and reproducing older benchmark results.

## Overview

In the original CTBench pipeline, model certification was done via a combination of IBP (fastest), PGD attack (fast) / autoattack (slow), CROWN-IBP (fast) and DeepPoly (medium) / MN-BaB (complete verifier, very slow). This is implemented in `mnbab_certify.py` in this directory.

## Environment

The original CTBench environment required Python 3.9 due to MN-BaB's dependency constraints (e.g., `gurobipy==9.1.2`):

```console
conda create --name CTBench python=3.9
conda activate CTBench
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r ../requirements.txt
```

This environment was used for both training and MN-BaB certification. If you are only using MN-BaB for certification (with a separate training environment), the same Python 3.9 constraint applies.

## Installation

Install MN-BaB according to the instructions at `https://github.com/eth-sri/mn-bab`.

## Usage

Certify models using `mnbab_certify.py` with the relevant model path and a corresponding config file from `../MNBAB_configs`:

```bash
CUDA_VISIBLE_DEVICES=0 python3 legacy/mnbab_certify.py \
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

Legacy certification scripts have been moved to `./scripts/examples/` within this directory, mirroring the original `scripts/examples/` layout. Each `cert.sh` uses the MN-BaB pipeline.
