# CTBench: A Library and Benchmark for Certified Training

## Before Getting Started

Deterministic certified training focuses on $L_\infty$ certified robustness. This is different to randomized certified training represented by Randomized Smoothing, in that it provides deterministic certification of the model, and do not introduce computational overhead at inference. In the following, we omit the word `deterministic` for brevity.

Following the convention in the literature, five settings are used for evaluation: $\epsilon=0.1/0.3$ for MNIST, $\epsilon=\frac{2}{255}/\frac{8}{255}$ for CIFAR-10, and $\epsilon=\frac{1}{255}$ for TinyImageNet. It is very common to observe different methods wins on different settings, in particular, some wins at small $\epsilon$ and other wins at large $\epsilon$. The models are trained on the train set and certified on the validation set, as adopted by the community, both in adversarial machine learning and certified machine learning.

Certified training is made possible by convex relaxation based methods. Such methods relaxes the layerwise output to derive an overapproximation of the final output. While complete methods exist for certification, they are too costly to enable training.

The community has found Interval Bound Propagation (IBP) as a very effective training method despite being the least precise. All SOTA except [COLT](https://openreview.net/forum?id=SJxSDxrKDr) applies IBP in various ways. This has been theoretically investigated by [Jovanović et. al.](https://arxiv.org/abs/2102.06700) and [Lee et. al.](https://openreview.net/forum?id=52weXyh2yh) and attributed to discontinuity, sensitivity and non-smoothness. The success and limit of IBP has also been theoretically investigated by [Baader et. al.](https://arxiv.org/abs/1909.13846), [Wang et. al.](https://arxiv.org/abs/2007.06093) and [Mirman et. al.](https://openreview.net/forum?id=fsacLLU35V). They find IBP to be able to approximate every continuous function yet the construction of such network is worse than NP-hard, and IBP can never be precise generally. [Mao et. al.](https://arxiv.org/abs/2306.10426) further pinpoints the regularization effect of IBP to be on the signs of parameter.

This library (CTBench) implements all SOTA methods built upon IBP since alternative methods are both computationally and performance-wisely worse. It is carefully designed to allow easy adaption for future work. Complete documentation are provided for usablity and unittests are conducted for correctness. While the focus of CTBench is for the development of future work, it may also be easily adapted as a library for certified training.

## Basic Design

*The design principle and architecture of CTBench is explained in this section. These are general conventions to make clean code and extensions of CTBench should follow the design unless the user is really sure about what they do.*

Argument parsing is defined in `args_factory.py`. It divides the arguments into three groups: `basic` for common options, `train` for training options and `cert` for certifying options. It is recommended to follow this paradigm when adding custom arguments. The core function in this file is called ```get_args```, which first parses the arguments and then check validity of the provided arguments.

The training methods should be wrapped as a special `model wrapper` class. These are all defined in ```model_wrapper.py``` and are subclasses of ```BasicModelWrapper```. These classes are defined with inheritance and only overides methods about the corresponding stages. It is recommended to wrap custom methods as a subclass as well to avoid unexpected side effects. Functionalities such as *gradient accumulation* and *sharpness aware minimization (SAM)* should be wrapped by `function wrapper` which are also subclasses of `model wrapper`. `get_model_wrapper` function is expected to be imported by the main file to get a model wrapper (and importing it alone should be sufficient). Run ```pyreverse model_wrapper.py -d figures -k; dot -Tpdf figures/classes.dot > figures/classes.pdf``` and check ```figures/classes.pdf``` for a visual guide of model wrappers.

The main training logic is implemented in ```mix_train.py```. For most cases, trivial modification to this file should be sufficient, e.g., modifying ```parse_save_root``` function to adapt to more interested hyperparameter. It is recommended to follow the comments in the python file rather than place your code arbitrarily. In particular, side-effect free code addition is expected rather than major changes. Major changes should be wrapped inside the `model wrapper`.

Tracking statistics of checkpoints is implemented in ```get_stat.py``` in the form of ```{stat}_loop```, e.g., ```relu_loop``` and ```PI_loop```. These functions are expected to be called at test time and will iterate over the full dataset to compute the corresponding statistics. It is recommended to implement new statistics tracking in a functional way similiarly.

Model certification is done via a combination of IBP (fastest), optional AutoAttack for external adversarial accuracy, alpha-CROWN incomplete bounds, and Branch-and-Bound complete verification via [alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN). This is implemented in ```abcrown_certify.py``` with a lightweight adapter (```abcrown_adapter.py```) bridging CTBench models and data to the alpha-beta-CROWN interface, and helper utilities in ```abcrown_utils.py```. For most cases (except when a new certification method is designed), it is recommended to **not** change these files at all.

Unit tests are included in ```Utility/test_functions.py``` and can be invoked via ```cd Utility; python test_functions.py; cd ..```. Note that these tests are not complete but serves as a minimal check. Make sure to include new unit tests for new `model wrapper`.

When batch norm is involved in the net, the batch statistics will always be set based on clean input (the convention in certified training). If other behaviors are desired, e.g., to set the batch statistics based on adversarial input, call ```compute_nat_loss_and_set_BN(x, y)``` on corresponding ```x```. Batch statistics will keep the same until the next call of ```compute_nat_loss_and_set_BN```.

## Current Support

The concrete arguments shown below are for illustration of the data type.

### Standard & Adversarial Training

Standard: by `--use-std-training`. This option specifies to use the standard training rather than certified training.

[PGD](https://arxiv.org/abs/1706.06083): by `--use-pgd-training --train-steps 1 --test-steps 1  --restarts 3`. The first option specifies to use PGD training. The second/third option specifies the number of steps used in training/testing, respectively, and the fourth option specifies the number of restarts during PGD search.

[EDAC](https://arxiv.org/abs/2310.04539): by `--use-EDAC-step --EDAC-step-size 0.3`. EDAC also takes attack-relevant hyperparameters, i.e., steps and restarts.

[MART](https://openreview.net/forum?id=rklOg6EFwS): by `--use-mart-training --mart-reg-weight 5`. MART also takes attack-relevant hyperparameters, i.e., steps and restarts. Not included in the benchmark.

[ARoW](https://arxiv.org/abs/2206.03353): by `--use-arow-training --arow-reg-weight 7`. ARoW also takes attack-relevant hyperparameters, i.e., steps and restarts. Not included in the benchmark.

### Certified Training

[IBP](https://arxiv.org/abs/1810.12715): by ```--use-ibp-training```. This option specifies to use interval arithmetic to propagate the bounds.

[Fast initialization and regularization for short warm-up](https://arxiv.org/abs/2103.17268): by ```--init fast --fast-reg 0.5```. The first option uses the initialization proposed and the second option controls the weight of the regularization proposed.

[CROWN-IBP](https://arxiv.org/abs/2002.12920): `--use-DPBox-training --use-loss-fusion`. By default, during test it will compute CROWN-IBP bounds without loss fusion. Testing with loss fusion can be enabled via `--keep-fusion-when-test`.

[SABR](https://arxiv.org/abs/2210.04871): by ```--use-ibp-training --use-small-box --eps-shrinkage 0.7 --relu-shrinkage 0.8```. The second option uses adversarially selected small box as the input box, the third option defines the relative magnitude of new $\epsilon$ to old $\epsilon$, and the fourth option specifies the shrinkage of box size after each ReLU layer.

[TAPS](https://arxiv.org/abs/2305.04574): by ```--use-taps-training --block-sizes 17 4 --taps-grad-scale 5```. The first option changes propagation method from interval arithmetic (IBP) to TAPS (IBP+PGD), the second option specifies the number of layers for interval arithmetic and adversarial estimation, respectively (must sum up to the total number of layers in the network), and the third option controls the gradient weight for TAPS over IBP.

[STAPS](https://arxiv.org/abs/2305.04574): by ```--use-taps-training --use-small-box --eps-shrinkage 0.7 --relu-shrinkage 0.8 --block-sizes 17 4 --taps-grad-scale 5```. A simple combination of TAPS (propagation method) and SABR (input box selection).

[MTL-IBP](https://arxiv.org/abs/2305.13991): by `--use-mtlibp-training --ibp-coef 0.1 --attack-range-scale 1 --model-selection None`. SWA can be used to further improve generalization by `--use-swa --swa-start 150` (start to register SWA after epoch 150). While SWA does not harm, in most cases it does not improve test accuracy as well.

### Functionality

[Precise BN](https://arxiv.org/abs/2105.07576): by `--use-pop-bn-stats`. This will reset BN based on the full train set after each epoch. Recommended to be the default.

[Sharpness Aware Minimization](https://arxiv.org/abs/2010.01412): by `--use-sam --sam-rho 1e-2`. Not included in the benchmark.

[Gaussian Gradient Descent](https://arxiv.org/abs/2311.00521): by `--use-weight-smooth --weight-smooth-std-scale 1e-2`. Not included in the benchmark.

Gradient Accumulation with Original BN: by `--grad-accu-batch 64`. Batch norm are set based on the full batch instead of subbatch.

### Logging

By default, all models are locally logged. One may enable the following additional logging.

[Neptune](https://neptune.ai): by `--enable-neptune --neptune-project your_proj --neptune-tags tag1 tag2`. Neptune needs to be set up with project keys.


## Environments

Recommended separated environment setup:
```console
conda create --name CTBench python=3.9
conda activate CTBench
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt
```

A single shared environment is also supported when the CTBench and alpha-beta-CROWN dependencies are compatible on your machine. The provided file is a Linux/NVIDIA CUDA 12.8 reference environment:
```console
conda env create -f environments/reference_unified_cuda128.yaml
conda activate unified_ctbench
```

This reference environment uses Python 3.11, PyTorch 2.8.0, and CUDA 12.8 wheels; make sure the target machine has a compatible NVIDIA driver for CUDA 12.8. This setup is not expected to exactly reproduce the original paper numbers, but our sanity-check runs produced on-par results.

## Certification

First, install alpha-beta-CROWN according to the instructions at `https://github.com/Verified-Intelligence/alpha-beta-CROWN`. Ensure it is located at `../alpha-beta-CROWN` relative to the CTBench workspace root:
```console
git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN ../alpha-beta-CROWN
```
The certification subprocess invokes `conda run -n unified_ctbench` by default, matching the optional reference environment above. If you use separate CTBench and alpha-beta-CROWN environments, pass the verifier environment explicitly with `--abcrown-conda-env`.

Certify your models with the parallel wrapper script ```./scripts/run_parallel_abcrown.sh```, which distributes evaluation across multiple GPUs. All arguments are passed as named flags and forwarded to ```abcrown_certify.py```. If ```--start-idx``` and ```--end-idx``` are provided, the script shards only that subrange; otherwise it shards the full test set. Logs are saved to the ```--save-dir``` directory if provided, otherwise to the model checkpoint's directory.

For example, to run full certification with alpha-beta-CROWN:
```bash
./scripts/run_parallel_abcrown.sh --dataset cifar10 --net cnn_7layer_bn \
    --load-model ./CTBenchRelease/cifar10/2.255/TAPS/model.ckpt \
    --abcrown-config abCROWN_configs/cifar10_eps2.255.yaml --test-batch 16 
```

To follow the recommended separated attack reporting flow, install AutoAttack in your active environment (either the separated `CTBench` environment or the shared `unified_ctbench` environment):
```bash
pip install git+https://github.com/fra31/auto-attack
```

Then run AutoAttack before alpha-beta-CROWN and disable alpha-beta-CROWN's internal PGD:
```bash
./scripts/run_parallel_abcrown.sh --dataset cifar10 --net cnn_7layer_bn \
    --load-model ./CTBenchRelease/cifar10/2.255/TAPS/model.ckpt \
    --abcrown-config abCROWN_configs/cifar10_eps2.255.yaml \
    --test-batch 128 --attack-batch 128 --abcrown-batch 16 \
    --enable-heuristic-dpb --use-autoattack --disable-abcrown-pgd
```
Here ```--test-batch``` controls the outer CTBench batch size, ```--attack-batch``` controls AutoAttack's internal batch size, and ```--abcrown-batch``` controls alpha-beta-CROWN's solver batch size. For long runs where verifier errors (for example OOMs) should be treated as unknown samples instead of stopping the run, add ```--tolerate-error```.

To run IBP + heuristic DeepPoly only (no alpha-beta-CROWN):
```bash
./scripts/run_parallel_abcrown.sh --dataset cifar10 --net cnn_7layer_bn \
    --load-model ./CTBenchRelease/cifar10/2.255/TAPS/model.ckpt \
    --test-eps 0.00784313725 --test-batch 16 \
    --disable-abcrown --enable-heuristic-dpb
```

To resume an interrupted parallel certification run, pass the previous output directory to ```--load-certify-directory```. You may reuse the same directory as ```--save-dir```, or write resumed outputs to a new directory for safety:
```bash
./scripts/run_parallel_abcrown.sh --dataset cifar10 --net cnn_7layer_bn \
    --load-model ./CTBenchRelease/cifar10/2.255/TAPS/model.ckpt \
    --abcrown-config abCROWN_configs/cifar10_eps2.255.yaml --test-batch 16 \
    --load-certify-directory ./results/cifar10_eps2.255_taps \
    --save-dir ./results/cifar10_eps2.255_taps_resume
```
Completed shards with ```complete_cert_<start>_<end>.json``` are copied to the output directory and skipped, while partial shards with ```cert_<start>_<end>.json``` resume from the last processed sample. Use the same GPU sharding as the original run so the shard ranges match.

The certification pipeline automatically performs the following cascade:
1. **IBP verification** (fastest) — certifies easy samples via interval arithmetic.
2. **Heuristic DeepPoly** (optional) — enabled via the ```--enable-heuristic-dpb``` flag.
3. **AutoAttack** (optional) — enabled via ```--use-autoattack``` and reported separately as external adversarial accuracy.
4. **alpha-beta-CROWN** — for remaining samples, delegates to the verifier for alpha-CROWN incomplete bounds and beta-CROWN complete verification. Its internal PGD can be disabled via ```--disable-abcrown-pgd```; if enabled, the resulting ```unsafe-pgd``` count is reported separately from AutoAttack.

Pre-built YAML configuration files are provided in ```./abCROWN_configs``` for all standard benchmark settings. Key parameters (epsilon, batch size, model/data paths) are automatically injected at runtime.

After certification completes, aggregate per-GPU results using:
```bash
python summarize_results.py <results_directory>
```
where `<results_directory>` is the `--save-dir` you specified, or the model checkpoint's directory if `--save-dir` was omitted (e.g., `./CTBenchRelease/cifar10/2.255/TAPS/`).
For abCROWN result files, the summary reports the staged certification pipeline directly from split fields:
- external AutoAttack unsafe samples (`num_autoattack_attacked`)
- verifier-internal PGD unsafe samples (`num_abcrown_pgd_attacked` / `num_abcrown_pgd_unsafe`)
- BaB unsafe/rejected samples (`num_bab_rejected`)
- individual certification counts

The reported adversarial accuracy corresponds to the accuracy after removing attack-found unsafe samples (e.g. AutoAttack and/or verifier-internal PGD, depending on the enabled pipeline stages). BaB unsafe/rejected samples are reported separately as a verifier bucket and are not folded into this adversarial-accuracy metric.

Legacy MN-BaB result files are still supported, but their combined `num_adv_attacked` / `adv_unattacked_rate` fields are treated as a coarse aggregate unsafe bucket rather than the main staged abCROWN summary breakdown.

If a fast evaluation is desired, pass ```--dp-only``` to skip beta-CROWN and rely only on fast incomplete lower bounds (alpha-CROWN). Alternatively, use ```--disable-abcrown``` to skip alpha-beta-CROWN verification altogether (```--test-eps``` is required in this case, since there is no YAML config to read epsilon from).

## CTBench Pretrained Models

Please download from [MEGA](https://mega.nz/folder/3QBgiLaD#YsidcFQ5aGKmGpJF7S1loQ). It takes 2.72GB memory.

## Benchmark

Please check our paper for more details. Scripts are included in `./scripts/examples`. For the benchmark models, set the correct hyperparameter either from the description of our paper or directly access the `train_args.json` file included in the pretrained models.

## Legacy Support (MN-BaB)

CTBench now uses **alpha-beta-CROWN** as the default certification pipeline. MN-BaB has been demoted to legacy status. If you still need to run the old MN-BaB pipeline for reproducibility or historical comparison, please refer to the [`legacy/`](./legacy) directory.
