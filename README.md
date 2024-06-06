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

Model certification is done via a combination of IBP (fastest), PGD attack (fast) / autoattack (slow), CROWN-IBP (fast) and DeepPoly (medium) / MN-BaB (complete verifier, very slow). These are implemented in ```mnbab_certify.py```. For most cases (except when a new certification method is designed), it is recommended to **not** change this file at all.

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

Recommended environment setup:
```console
conda create --name CTBench python=3.9
conda activate CTBench
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

To install further requirements please run 
```
pip install -r requirements.txt
```

Python=3.9 is necessary to install MN-BaB later as some dependencies are not available in other versions, e.g. gurobipy==9.1.2. However, one may use separate training & certification environment to avoid this.

## Certification


First, install MN-BaB according to `https://github.com/eth-sri/mn-bab`.


Then, certify your models with command ```./mnbab_certify``` with relevant model path and corresponding config file listed in ```./MNBAB_configs```.

If AutoAttack is desired for stronger attack strength, run ```pip install git+https://github.com/fra31/auto-attack``` and specify ```--use-autoattack```. In most cases (when no gradient masking is expected), using the default PGD attack is faster and provides similar numbers.

If a fast evaluation is desired, MNBaB can be disabled via ```--disable-mnbab```. This will skip the complete certification provided by MN-BaB.

## CTBench Pretrained Models

Please download from [MEGA](https://mega.nz/folder/3QBgiLaD#YsidcFQ5aGKmGpJF7S1loQ). It takes 2.72GB memory.

## Benchmark

Please check our paper for more details. Scripts are included in `./scripts/examples`. For the benchmark models, set the correct hyperparameter either from the description of our paper or directly access the `train_args.json` file included in the pretrained models.