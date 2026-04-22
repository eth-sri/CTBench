import argparse
import warnings
from typing import Iterable, Callable
import logging
import torch

logging.basicConfig(level=logging.INFO)

# --- helper functions for argument type conversion ---
def dtype_or_none(value:str, dtype:Callable):
    try:
        if value.lower() == 'none':
            return None
        else:
            return dtype(value)
    except:
        raise ValueError(f"Cannot convert {value} to {dtype}.")

def float_or_none(value:str):
    return dtype_or_none(value, float)
    
def str_or_none(value:str):
    return dtype_or_none(value, str)

# --- main function for argument parsing ---
def get_args(include:Iterable=["basic", "train", "cert"]):
    '''
    @param:
        include: a list of parameter groups wanted.
            - basic: dataset, net, batch, roots, logs etc.
            - train: optimizer, training methods etc.
            - cert: timeout etc.

    @remarks:
        - To perform certified training, include "basic" and "train".
        - To perform certification, include "basic" and "cert".
    '''
    parser = argparse.ArgumentParser(description='A easy-to-modify library for IBP-based certified training.')

    logging.info(f"Using arguments group: {', '.join(include)}")
    
    if "basic" in include:
        # Basic arguments
        parser.add_argument('--dataset', required=True, type=str, help='Dataset to use.')
        parser.add_argument('--net', required=True, type=str, help='Network to use.')
        parser.add_argument('--init', default='default', type=str, help='Initialization to use.')
        parser.add_argument('--load-model', default=None, type=str, help='Path of the model to load (None for randomly initialized models).')
        parser.add_argument('--frac-valid', default=None, type=float, help='Fraction of validation samples (None for no validation).')
        parser.add_argument('--save-dir', default=None, type=str, help='Path to save the logs and the best checkpoint.')
        parser.add_argument('--random-seed', default=123, type=int, help="Global random seed for setting up torch, numpy and random.")
        parser.add_argument('--train-eps', required=False, type=float, help='Input epsilon to train with. Set eps=0 for standard training.')
        parser.add_argument('--test-eps', required=True, type=float, help='Input epsilon to test with.')
        parser.add_argument('--train-batch', default=100, type=int, help='Batch size for training.')
        parser.add_argument('--test-batch', default=100, type=int, help='Batch size for testing.')
        # gradient ascent attack arguments
        parser.add_argument('--step-size', default=None,  type=float, help='The size of each pgd step. Step size is scaled by the corresponding search box size, i.e. size should be chosen in (0, 1].')
        parser.add_argument('--train-steps', default=None,  type=int, help='The number of pgd steps taken during training.')
        parser.add_argument('--test-steps', default=None,  type=int, help='The number of pgd steps taken during testing.')
        parser.add_argument('--restarts', default=1,  type=int, help='the number of pgd restarts.')
        parser.add_argument("--grad-accu-batch", default=None, type=int, help="If None, do not use grad accumulation; If an int, use the specified number as the batch size and accumulate grad for the whole batch (train/test).")
        # neptune logging
        parser.add_argument('--enable-neptune', action='store_true', help='Whether to enable neptune logging.')
        parser.add_argument('--neptune-project', default="", type=str, help='The neptune project name to log to.')
        parser.add_argument('--neptune-tags', default=None, type=str, nargs='*', help='The neptune tags to log to.')

    if "train" in include:
        # Optimizer and learning rate scheduling
        parser.add_argument('--opt', default='adam', type=str, choices=['adam', 'sgd'], help='Optimizer to use.')
        parser.add_argument('--n-epochs', default=1, type=int, help='Number of train epochs.')
        parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate for optimizer.')
        parser.add_argument('--lr-milestones', default=None,  type=int, nargs='*', help='The milestones for MultiStepLR.')
        parser.add_argument('--lr-decay-factor', default=0.2,  type=float, help='The decay rate of lr.')
        parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for SGD optimizer.')
        parser.add_argument('--grad-clip', default=1e10,  type=float, help="Maximum gradient L2 norm for each step.")
        parser.add_argument('--model-selection', default="robust_accu", type=str_or_none, help='The criterium for selecting models.')
        parser.add_argument("--use-swa", action='store_true', help="Whether to use Schocastic Weight Averaging.")
        parser.add_argument('--swa-start', default=None, type=int, help='The epoch to start SWA.')
        parser.add_argument('--swa-freq', default=10, type=int, help='The frequency (in terms of steps) of SWA.')
        parser.add_argument('--swa-lr', default=None, type=float, help='The learning rate for SWA.')
        # parser.add_argument('--data-aug', default="default", type=str_or_none, help='Data augmentation method to use. Default is flip-pad.')
        parser.add_argument("--use-amp", action='store_true', help="Whether to use Automatic Mixed Precision for forward pass. This will use torch.float16 instead of torch.float32.")

        # Euclidean regularization
        parser.add_argument('--L1-reg', default=0,  type=float, help='the L1 reg coefficient.')

        # customized functionality
        parser.add_argument('--save-every-epoch', action='store_true', help='Whether to store the model after every epoch.')
        parser.add_argument('--verbose-first-epoch', action='store_true', help='Whether to verbose the first epoch.')
        parser.add_argument('--verbose-gap', default=0.05, type=float, help='Percentage in the first epoch for each logging.')

        # Standard training
        parser.add_argument('--use-std-training', action='store_true', help='Whether to only use natural loss for training.')

    
        # Configuration of basic robust training
        parser.add_argument('--no-anneal', action='store_true', help='Whether to use eps annealing. Behavior can be customized, e.g. specify using train_eps or test_eps.')
        parser.add_argument('--start-value-robust-weight', default=0,  type=float, help='the start value of the weight of the robust loss')
        parser.add_argument('--end-value-robust-weight', default=0,  type=float, help='the end value of the weight of the robust loss')
        parser.add_argument('--start-epoch-robust-weight', default=0,  type=int)
        parser.add_argument('--end-epoch-robust-weight', default=0,  type=int)
        ## Techniques to improve generalization
        ### Weight smoothing 
        parser.add_argument('--use-weight-smooth', action='store_true', help='Whether to use weight smoothing.')
        parser.add_argument('--weight-smooth-std-scale', default=1e-2, type=float, help="The scale of the weight smoothing std.")
        ### Sharpness-aware minimization
        parser.add_argument("--use-sam", action='store_true', help="Whether to use Sharpness-Aware Minimization.")
        parser.add_argument("--sam-rho", default=1e-2, type=float, help="The rho for SAM.")
        parser.add_argument("--adaptive-sam-rho", action='store_true', help="Whether to schedule SAM rho according to current eps.")
        ### Precise BN (population stats)
        parser.add_argument('--use-pop-bn-stats', action='store_true', help='Whether to use population BN stats for validation.')
        
        # Configuration of PGD training
        parser.add_argument('--use-pgd-training', action='store_true', help='Whether to use PGD training. This would override configuration of all other training methods, i.e. resulting in purely PGD training.')
        parser.add_argument('--use-multipgd-training', action='store_true', help='Whether to use Multi-PGD training.')
        parser.add_argument('--use-EDAC-step', action='store_true', help='Whether to use EDAC extragradient step.')
        parser.add_argument('--EDAC-step-size', default=0.3, type=float, help='Hyperparameter for EDAC step size.')


        # Configuration of ARoW training
        parser.add_argument('--use-arow-training', action='store_true', help='Whether to use ARoW training.')
        parser.add_argument('--arow-reg-weight', default=7, type=float, help='Equivalent to 2*lambda in ARoW paper.')
        parser.add_argument('--arow-label-smoothing', default=0.2, type=float, help='Hyperparameter for ARoW label smoothing.')

        # Configuration of MART training
        parser.add_argument('--use-mart-training', action='store_true', help='Whether to use MART training.')
        parser.add_argument('--mart-reg-weight', default=5, type=float, help='The reg weight for MART.')


        # Configuration of Certified training
        parser.add_argument('--start-epoch-eps', default=0, type=int, help="Start epoch of eps annealing.")
        parser.add_argument('--end-epoch-eps', default=40, type=int, help="End epoch of eps annealing.")
        parser.add_argument('--start-value-eps', default=0, type=float, help="Start value of eps annealing.")
        parser.add_argument('--end-value-eps', default=0, type=float, help="End value of eps annealing.")
        parser.add_argument("--schedule", default="smooth", type=str, choices=["smooth", "linear", "step"], help="Schedule for eps annealing.")
        parser.add_argument("--step-epoch", default=1, type=int,  help="Epoch for each step; only takes effect for step schedule.")


        # IBP training
        parser.add_argument('--use-ibp-training', action='store_true', help='Whether to use vanilla IBP. This would override use-taps-training. If combined with use_small_box, it would invoke SABR.')
        # Configuration of fast regularization
        parser.add_argument('--fast-reg', default=0, type=float, help="Weight of fast regularization. This regularization shortens eps annealing for IBP and increases the performance of IBP-based methods in general.")
        parser.add_argument('--min-eps-reg', default=1e-6, type=float, help="Minimum eps used for regularization computation.")

        # MTL-IBP training
        parser.add_argument('--use-mtlibp-training', action='store_true', help='Whether to use MTL-IBP.')
        parser.add_argument('--use-expibp-training', action='store_true', help='Whether to use EXP-IBP.')
        parser.add_argument('--use-ccibp-training', action='store_true', help='Whether to use CC-IBP.')
        parser.add_argument('--ibp-coef', default=1, type=float, help='The coefficient of IBP loss in MTL-IBP / EXP-IBP / CC-IBP.')
        parser.add_argument('--attack-range-scale', default=1, type=float, help='The attack eps as scale*train_eps in MTL-IBP / EXP-IBP / CC-IBP.')


        # (Small box) SABR Training
        parser.add_argument('--use-small-box', action='store_true', help='Whether to use small box. When combined with use-ibp-training, it invokes SABR; when combined with use-taps-training. it invokes STAPS.')
        parser.add_argument('--eps-shrinkage', default=1, type=float, help="The effective eps would be shrinkage * eps. Equivalent to lambda in SABR paper.")
        parser.add_argument('--relu-shrinkage', default=None, type=float_or_none, help="A positive constant smaller than 1, indicating the ratio of box shrinkage after each ReLU. Only useful in eps=2/255 CIFAR10 in SABR paper (set to 0.8). None for no ReLU shrinkage.")


        # TAPS training
        parser.add_argument('--use-taps-training', action='store_true', help='Whether to use TAPS. When combined with use-taps-training. it invokes STAPS.')
        parser.add_argument('--block-sizes', default=None,  type=int, nargs='*', help='A list of sizes of different blocks. Must sum up to the total number of layers in the network.')
        parser.add_argument('--estimation-batch', default=None, type=int, help='Batch size for bound estimation.')
        parser.add_argument('--soft-thre', default=0.5, type=float, help='The hyperparameter of soft gradient link. Equivalent to c in TAPS paper.')
        parser.add_argument('--taps-grad-scale', default=1, type=float, help='The gradient scale of TAPS gradient w.r.t. box gradient. Equivalent to w in TAPS paper.')
        parser.add_argument('--no-ibp-anneal', action='store_true', help='Whether to use IBP for annealing. Typically used for checking whether TAPS is out-of-memory. Use IBP for eps annealing can increase performance in general.')
        parser.add_argument('--no-ibp-multiplier', action='store_true', help='Whether to disable IBP*TAPS as the training loss. Using the multiplication loss can increase certifiability.')

        # DeepPoly training
        parser.add_argument('--use-DP-training', action='store_true', help='Whether to use DeepPoly.')
        parser.add_argument('--use-DPBox-training', action='store_true', help='Whether to use CROWN-IBP.')
        parser.add_argument('--use-loss-fusion', action='store_true', help='Whether to use loss fusion for CROWN-IBP.')
        parser.add_argument('--keep-fusion-when-test', action='store_true', help='Whether to use loss fusion for CROWN-IBP during test.')

        # propagation invariance reg
        parser.add_argument('--PI-reg', default=0, type=float, help="The weight for propagation invariance regularization.")
        parser.add_argument('--PI-detach-opt', action='store_true', help='Whether to detach opt.')
        parser.add_argument('--PI-relu-adjust', default="local", type=str, choices=["local", "center", "dead", "random_value_avg", "adv", "weighted_adv"], help='The type of relu adjustment for propagation invariance regularization.')
        parser.add_argument('--weighted_adv_PI_inv_temp', default=1, type=float, help="The inverse temperature for weighted adv PI reg.")


    if "cert" in include:
        # certify
        parser.add_argument('--load-certify-file', default=None, type=str, help='the certify file to load. A single filename in the same directory as the model.')
        parser.add_argument('--timeout', default=1000, type=float, help='the time limit for certifying one label.')
        parser.add_argument('--mnbab-config', default=None, type=str, help='the config file for MN-BaB.')
        parser.add_argument('--tolerate-error', action='store_true', help='Whether to ignore MNBaB errors. Normally these are memory overflows.')
        parser.add_argument('--use-autoattack', action='store_true', help='Whether to invoke AutoAttack. Slightly larger batch size is recommended to reduce amortized cost.')
        parser.add_argument('--disable-mnbab', action='store_true', help='Whether to disable MNBaB certification. As a result, it will only invoke IBP, DPBox and the adversarial attack specified.')
        parser.add_argument('--dp-only', action='store_true', help='Whether to disable alpha, prima and Bab. As a result, MNBab certification will be exactly DeepPoly. When combined with disable-mnbab, this option will have no effect.')


        parser.add_argument('--start-idx', default=0, type=int, help='the start index of the input in the test dataset (inclusive).')
        parser.add_argument('--end-idx', default=-1, type=int, help='the end index of the input in the test dataset (exclusive). -1 for the end of the dataset.')

    args = parser.parse_args()
    check_args(args, include)

    return args

def check_args(args, include):
    if "train" in include:
        if args.use_taps_training:
            assert args.block_sizes is not None and len(args.block_sizes)==2, "TAPS requires block_sizes to be a list containing 2 integers summing up to the total number of layers."

        if args.end_value_eps == 0:
            args.end_value_eps = args.train_eps
        if args.estimation_batch is None:
            args.estimation_batch = args.train_batch
        if args.relu_shrinkage is not None:
            assert 0 <= args.relu_shrinkage <= 1, "Shrinkage must be between 0 and 1."

        if args.use_swa:
            assert args.model_selection is None, "Model selection is impossible for Stochastic Weighted Average optimizer."
            logging.info("Using SWA optimizer.")

        if args.use_loss_fusion:
            assert args.use_DPBox_training, "Loss fusion is only available for CROWN-IBP."
    
        assert args.use_sam + args.use_weight_smooth <= 1, "Only one of SAM and weight smoothing can be used."

    if "cert" in include:
        assert args.load_model is not None, "A saved model is required to be loaded."
        assert (args.start_idx is None) + (args.end_idx is None) in [0, 2], "If a start idx or end idx is specified, then both must be specified"