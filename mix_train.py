import torch
import torch.nn as nn
from args_factory import get_args
from loaders import get_loaders
from utils import Scheduler, Statistics
from networks import get_network, fuse_BN_wrt_Flatten, add_BN_wrt_Flatten
from model_wrapper import get_model_wrapper, BasicModelWrapper, PGDModelWrapper, BoxModelWrapper, TAPSModelWrapper, STAPSModelWrapper, SmallBoxModelWrapper, GradAccuFunctionWrapper
import os
from utils import write_perf_to_json, load_perf_from_json, fuse_BN, seed_everything, reset_bn_to_population_statistics
from tqdm.auto import tqdm
import random
import numpy as np
from regularization import compute_fast_reg, compute_vol_reg, compute_L1_reg, compute_PI_reg, compute_neg_reg
import time
from datetime import datetime
from AIDomains.abstract_layers import Sequential
import logging
from AIDomains.zonotope import HybridZonotope
from Utility.SWA import SWA

import warnings
warnings.filterwarnings("ignore")

from get_stat import PI_loop, relu_loop, test_loop

try:
    import neptune
except:
    neptune = None
nep_log = None # A global variable to store neptune log

def train_loop(model_wrapper:BasicModelWrapper, eps_scheduler:Scheduler, robust_weight_scheduler:Scheduler, train_loader, epoch_idx, optimizer, lr_schedular, device, args, verbose:bool=False):
    model_wrapper.net.train()
    model_wrapper.summary_accu_stat = False
    model_wrapper.freeze_BN = False

    if hasattr(model_wrapper, "num_steps"):
        # PGD
        model_wrapper.num_steps = args.train_steps
    elif hasattr(model_wrapper, "latent_search_steps"):
        # TAPS and STAPS
        model_wrapper.latent_search_steps = args.train_steps
    elif hasattr(model_wrapper, "input_search_steps"):
        # SABR (note STAPS also has input_search_steps but will not be updated here)
        model_wrapper.input_search_steps = args.train_steps

    # Design of TAPS: use IBP for annealing to increase speed and performance.
    if isinstance(model_wrapper, TAPSModelWrapper) or (isinstance(model_wrapper, GradAccuFunctionWrapper) and isinstance(model_wrapper.wrapper, TAPSModelWrapper)):
        TAPS_wrapper = model_wrapper.wrapper if isinstance(model_wrapper, GradAccuFunctionWrapper) else model_wrapper
        if not args.no_ibp_anneal:
            TAPS_wrapper.disable_TAPS = True if epoch_idx < args.end_epoch_eps else False
        else:
            TAPS_wrapper.disable_TAPS = False
    model_wrapper.num_steps = args.train_steps

    # Design of fast regularization: use fast reg only for annealing.
    fast_reg = (args.fast_reg > 0) and epoch_idx < args.end_epoch_eps
    if fast_reg:
        # store box bounds for fast reg
        model_wrapper.store_box_bounds = True

    # Define custom tracking of statistics here
    fastreg_stat, nat_accu_stat, robust_accu_stat, loss_stat = Statistics.get_statistics(4, momentum=0.1)

    # Define custom logging behavior for the first epoch here if verbose-first-epoch is set.
    if args.verbose_first_epoch and epoch_idx == 0:
        epoch_perf = {"robust_loss_curve":[], "fast_reg_curve":[], "PI_curve":[]}

    pbar = tqdm(train_loader)
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        eps = eps_scheduler.getcurrent(epoch_idx * len(train_loader) + batch_idx)
        robust_weight = robust_weight_scheduler.getcurrent(epoch_idx * len(train_loader) + batch_idx)
        model_wrapper.robust_weight = robust_weight

        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=args.use_amp):
            (loss, nat_loss, robust_loss), (nat_accu, robust_accu), (is_nat_accu, is_robust_accu) = model_wrapper.compute_model_stat(x, y, eps)
            if verbose:
                print(f"Batch {batch_idx}:", nat_accu, robust_accu, loss.item())
            loss_stat.update(loss.item(), len(x))

            # Define and update additional regularization here
            reg_eps = max(eps, args.min_eps_reg)

            if fast_reg:
                # add fast reg to the loss
                if reg_eps != eps or model_wrapper.net[1].bounds is None:
                    # recompute bounds for fast reg
                    model_wrapper.net.reset_bounds()
                    abs_x = HybridZonotope.construct_from_noise(x, reg_eps, "box")
                    model_wrapper.net(abs_x)
                reg = args.fast_reg * (1 - reg_eps/eps_scheduler.end_value) * compute_fast_reg(model_wrapper.net, reg_eps)
                loss = loss + reg
                fastreg_stat.update(reg.item(), len(x))
            if args.L1_reg > 0:
                loss = loss + args.L1_reg * compute_L1_reg(model_wrapper.net)

            if args.PI_reg > 0:
                advx = model_wrapper.current_advx if hasattr(model_wrapper, "current_advx") else None
                advy = model_wrapper.current_advy if hasattr(model_wrapper, "current_advy") else None
                dim_weight = None if advy is None else torch.softmax(advy * args.weighted_adv_PI_inv_temp, dim=1) * (args.num_classes - 1)
                PI_reg = args.PI_reg * compute_PI_reg(model_wrapper.net, x, y, args.num_classes, relu_adjust=args.PI_relu_adjust, eps=reg_eps, detach_opt=args.PI_detach_opt, advx=advx, dim_weight=dim_weight)
                loss = loss + PI_reg
            else:
                PI_reg = 0

        # Customize the verbose-first-epoch behavior here
        if args.verbose_first_epoch and epoch_idx == 0 and ((batch_idx % int(args.verbose_gap * len(train_loader))) == 0):
            epoch_perf["robust_loss_curve"].append(loss_stat.last)
            epoch_perf["fast_reg_curve"].append(fastreg_stat.last)
            write_perf_to_json(epoch_perf, args.save_root, "first_epoch.json")


        model_wrapper.net.reset_bounds()
        if args.use_amp:
            model_wrapper.grad_scaler.scale(loss).backward()
            model_wrapper.grad_scaler.unscale_(optimizer)
        else:
            loss.backward()
        model_wrapper.grad_postprocess() # can be inherited to customize gradient postprocessing; default: clip gradients

        if args.use_amp:
            model_wrapper.grad_scaler.step(optimizer)
            model_wrapper.grad_scaler.update()
        else:
            optimizer.step()
        model_wrapper.param_postprocess() # can be inherited to customize parameter postprocessing; default: no parameter postprocessing
        if args.lr_schedule == "cyclic":
            lr_schedular.step()
            model_wrapper.current_lr = lr_schedular.get_last_lr()[0]
        nat_accu_stat.update(nat_accu, len(x))
        robust_accu_stat.update(robust_accu, len(x))

        # Update the progress bar here
        postfix_str = f"nat_accu: {nat_accu_stat.avg:.3f}, robust_accu: {robust_accu_stat.avg:.3f}, train_loss: {loss_stat.avg:.3f}"
        pbar.set_postfix_str(postfix_str)

    # reset batch norm (if exists) with population statistics
    if args.use_pop_bn_stats:
        model_wrapper._set_BN(model_wrapper.BNs, True) # allow BN stats to change
        model_wrapper.net = reset_bn_to_population_statistics(model_wrapper.net, train_loader, device)
        model_wrapper._set_BN(model_wrapper.BNs, False) # restore to default

    return nat_accu_stat.avg, robust_accu_stat.avg, eps, fastreg_stat.avg, loss_stat.avg



def get_train_mode(args):
    '''
    Define the name of the training method here.
    '''
    assert args.use_std_training + args.use_pgd_training + args.use_multipgd_training + args.use_arow_training + args.use_mart_training + args.use_ibp_training + args.use_taps_training + args.use_DP_training + args.use_DPBox_training + args.use_mtlibp_training + args.use_expibp_training + args.use_ccibp_training + args.use_adcert_training + args.use_ccdist_training == 1, "Only one training method can be used at a time."
    if args.use_pgd_training:
        if args.use_EDAC_step:
            mode = "EDAC_trained"
        else:
            mode = "PGD_trained"
    elif args.use_multipgd_training:
        mode = "MultiPGD_trained"
    elif args.use_arow_training:
        mode = "ARoW_trained"
    elif args.use_mart_training:
        mode = "MART_trained"
    elif args.use_ibp_training:
        mode = "IBP_trained" if not args.use_small_box else "SABR_trained"
    elif args.use_taps_training:
        mode = "TAPS_trained" if not args.use_small_box else "STAPS_trained"
    elif args.use_DP_training:
        mode = "DP_trained"
    elif args.use_DPBox_training:
        mode = "DPBox_trained"
    elif args.use_mtlibp_training:
        mode = "MTLIBP_trained"
    elif args.use_expibp_training:
        mode = "EXPIBP_trained"
    elif args.use_ccibp_training:
        mode = "CCIBP_trained"
    elif args.use_ccdist_training:
        mode = "CCDIST_trained"
    elif args.use_adcert_training:
        mode = "ADCERT_trained"
    elif args.use_std_training:
        mode = "std_trained"
    else:
        raise NotImplementedError("Unknown training mode.")
    return mode

def parse_save_root(args, mode):
    '''
    Define the save root here.
    '''
    eps_str = f"eps{args.test_eps:.5g}"
    if args.train_eps != args.test_eps:
        eps_str = f"{eps_str}/train_eps{args.train_eps:.5g}"
    init_str = f"init_{args.init}" if args.load_model is None else f"init_pretrained"
    save_root = os.path.join(args.save_dir, args.dataset, eps_str, mode, args.net, init_str)
    if args.end_value_robust_weight != 1:
        save_root = os.path.join(save_root, f"robust_weight_{args.end_value_robust_weight}")
    if args.fast_reg > 0:
        save_root = os.path.join(save_root, f"fast_reg_{args.fast_reg}")
    if args.use_small_box:
        save_root = os.path.join(save_root, f"eps_shrink_{args.eps_shrinkage}")
    if args.relu_shrinkage is not None:
        save_root = os.path.join(save_root, f"relu_shrink_{args.relu_shrinkage}")
    if args.use_taps_training:
        save_root = os.path.join(save_root, f"TAPS_block_{args.block_sizes[-1]}_scale_{args.taps_grad_scale}")
    if args.use_arow_training:
        save_root = os.path.join(save_root, f"arow_reg_{args.arow_reg_weight}_labelsmooth_{args.arow_label_smoothing}")
    if args.use_EDAC_step:
        save_root = os.path.join(save_root, f"EDAC_step_{args.EDAC_step_size}")
    if args.use_mtlibp_training or args.use_expibp_training or args.use_ccibp_training:
        save_root = os.path.join(save_root, f"ibp_coef_{args.ibp_coef}")
    if args.use_adcert_training or args.use_ccdist_training:
        save_root = os.path.join(save_root, f"ibp_coef_{args.ibp_coef}")
        if args.teacher_tag != "adv":
            save_root = os.path.join(save_root, f"teacher_{args.teacher_tag}")
        if args.teacher_input != "clean":
            save_root = os.path.join(save_root, f"teacher_input_{args.teacher_input}")
        if args.use_rslad:
            save_root = os.path.join(save_root, f"rslad")
        if args.use_nat_kl:
            save_root = os.path.join(save_root, f"nat_kl")
    if args.use_softlabel_attack:
       save_root = os.path.join(save_root, f"sl_attack")
    if args.L1_reg > 0:
        save_root = os.path.join(save_root, f"L1_{args.L1_reg}")
    if args.PI_reg > 0:
        save_root = os.path.join(save_root, f"{args.PI_relu_adjust}PI_{args.PI_reg}{'_opt_detach' if args.PI_detach_opt else ''}")
        if args.PI_relu_adjust == "weighted_adv":
            save_root = os.path.join(save_root, f"inv_temp_{args.weighted_adv_PI_inv_temp}")
    if args.use_swa:
        save_root = os.path.join(save_root, f"optim_swa")
    if args.use_amp:
        save_root = os.path.join(save_root, f"amp")
    if args.use_weight_smooth:
        save_root = os.path.join(save_root, f"weight_smooth_{args.weight_smooth_std_scale}")
    if args.use_sam:
        save_root = os.path.join(save_root, f"SAM_{args.sam_rho}{'_ada' if args.adaptive_sam_rho else ''}")
    if args.use_pop_bn_stats:
        save_root = os.path.join(save_root, f"pop_bn_stats")
    os.makedirs(save_root, exist_ok=True)
    args.save_root = save_root
    logging.info(f"The model will be saved at: {save_root}")
    return save_root


def run(args):
    # --- logging ---
    # local logging
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    perf_dict = {'val_nat_curve':[], 'val_robust_curve':[], 'val_loss_curve':[], 'train_nat_curve':[], 'train_robust_curve':[], 'train_loss_curve':[], 'lr_curve':[], "eps_curve":[]}
    PI_dict = {"PI_curve":[]}
    relu_dict = {"dead_relu_curve":[], "active_relu_curve":[], "unstable_relu_curve":[]}
    reg_dict = {"fastreg_curve":[]}

    # Add more perf_dict here to track more statistics.
    perf_dict = perf_dict | PI_dict | relu_dict | reg_dict
    perf_dict["start_time"] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    verbose = False

    # neptune logging
    global nep_log, neptune
    if args.enable_neptune:
        assert neptune is not None, "Neptune is not installed."
        nep_log = neptune.init_run(project=args.neptune_project, tags=args.neptune_tags)
        perf_dict["neptune_id"] = nep_log["sys/id"].fetch()
    else:
        neptune = None

    # --- data loading ---

    # Define dataset here
    loaders, input_size, input_channel, n_class = get_loaders(args)
    input_dim = (input_channel, input_size, input_size)
    args.num_classes = n_class
    if len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
        
    else:
        train_loader, test_loader = loaders
        val_loader = None
    perf_dict['model_selection'] = args.model_selection # tradition for certified training: use test set for model selection :(

    # Adjust the sequence to have different preference of training when the args is ambiguous.
    # Define custom behavior for different training here.
    mode = get_train_mode(args)

    # Schedule for robust weight
    # Assume the loss is in the form (1-w) * natural_loss + w * robust_loss
    robust_weight_scheduler = Scheduler(args.start_epoch_robust_weight*len(train_loader), args.end_epoch_robust_weight*len(train_loader), args.start_value_robust_weight, args.end_value_robust_weight, mode="linear", s=len(train_loader))

    # Schedule for input epsilon
    if args.no_anneal:
        # use const eps == train_eps
        eps_scheduler = Scheduler(args.start_epoch_eps*len(train_loader), args.end_epoch_eps*len(train_loader), args.train_eps, args.train_eps, "linear", s=len(train_loader))
    else:
        if args.schedule in ["smooth", "linear", "step"]:
            eps_scheduler = Scheduler(args.start_epoch_eps*len(train_loader), args.end_epoch_eps*len(train_loader), args.start_value_eps, args.end_value_eps, args.schedule, s=args.step_epoch*len(train_loader))
        else:
            raise NotImplementedError(f"Unknown schedule: {args.schedule}")

    # Define concrete (torch) model and convert it to abstract model here
    torch_net = get_network(args.net, args.dataset, device, init=args.init)
    # summary(net, (input_channel, input_size, input_size))
    net = Sequential.from_concrete_network(torch_net, input_dim, disconnect=False)
    net.set_dim(torch.zeros((test_loader.batch_size, *input_dim), device='cuda'))
    if args.load_model:
        net.load_state_dict(torch.load(args.load_model))
        print("Loaded:", args.load_model)
    print(net)

    # Parse save root here
    save_root = parse_save_root(args, mode)

    # Define model wrapper here: this wraps how to compute the loss and how to compute the robust accuracy.
    model_wrapper = get_model_wrapper(args, net, device, input_dim)
    model_wrapper.max_eps = eps_scheduler.end_value

    # Define training hyperparameter here
    param_list = set(model_wrapper.net.parameters()) - set(model_wrapper.net[0].parameters()) # exclude normalization
    lr = args.lr

    perf_dict["best_val_robust_accu"] = -1
    perf_dict["best_val_loss"] = 1e8

    if args.opt == 'adam':
        optimizer = torch.optim.Adam(param_list, lr=lr)
    elif args.opt == 'sgd':
        if args.use_sgd_momentum:
            optimizer = torch.optim.SGD(param_list, lr=lr, momentum=args.momentum)
        else:
            optimizer = torch.optim.SGD(param_list, lr=lr)
    else:
        raise ValueError(f"{args.opt} not supported.")
    if args.use_swa:
        optimizer = SWA(optimizer, swa_start=args.swa_start*len(train_loader), swa_freq=args.swa_freq, swa_lr=args.swa_lr)
    if args.use_EDAC_step:
        EDAC_optimizer = optimizer if not args.use_swa else optimizer.optimizer
        model_wrapper.register_EDAC_hyperparam(EDAC_optimizer, args.EDAC_step_size)

    if args.lr_schedule == 'cyclic':
        # cyclic lr scheduler adapted from https://github.com/pdejorge/N-FGSM, paper: https://arxiv.org/abs/2202.01181
        lr_steps = args.n_epochs * len(train_loader)
        lr_schedular = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=max(args.lr_min, 1e-8), max_lr=args.lr, step_size_up=lr_steps/2, step_size_down=lr_steps/2)
    else:
        lr_schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_milestones, gamma=args.lr_decay_factor)

    model_wrapper.current_lr = lr_schedular.get_last_lr()[0]
    train_time = 0.0
    for epoch_idx in range(args.n_epochs):
        print("Epoch", epoch_idx)

        #### ---- train loop below ----
        train_start_time = time.time()
        train_nat_accu, train_robust_accu, eps, fast_reg_avg, train_loss = train_loop(model_wrapper, eps_scheduler, robust_weight_scheduler, train_loader, epoch_idx, optimizer, lr_schedular, device, args, verbose=verbose)
        train_time += time.time() - train_start_time
        print(f"train_nat_accu: {train_nat_accu: .4f}, train_robust_accu: {train_robust_accu: .4f}, train_loss:{train_loss: .4f}")

        # Track train statistics here.
        perf_dict["fastreg_curve"].append(fast_reg_avg)
        perf_dict['train_nat_curve'].append(train_nat_accu)
        perf_dict['train_robust_curve'].append(train_robust_accu)
        perf_dict["train_loss_curve"].append(train_loss)

        # Update learning rate here.
        if args.lr_schedule == "multistep":
            lr_schedular.step()
        lr = lr_schedular.get_last_lr()[0]
        perf_dict['lr_curve'].append(lr)
        model_wrapper.current_lr = lr

        eps = min(eps, args.test_eps)
        perf_dict["eps_curve"].append(eps)
        print("current eps:", eps)
        print("current robust_weight:", model_wrapper.robust_weight)

        # train_nat_accu, train_robust_accu, train_loss = test_loop(model_wrapper, eps, train_loader, device, args)
        # print(f"train_nat_accu: {train_nat_accu: .4f}, train_robust_accu: {train_robust_accu: .4f}, train_loss:{train_loss: .4f}")

        #### ---- test loop below ----
        val_nat_accu, val_robust_accu, val_loss = test_loop(model_wrapper, eps, val_loader if val_loader is not None else test_loader, device, args)
        print(f"val_nat_accu: {val_nat_accu: .4f}, val_robust_accu: {val_robust_accu: .4f}, val_loss:{val_loss: .4f}")

        # Track val statistics here.
        perf_dict['val_nat_curve'].append(val_nat_accu)
        perf_dict['val_robust_curve'].append(val_robust_accu)
        perf_dict["val_loss_curve"].append(val_loss)

        # #### ---- additional model statistics tracking below ----
        # -- propagation tightness --
        PI = PI_loop(model_wrapper.net, max(eps, args.min_eps_reg), val_loader if val_loader is not None else test_loader, device, args.num_classes, args, relu_adjust="local")
        print(f"Propagation Tightness: {PI:.3E}")
        perf_dict["PI_curve"].append(PI)

        # -- relu status --
        dead, unstable, active = relu_loop(model_wrapper.net, max(eps, 1e-6), val_loader if val_loader is not None else test_loader, device, args)
        perf_dict["dead_relu_curve"].append(dead)
        perf_dict["unstable_relu_curve"].append(unstable)
        perf_dict["active_relu_curve"].append(active)
        print(f"Dead: {dead:.3f}; Unstable: {unstable:.3f}; Active: {active:.3f}")

        #### ---- model selection below ----
        if eps == args.test_eps:
            if (perf_dict["model_selection"] == "robust_accu" and val_robust_accu > perf_dict["best_val_robust_accu"]) or (perf_dict["model_selection"] == "loss" and val_loss < perf_dict["best_val_loss"]):
                torch.save(model_wrapper.net.state_dict(), os.path.join(save_root, "model.ckpt"))
                print("New checkpoint saved.")
                perf_dict["best_ckpt_epoch"] = epoch_idx
            perf_dict["best_val_robust_accu"] = max(perf_dict["best_val_robust_accu"], val_robust_accu)
            perf_dict["best_val_loss"] = min(perf_dict["best_val_loss"], val_loss)

        if args.save_every_epoch:
            os.makedirs(os.path.join(save_root, "Every_Epoch_Model"), exist_ok=True)
            torch.save(model_wrapper.net.state_dict(), os.path.join(save_root, "Every_Epoch_Model", f"epoch_{epoch_idx}.ckpt"))

        if perf_dict["model_selection"] is None:
            # adjust BN stats for SWA
            if isinstance(optimizer, SWA) and epoch_idx == args.n_epochs - 1:
                optimizer.swap_swa_sgd()
                model_wrapper._set_BN(model_wrapper.BNs, True) # allow BN stats to change
                optimizer.bn_update(train_loader, model_wrapper.net, device=device)
                model_wrapper._set_BN(model_wrapper.BNs, False) # restore to default

            # No model selection. Save the final model.
            torch.save(model_wrapper.net.state_dict(), os.path.join(save_root, "model.ckpt"))
            print("New checkpoint saved.")

        # Write the logs to json file
        write_perf_to_json(perf_dict, save_root, "monitor.json")
        write_perf_to_json(args.__dict__, save_root, "train_args.json")

        # Write the logs to neptune
        if nep_log is not None:
            nep_log["train_nat_accu_curve"].append(train_nat_accu)
            nep_log["train_robust_accu_curve"].append(train_robust_accu)
            nep_log["train_loss_curve"].append(train_loss)
            nep_log["val_nat_accu_curve"].append(val_nat_accu)
            nep_log["val_robust_accu_curve"].append(val_robust_accu)
            nep_log["val_loss_curve"].append(val_loss)
            nep_log["eps_curve"].append(eps)
            nep_log["lr_curve"].append(lr)
            nep_log["PI_curve"].append(PI)
            nep_log["dead_relu_curve"].append(dead)
            nep_log["unstable_relu_curve"].append(unstable)
            nep_log["active_relu_curve"].append(active)
            nep_log["fastreg_curve"].append(fast_reg_avg)

    # test for the best ckpt
    ckpt_name = f"Epoch {perf_dict['best_ckpt_epoch'] if args.model_selection is not None else 'final'}" if not isinstance(optimizer, SWA) else "SWA"
    print("-"*10 + f"Model Selection: {perf_dict['model_selection']}. Testing selected checkpoint ({ckpt_name})." + "-"*10)
    model_wrapper.net.load_state_dict(torch.load(os.path.join(save_root, "model.ckpt")))
    test_nat_accu, test_robust_accu, loss = test_loop(model_wrapper, args.test_eps, test_loader, device, args)
    print(f"test_nat_accu: {test_nat_accu: .4f}, test_robust_accu: {test_robust_accu: .4f}")
    perf_dict["test_nat_accu"] = test_nat_accu
    perf_dict["test_robust_accu"] = test_robust_accu
    perf_dict["train_time"] = train_time
    perf_dict["end_time"] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    
    write_perf_to_json(perf_dict, save_root, "monitor.json")
    write_perf_to_json(args.__dict__, save_root, "train_args.json")

    if nep_log is not None:
        nep_log["test_nat_accu"] = test_nat_accu
        nep_log["test_robust_accu"] = test_robust_accu
        nep_log["train_time"] = train_time
        nep_log["end_time"] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        nep_log["args"] = args.__dict__
        nep_log.stop()


def main():
    args = get_args(["basic", "train"])
    seed_everything(args.random_seed)
    run(args)

if __name__ == '__main__':
    main()
