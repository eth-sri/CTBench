import torch
import torch.nn as nn
import os
import numpy as np
import time
import math
from args_factory import get_args
from loaders import get_loaders
from networks import get_network, fuse_BN_wrt_Flatten, remove_BN_wrt_Flatten
from model_wrapper import BoxModelWrapper
from attacks import adv_whitebox
from AIDomains.zonotope import HybridZonotope
from AIDomains.abstract_layers import Sequential, Flatten
from AIDomains.concrete_layers import Normalization as PARC_normalize
from AIDomains.wrapper import propagate_abs
from AIDomains.wrapper import construct_C

from utils import write_perf_to_json, load_perf_from_json, fuse_BN

try:
    from autoattack import AutoAttack
except:
    AutoAttack = None

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('prima4complete/')
sys.path.append('prima4complete/ELINA/python_interface/')
try:
    from src.mn_bab_verifier import MNBaBVerifier
    from src.abstract_layers.abstract_network import AbstractNetwork
    from src.utilities.argument_parsing import get_config_from_json
    from src.utilities.config import make_config
    from src.utilities.loading.network import freeze_network
    from src.concrete_layers.normalize import Normalize as mnbab_normalize
    from src.verification_instance import VerificationInstance
    from src.utilities.initialization import seed_everything as mnbab_seed
    from bunch import Bunch
except:
    raise ModuleNotFoundError("MN-BaB not found. Please install it to ./prima4complete/")

try:
    import neptune
except:
    neptune = None
nep_log = None # A global variable to store neptune log

def verify_with_mnbab(net, mnbab_verifier, x, y, eps, norm_mean, norm_std, device, mnbab_config, num_classes:int=10, tolerate_error:bool=False):
    is_verified = torch.zeros(len(x), dtype=torch.bool, device=device)
    is_undecidable = torch.zeros(len(x), dtype=torch.bool, device=device)
    is_attacked = torch.zeros(len(x), dtype=torch.bool, device=device)
    for i in range(len(x)):
        try:
            net.reset_input_bounds()
            net.reset_output_bounds()
            net.reset_optim_input_bounds()
            input = x[i:i+1]
            label = y[i:i+1]
            input_lb = (input - eps).clamp(min=0)
            input_ub = (input + eps).clamp(max=1)
            # normalize the input here
            input = (input - norm_mean) / norm_std
            input_lb = (input_lb - norm_mean) / norm_std
            input_ub = (input_ub - norm_mean) / norm_std
            with torch.enable_grad():
                inst = VerificationInstance.create_instance_for_batch_ver(net, mnbab_verifier, input, input_lb, input_ub, int(label), mnbab_config, num_classes)
                inst.run_instance()
            if inst.is_verified:
                is_verified[i] = True
                print("mnbab verifies a new one!")
            if not inst.is_verified and inst.adv_example is None:
                is_undecidable[i] = True
                print("mnbab cannot decide!")
            if inst.adv_example is not None:
                is_attacked[i] = True
                print("mnbab finds an adex!")
            inst.free_memory()
        except Exception as e:
            if tolerate_error:
                print("mnbab error! Either GPU/CPU memory overflow.")
                is_undecidable[i] = True
                continue
            else:
                raise e
    return is_verified, is_undecidable, is_attacked
        

def update_perf(save_root, args, num_cert_ibp, num_nat_accu, num_cert_dp_box, num_mnbab_verified, num_total, num_adv_attacked, is_nat_cert_accurate, certify_start_time, previous_time, batch_idx, test_loader, postfix=""):
    perf_dict = {
        'num_cert_ibp':num_cert_ibp, 
        'num_nat_accu':num_nat_accu, 
        'num_cert_dpb':num_cert_dp_box,
        'num_cert_mnbab':num_mnbab_verified,
        'num_undecided': num_nat_accu - num_adv_attacked - num_cert_ibp - num_cert_dp_box - num_mnbab_verified,
        'num_total':num_total, 
        'num_adv_attacked':num_adv_attacked,
        'nat_accu': round(num_nat_accu / num_total * 100, 2),
        'ibp_cert_rate': round(num_cert_ibp / num_total * 100, 2),
        'dpb_cert_rate': round(num_cert_dp_box / num_total * 100, 2),
        'mnbab_cert_rate': round(num_mnbab_verified / num_total * 100, 2),
        'adv_unattacked_rate': round((num_nat_accu - num_adv_attacked) / num_total * 100, 2),
        "total_cert_rate": round((num_cert_ibp + num_cert_dp_box + num_mnbab_verified) / num_total * 100, 2),
        "total_time": round(time.time() - certify_start_time + previous_time, 2),
        "batch_remain": len(test_loader) - batch_idx - 1,
        "is_nat_cert_accurate": is_nat_cert_accurate
        }
    write_perf_to_json(perf_dict, save_root, filename=f"cert{postfix}.json")
    write_perf_to_json(args.__dict__, save_root, filename=f"cert_args{postfix}.json")

    if nep_log is not None:
        nep_log['num_cert_ibp'].append(num_cert_ibp)
        nep_log['num_nat_accu'].append(num_nat_accu)
        nep_log['num_cert_dp_box'].append(num_cert_dp_box)
        nep_log['num_cert_mnbab'].append(num_mnbab_verified)
        nep_log['num_total'].append(num_total)
        nep_log['num_adv_attacked'].append(num_adv_attacked)
        nep_log['nat_accu'].append(perf_dict['nat_accu'])
        nep_log['ibp_cert_rate'].append(perf_dict['ibp_cert_rate'])
        nep_log['dpb_cert_rate'].append(perf_dict['dpb_cert_rate'])
        nep_log['mnbab_cert_rate'].append(perf_dict['mnbab_cert_rate'])
        nep_log['adv_unattacked_rate'].append(perf_dict['adv_unattacked_rate'])
        nep_log['total_cert_rate'].append(perf_dict['total_cert_rate'])

    return perf_dict

def transform_abs_into_torch(abs_net, torch_net):
    '''
    load the params in the abs_net into torch net
    '''
    abs_state = abs_net.state_dict()
    torch_state = {}
    for key, value in abs_state.items():
        key = key.lstrip("layers.")
        if key == "0.sigma":
            key = "0.std"
        torch_state[key] = value

    torch_net.load_state_dict(torch_state)
    return torch_net

# def switch_normalization_version(torch_net):
#     '''
#     Using the normalization layer defined in MN-BaB instead
#     '''
#     for i, layer in enumerate(torch_net):
#         if isinstance(layer, PARC_normalize):
#             mnbab_layer = mnbab_normalize(layer.mean, layer.std, channel_dim=1)
#             torch_net[i] = mnbab_layer
#     return torch_net


def run(args):
    # neptune logging
    global nep_log, neptune
    if args.enable_neptune:
        assert neptune is not None, "Neptune is not installed."
        nep_log = neptune.init_run(project=args.neptune_project, tags=args.neptune_tags)
    else:
        neptune = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # set up MN-BaB config
    config_file = get_config_from_json(args.mnbab_config)
    if not hasattr(config_file, "outer_verifier"):
        config_file.outer_verifier = Bunch()
    config_file.outer_verifier.adversarial_attack = False # don't run adversarial attack in MN-BaB; we have it before
    if args.dp_only:
        config_file.run_BaB = False
        config_file.optimize_alpha = False
        config_file.optimize_prima = False
    mnbab_config = make_config(**config_file)

    loaders, input_size, input_channel, n_class = get_loaders(args, shuffle_test=False) # don't shuffle the test set to make sure we can continue from a breakpoint
    input_dim = (input_channel, input_size, input_size)

    if len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
    else:
        train_loader, test_loader = loaders
        val_loader = None

    torch_net = get_network(args.net, args.dataset, device)
    torch_net.eval()
    net = Sequential.from_concrete_network(torch_net, input_dim, disconnect=True)
    net.eval()

    assert os.path.isfile(args.load_model), f"There is no such file {args.load_model}."
    save_root = os.path.dirname(args.load_model) # save in the same directory as the model checkpoint
    net.load_state_dict(torch.load(args.load_model, map_location=device))
    print(f"Loaded {args.load_model}")

    # merge BN into linear/conv layers for the loaded model to avoid overhead
    net = fuse_BN_wrt_Flatten(net, device, remove_all=True)
    # use BoxModelWrapper to compute natural accuracy and IBP certified accuracy
    model_wrapper = BoxModelWrapper(net, nn.CrossEntropyLoss(), (input_channel, input_size, input_size), device, args)
    model_wrapper.summary_accu_stat = False
    model_wrapper.robust_weight = 0
    model_wrapper.net.eval()
    model_wrapper.net.set_dim(torch.zeros((test_loader.batch_size, *input_dim), device=device))
    print(net)

    eps = args.test_eps
    print("Certifying for eps:", eps)

    # prepare statistics
    num_cert_ibp, num_nat_accu, num_cert_dp_box, num_mnbab_verified, num_total, num_adv_attacked = 0, 0, 0, 0, 0, 0
    previous_time = 0
    is_nat_cert_accurate = []
    if args.load_certify_file:
        perf_dict = load_perf_from_json(save_root, args.load_certify_file)
        if perf_dict is not None:
            num_cert_ibp, num_nat_accu, num_cert_dp_box, num_mnbab_verified, num_total, num_adv_attacked, previous_time, is_nat_cert_accurate = perf_dict['num_cert_ibp'], perf_dict['num_nat_accu'], perf_dict['num_cert_dpb'],perf_dict['num_cert_mnbab'], perf_dict['num_total'], perf_dict['num_adv_attacked'], perf_dict["total_time"], perf_dict["is_nat_cert_accurate"]
    temp_total_certified = num_cert_ibp + num_cert_dp_box + num_mnbab_verified
    assert num_total == len(is_nat_cert_accurate) and num_total >= num_nat_accu and num_nat_accu >= temp_total_certified + num_adv_attacked, "The loaded certify file is not consistent. This suggests corruption or manual modification. Please check the file and remove it if necessary."
    if num_total > 0:
        assert num_nat_accu == sum([int(i[0]) for i in is_nat_cert_accurate]) and temp_total_certified == sum([int(i[1]) for i in is_nat_cert_accurate]), "The loaded certify file is not consistent. This suggests corruption or manual modification. Please check the file and remove it if necessary."

    # prepare the mn-bab verifier
    # torch net is not loaded with the checkpoint, so we can simply adjust the structure to a non-BN one
    torch_net = remove_BN_wrt_Flatten(torch_net, device, remove_all=True)
    # load the abs net weights into the non-BN torch net
    torch_net = transform_abs_into_torch(net, torch_net)
    mnbab_net = AbstractNetwork.from_concrete_module(
        torch_net[1:], mnbab_config.input_dim
    ).to(device) # remove normalization layer, which would be done directly to its input
    freeze_network(mnbab_net)
    mnbab_verifier = MNBaBVerifier(mnbab_net, device, mnbab_config.verifier)

    # parse the start and end of the certify loop
    assert args.start_idx >= 0, "Start index must be a non-negative integer."
    assert args.end_idx == -1 or args.end_idx>args.start_idx, "End index must be larger than start index or -1."
    postfix = "" if args.start_idx==0 and args.end_idx==-1 else f"{args.start_idx}_{args.end_idx}"
    # the range considered is [start_idx, end_idx)
    current_start_idx = args.start_idx + num_total
    current_end_idx = args.end_idx if args.end_idx != -1 else math.inf

    # main certify loop
    certify_start_time = time.time()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            # check whether this batch is in the range considered
            batch_start, batch_end = batch_idx*args.test_batch, (batch_idx+1)*args.test_batch # [batch_start, batch_end)
            if batch_end <= current_start_idx:
                continue
            elif batch_start >= current_end_idx:
                break
            else:
                # has at least part of the batch in the range
                subbatch_start = max(current_start_idx - batch_start, 0)
                subbatch_end = min(current_end_idx - batch_start, args.test_batch)
                x = x[subbatch_start:subbatch_end]
                y = y[subbatch_start:subbatch_end]

            print("Batch id:", batch_idx)
            model_wrapper.net = model_wrapper.net.to(device)
            x, y = x.to(device), y.to(device)
            # 1. try to verify with IBP 
            _, _, (is_nat_accu, is_IBP_cert_accu) = model_wrapper.compute_model_stat(x, y, eps)
            num_nat_accu += is_nat_accu.sum().item()
            num_cert_ibp += is_IBP_cert_accu.sum().item()
            num_total += len(x)
            print(f"Batch size: {len(x)}, Nat accu: {is_nat_accu.sum().item()}, IBP cert: {is_IBP_cert_accu.sum().item()}")

            # only consider classified correct and not IBP verified below
            x = x[is_nat_accu & (~is_IBP_cert_accu)]
            y = y[is_nat_accu & (~is_IBP_cert_accu)]
            kept_idx = torch.where(is_nat_accu & (~is_IBP_cert_accu))[0]
            is_cert_accu = is_IBP_cert_accu.clone().detach() # add IBP certified ones to the list
            if len(x) == 0:
                is_nat_cert_accurate += [f"{int(is_nat_accu[i].item())}{int(is_cert_accu[i].item())}" for i in range(len(is_nat_accu))]
                perf_dict = update_perf(save_root, args, num_cert_ibp, num_nat_accu, num_cert_dp_box, num_mnbab_verified, num_total, num_adv_attacked, is_nat_cert_accurate, certify_start_time, previous_time, batch_idx, test_loader, postfix)
                continue

            # 2. try to verify with dp_box
            data_abs = HybridZonotope.construct_from_noise(x, eps, "box")
            dpb, pesudo_label = propagate_abs(model_wrapper.net, "deeppoly_box", data_abs, y)
            is_dpb_cert = (dpb.argmax(1) == pesudo_label)
            num_cert_dp_box += is_dpb_cert.sum().item()
            print(f"  DPB cert: {is_dpb_cert.sum().item()}")

            # only consider not dpb verified below
            for sample_idx, verified in zip(kept_idx, is_dpb_cert):
                is_cert_accu[sample_idx] = verified
            x = x[~is_dpb_cert]
            y = y[~is_dpb_cert]
            kept_idx = kept_idx[torch.where(~is_dpb_cert)[0]]
            if len(x) == 0:
                is_nat_cert_accurate += [f"{int(is_nat_accu[i].item())}{int(is_cert_accu[i].item())}" for i in range(len(is_nat_accu))]
                perf_dict = update_perf(save_root, args, num_cert_ibp, num_nat_accu, num_cert_dp_box, num_mnbab_verified, num_total, num_adv_attacked, is_nat_cert_accurate, certify_start_time, previous_time, batch_idx, test_loader, postfix)
                continue

            # 3. try to attack with pgd
            is_adv_attacked = torch.zeros(len(x), dtype=torch.bool, device=device)
            if args.use_autoattack:
                assert AutoAttack is not None, "AutoAttack is not installed."
                adversary = AutoAttack(model_wrapper.net, norm='Linf', eps=eps, version="standard", device=device)
                x_adv = adversary.run_standard_evaluation(x, y)
            else:
                x_adv = adv_whitebox(model_wrapper.net, x, y, (x-eps).clamp(min=0), (x+eps).clamp(max=1), device, lossFunc='pgd', restarts=5, num_steps=args.test_steps)
            y_adv = model_wrapper.net(x_adv).argmax(dim=1)
            is_adv_attacked[(y_adv != y)] = True
            num_adv_attacked += is_adv_attacked.sum().item()
            print(f"  Adv attacked: {is_adv_attacked.sum().item()}")

            # only consider not adv attacked below
            x = x[~is_adv_attacked]
            y = y[~is_adv_attacked]
            kept_idx = kept_idx[torch.where(~is_adv_attacked)[0]]
            if len(x) == 0:
                is_nat_cert_accurate += [f"{int(is_nat_accu[i].item())}{int(is_cert_accu[i].item())}" for i in range(len(is_nat_accu))]
                perf_dict = update_perf(save_root, args, num_cert_ibp, num_nat_accu, num_cert_dp_box, num_mnbab_verified, num_total, num_adv_attacked, is_nat_cert_accurate, certify_start_time, previous_time, batch_idx, test_loader, postfix)
                continue


            if not args.disable_mnbab:
                # 4. try to verify with MN-BaB
                is_mnbab_verified, is_undecidable, is_mnbab_attacked = verify_with_mnbab(mnbab_net, mnbab_verifier, x, y, eps, torch_net[0].mean, torch_net[0].std, device, mnbab_config, n_class, tolerate_error=args.tolerate_error)
                num_mnbab_verified += is_mnbab_verified.sum().item()
                x = x[is_undecidable]
                y = y[is_undecidable]
                num_adv_attacked += is_mnbab_attacked.sum().item()

                for sample_idx, verified in zip(kept_idx, is_mnbab_verified):
                    is_cert_accu[sample_idx] = verified

            is_nat_cert_accurate += [f"{int(is_nat_accu[i].item())}{int(is_cert_accu[i].item())}" for i in range(len(is_nat_accu))]
            perf_dict = update_perf(save_root, args, num_cert_ibp, num_nat_accu, num_cert_dp_box, num_mnbab_verified, num_total, num_adv_attacked, is_nat_cert_accurate, certify_start_time, previous_time, batch_idx, test_loader, postfix)


        perf_dict = update_perf(save_root, args, num_cert_ibp, num_nat_accu, num_cert_dp_box, num_mnbab_verified, num_total, num_adv_attacked, is_nat_cert_accurate, certify_start_time, previous_time, batch_idx, test_loader, postfix)
        write_perf_to_json(perf_dict, save_root, filename=f"complete_cert{postfix}.json")


        

def main():
    args = get_args(["basic", "cert"])
    run(args)

if __name__ == '__main__':
    mnbab_seed(123)
    main()


