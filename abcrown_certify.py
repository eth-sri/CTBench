import os
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"

import torch
import torch.nn as nn
import time
import math
from args_factory import get_args
from loaders import get_loaders
from networks import get_network, fuse_BN_wrt_Flatten, remove_BN_wrt_Flatten
from model_wrapper import BoxModelWrapper
from AIDomains.abstract_layers import Sequential
from AIDomains.zonotope import HybridZonotope
from AIDomains.wrapper import propagate_abs
from abcrown_utils import (
    copy_certify_artifacts,
    load_abcrown_yaml_and_eps,
    prepare_abcrown_config,
    run_autoattack,
    transform_abs_into_torch,
    verify_with_abcrown,
)

from utils import write_perf_to_json, load_perf_from_json, seed_everything

import warnings
warnings.filterwarnings("ignore")

try:
    import neptune
except:
    neptune = None
nep_log = None # A global variable to store neptune log

def update_perf(save_root, args, num_cert_ibp, num_nat_accu, num_heuristic_dpb, num_alpha_crown, num_abcrown_bab, num_total, num_autoattack_attacked, num_abcrown_pgd_attacked, num_bab_rejected, is_nat_cert_accurate, certify_start_time, previous_time, batch_idx, test_loader, postfix="", end_idx=math.inf):
    num_external_attacked = num_autoattack_attacked
    num_internal_attacked = num_abcrown_pgd_attacked
    num_unsafe = num_external_attacked + num_internal_attacked + num_bab_rejected
    has_autoattack = getattr(args, "use_autoattack", False) or num_autoattack_attacked > 0
    perf_dict = {
        'num_cert_ibp':num_cert_ibp,
        'num_nat_accu':num_nat_accu,
        'num_heuristic_dpb':num_heuristic_dpb,
        'num_cert_alpha_crown':num_alpha_crown,
        'num_cert_abcrown':num_abcrown_bab,
        'num_undecided': num_nat_accu - num_unsafe - num_cert_ibp - num_heuristic_dpb - num_alpha_crown - num_abcrown_bab,
        'num_total':num_total,
        'num_adv_attacked':num_external_attacked,
        'num_autoattack_attacked':num_autoattack_attacked,
        'num_abcrown_pgd_attacked':num_abcrown_pgd_attacked,
        'num_abcrown_pgd_unsafe':num_abcrown_pgd_attacked,
        'num_bab_rejected':num_bab_rejected,
        'nat_accu': round(num_nat_accu / num_total * 100, 2) if num_total > 0 else 0,
        'ibp_cert_rate': round(num_cert_ibp / num_total * 100, 2) if num_total > 0 else 0,
        'heuristic_dpb_cert_rate': round(num_heuristic_dpb / num_total * 100, 2) if num_total > 0 else 0,
        'alpha_crown_cert_rate': round(num_alpha_crown / num_total * 100, 2) if num_total > 0 else 0,
        'abcrown_bab_cert_rate': round(num_abcrown_bab / num_total * 100, 2) if num_total > 0 else 0,
        'adv_unattacked_rate': round((num_nat_accu - num_external_attacked) / num_total * 100, 2) if num_total > 0 else 0,
        'autoattack_adv_accuracy': round((num_nat_accu - num_autoattack_attacked) / num_total * 100, 2) if num_total > 0 and has_autoattack else None,
        'abcrown_pgd_unsafe_rate': round(num_abcrown_pgd_attacked / num_total * 100, 2) if num_total > 0 else 0,
        "total_cert_rate": round((num_cert_ibp + num_heuristic_dpb + num_alpha_crown + num_abcrown_bab) / num_total * 100, 2) if num_total > 0 else 0,
        "total_time": round(time.time() - certify_start_time + previous_time, 2),
        "batch_remain": math.ceil(end_idx / args.test_batch) - batch_idx - 1 if end_idx != math.inf else len(test_loader) - batch_idx - 1,
        "is_nat_cert_accurate": is_nat_cert_accurate
        }
    write_perf_to_json(perf_dict, save_root, filename=f"cert{postfix}.json")
    write_perf_to_json(args.__dict__, save_root, filename=f"cert_args{postfix}.json")

    if nep_log is not None:
        nep_log['num_cert_ibp'].append(num_cert_ibp)
        nep_log['num_nat_accu'].append(num_nat_accu)
        nep_log['num_heuristic_dpb'].append(num_heuristic_dpb)
        nep_log['num_cert_alpha_crown'].append(num_alpha_crown)
        nep_log['num_cert_abcrown'].append(num_abcrown_bab)
        nep_log['num_total'].append(num_total)
        nep_log['num_adv_attacked'].append(num_external_attacked)
        nep_log['num_autoattack_attacked'].append(num_autoattack_attacked)
        nep_log['num_abcrown_pgd_attacked'].append(num_abcrown_pgd_attacked)
        nep_log['nat_accu'].append(perf_dict['nat_accu'])
        nep_log['ibp_cert_rate'].append(perf_dict['ibp_cert_rate'])
        nep_log['heuristic_dpb_cert_rate'].append(perf_dict['heuristic_dpb_cert_rate'])
        nep_log['alpha_crown_cert_rate'].append(perf_dict['alpha_crown_cert_rate'])
        nep_log['abcrown_bab_cert_rate'].append(perf_dict['abcrown_bab_cert_rate'])
        nep_log['adv_unattacked_rate'].append(perf_dict['adv_unattacked_rate'])
        if perf_dict['autoattack_adv_accuracy'] is not None:
            nep_log['autoattack_adv_accuracy'].append(perf_dict['autoattack_adv_accuracy'])
        nep_log['abcrown_pgd_unsafe_rate'].append(perf_dict['abcrown_pgd_unsafe_rate'])
        nep_log['total_cert_rate'].append(perf_dict['total_cert_rate'])

    return perf_dict

def run(args):
    # neptune logging
    global nep_log, neptune
    if args.enable_neptune:
        assert neptune is not None, "Neptune is not installed."
        nep_log = neptune.init_run(project=args.neptune_project, tags=args.neptune_tags)
    else:
        neptune = None

    seed_everything(args.random_seed, strict=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    loaders, input_size, input_channel, n_class = get_loaders(args, shuffle_test=False) 
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
    # Use --save-dir if provided, otherwise fall back to the model checkpoint directory
    if hasattr(args, 'save_dir') and args.save_dir:
        save_root = args.save_dir
        os.makedirs(save_root, exist_ok=True)
    else:
        save_root = os.path.dirname(args.load_model)
    net.load_state_dict(torch.load(args.load_model, map_location=device))
    print(f"Loaded {args.load_model}")

    # Read alpha-beta-CROWN config (if needed) and resolve epsilon before creating BoxModelWrapper
    abcrown_yaml, eps = load_abcrown_yaml_and_eps(args)
    print("Certifying for eps:", eps)

    # merge BN into linear/conv layers for the loaded model to avoid overhead
    net = fuse_BN_wrt_Flatten(net, device, remove_all=True)
    # use BoxModelWrapper to compute natural accuracy and IBP certified accuracy
    model_wrapper = BoxModelWrapper(net, nn.CrossEntropyLoss(), (input_channel, input_size, input_size), device, args)
    model_wrapper.summary_accu_stat = False
    model_wrapper.robust_weight = 0
    model_wrapper.net.eval()
    model_wrapper.net.set_dim(torch.zeros((test_loader.batch_size, *input_dim), device=device))
    print(net)

    torch_net = remove_BN_wrt_Flatten(torch_net, device, remove_all=True)
    torch_net = transform_abs_into_torch(net, torch_net)
    torch_net = torch_net.to(device)
    torch_net.eval()

    if args.enable_heuristic_dpb is not None:
        enable_heuristic_dpb = args.enable_heuristic_dpb
    else:
        enable_heuristic_dpb = False
    if enable_heuristic_dpb:
        print("Heuristic DeepPoly evaluation is ENABLED.")

    # parse the start and end of the certify loop (must happen before resume logic)
    assert args.start_idx >= 0, "Start index must be a non-negative integer."
    assert args.end_idx == -1 or args.end_idx>args.start_idx, "End index must be larger than start index or -1."
    postfix = "" if args.start_idx==0 and args.end_idx==-1 else f"_{args.start_idx}_{args.end_idx}"

    # --- Resume logic: early exit for completed shards ---
    certify_file_to_load = None
    if args.load_certify_directory:
        resume_dir = args.load_certify_directory
        complete_path = os.path.join(resume_dir, f"complete_cert{postfix}.json")
        partial_path = os.path.join(resume_dir, f"cert{postfix}.json")
        if os.path.isfile(complete_path):
            shard_label = f"[{args.start_idx}, {args.end_idx})" if postfix else "full"
            copy_certify_artifacts(resume_dir, save_root, postfix, complete=True)
            print(f"INFO: Shard {shard_label} already complete ({complete_path}), skipping.")
            return
        elif os.path.isfile(partial_path):
            copy_certify_artifacts(resume_dir, save_root, postfix, complete=False)
            certify_file_to_load = partial_path
        # else: no file found, start fresh (no message needed)
    elif args.load_certify_file:
        print("WARNING: --load-certify-file is deprecated. Use --load-certify-directory instead.")
        certify_file_to_load = os.path.join(save_root, args.load_certify_file)
        if not os.path.isfile(certify_file_to_load):
            certify_file_to_load = None

    # Prepare alpha-beta-CROWN configurations once (only if abcrown is enabled)
    if not args.disable_abcrown:
        model_path, yaml_tmp_path = prepare_abcrown_config(torch_net, abcrown_yaml, args, device)
    
    # prepare statistics
    num_cert_ibp, num_nat_accu, num_heuristic_dpb, num_alpha_crown, num_abcrown_bab, num_total = 0, 0, 0, 0, 0, 0
    num_autoattack_attacked, num_abcrown_pgd_attacked, num_bab_rejected = 0, 0, 0
    previous_time = 0
    is_nat_cert_accurate = []

    # --- Resume logic: load partial state ---
    if certify_file_to_load is not None:
        perf_dict = load_perf_from_json(os.path.dirname(certify_file_to_load), os.path.basename(certify_file_to_load))
        if perf_dict is not None:
            num_cert_ibp = perf_dict.get('num_cert_ibp', 0)
            num_nat_accu = perf_dict.get('num_nat_accu', 0)
            num_heuristic_dpb = perf_dict.get('num_heuristic_dpb', 0)
            num_alpha_crown = perf_dict.get('num_cert_alpha_crown', perf_dict.get('num_cert_dpb', 0))
            num_abcrown_bab = perf_dict.get('num_cert_abcrown', 0)
            num_total = perf_dict.get('num_total', 0)
            if 'num_autoattack_attacked' in perf_dict or 'num_abcrown_pgd_attacked' in perf_dict:
                num_autoattack_attacked = perf_dict.get('num_autoattack_attacked', perf_dict.get('num_adv_attacked', 0))
                num_abcrown_pgd_attacked = perf_dict.get('num_abcrown_pgd_attacked', perf_dict.get('num_abcrown_pgd_unsafe', 0))
            else:
                num_autoattack_attacked = 0
                num_abcrown_pgd_attacked = perf_dict.get('num_adv_attacked', 0)
            num_bab_rejected = perf_dict.get('num_bab_rejected', 0)
            previous_time = perf_dict.get('total_time', 0)
            is_nat_cert_accurate = perf_dict.get('is_nat_cert_accurate', [])
            print(f"WARNING: Resuming certification from {certify_file_to_load}")
            print(f"  Loaded: num_total={num_total}, num_nat_accu={num_nat_accu}, "
                  f"num_cert_ibp={num_cert_ibp}, num_heuristic_dpb={num_heuristic_dpb}, "
                  f"num_alpha_crown={num_alpha_crown}, num_abcrown_bab={num_abcrown_bab}, "
                  f"num_autoattack_attacked={num_autoattack_attacked}, "
                  f"num_abcrown_pgd_attacked={num_abcrown_pgd_attacked}, "
                  f"num_bab_rejected={num_bab_rejected}")
            print(f"  Will continue from sample index {args.start_idx + num_total}.")
            print(f"  If this is unexpected, delete {certify_file_to_load} and re-run.")

    temp_total_certified = num_cert_ibp + num_heuristic_dpb + num_alpha_crown + num_abcrown_bab
    temp_total_unsafe = num_autoattack_attacked + num_abcrown_pgd_attacked + num_bab_rejected
    assert num_total == len(is_nat_cert_accurate) and num_total >= num_nat_accu and num_nat_accu >= temp_total_certified + temp_total_unsafe, "The loaded certify file is not consistent. This suggests corruption or manual modification. Please check the file and remove it if necessary."
    if num_total > 0:
        assert num_nat_accu == sum([int(i[0]) for i in is_nat_cert_accurate]) and temp_total_certified == sum([int(i[1]) for i in is_nat_cert_accurate]), "The loaded certify file is not consistent. This suggests corruption or manual modification. Please check the file and remove it if necessary."

    # the range considered is [start_idx, end_idx)
    current_start_idx = args.start_idx + num_total
    current_end_idx = args.end_idx if args.end_idx != -1 else math.inf

    # Truncate the abcrown log file at the start of a fresh run (append mode is used per-subprocess)
    # On resume, keep existing logs so prior subprocess output is preserved
    if not args.disable_abcrown and getattr(args, "subprocess_verbosity", "summary") != "ignore":
        log_file_path = os.path.join(save_root, f"abcrown_log{postfix}.txt")
        if certify_file_to_load is None:
            open(log_file_path, "w").close()

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

            # Filter correctly classified and not IBP verified
            x = x[is_nat_accu & (~is_IBP_cert_accu)]
            y = y[is_nat_accu & (~is_IBP_cert_accu)]
            kept_idx = torch.where(is_nat_accu & (~is_IBP_cert_accu))[0]
            is_cert_accu = is_IBP_cert_accu.clone().detach() # add IBP certified ones to the list
            if len(x) == 0:
                is_nat_cert_accurate += [f"{int(is_nat_accu[i].item())}{int(is_cert_accu[i].item())}" for i in range(len(is_nat_accu))]
                perf_dict = update_perf(save_root, args, num_cert_ibp, num_nat_accu, num_heuristic_dpb, num_alpha_crown, num_abcrown_bab, num_total, num_autoattack_attacked, num_abcrown_pgd_attacked, num_bab_rejected, is_nat_cert_accurate, certify_start_time, previous_time, batch_idx, test_loader, postfix, end_idx=current_end_idx)
                continue

            # 2. try to verify with dp_box
            if enable_heuristic_dpb:
                data_abs = HybridZonotope.construct_from_noise(x, eps, "box")
                dpb, pesudo_label = propagate_abs(model_wrapper.net, "deeppoly_box", data_abs, y)
                is_dpb_cert = (dpb.argmax(1) == pesudo_label)
                num_heuristic_dpb += is_dpb_cert.sum().item()
                print(f"  Heuristic DPB cert: {is_dpb_cert.sum().item()}")

                # only consider not dpb verified below
                for sample_idx, verified in zip(kept_idx, is_dpb_cert):
                    is_cert_accu[sample_idx] = is_cert_accu[sample_idx] | verified
                x = x[~is_dpb_cert]
                y = y[~is_dpb_cert]
                kept_idx = kept_idx[torch.where(~is_dpb_cert)[0]]

                if len(x) == 0:
                    is_nat_cert_accurate += [f"{int(is_nat_accu[i].item())}{int(is_cert_accu[i].item())}" for i in range(len(is_nat_accu))]
                    perf_dict = update_perf(save_root, args, num_cert_ibp, num_nat_accu, num_heuristic_dpb, num_alpha_crown, num_abcrown_bab, num_total, num_autoattack_attacked, num_abcrown_pgd_attacked, num_bab_rejected, is_nat_cert_accurate, certify_start_time, previous_time, batch_idx, test_loader, postfix, end_idx=current_end_idx)
                    continue

            if args.use_autoattack:
                # 3. run an external attack before alpha-beta-CROWN so attack
                # statistics are separated from verifier-internal PGD.
                is_autoattack_attacked = run_autoattack(model_wrapper.net, x, y, eps, device, args)
                num_autoattack_attacked += is_autoattack_attacked.sum().item()
                print(f"  AutoAttack attacked: {is_autoattack_attacked.sum().item()}")

                x = x[~is_autoattack_attacked]
                y = y[~is_autoattack_attacked]
                kept_idx = kept_idx[torch.where(~is_autoattack_attacked)[0]]

                if len(x) == 0:
                    is_nat_cert_accurate += [f"{int(is_nat_accu[i].item())}{int(is_cert_accu[i].item())}" for i in range(len(is_nat_accu))]
                    perf_dict = update_perf(save_root, args, num_cert_ibp, num_nat_accu, num_heuristic_dpb, num_alpha_crown, num_abcrown_bab, num_total, num_autoattack_attacked, num_abcrown_pgd_attacked, num_bab_rejected, is_nat_cert_accurate, certify_start_time, previous_time, batch_idx, test_loader, postfix, end_idx=current_end_idx)
                    continue

            if not args.disable_abcrown:
                # 4. try to verify with alpha-beta-CROWN
                log_file_path = os.path.join(save_root, f"abcrown_log{postfix}.txt") if getattr(args, "subprocess_verbosity", "summary") != "ignore" else None
                abc_dpb, abc_cert, abc_adv, abc_rej, abc_undec = verify_with_abcrown(torch_net, x, y, eps, device, config_path=args.abcrown_config, args=args, log_file_path=log_file_path, tolerate_error=args.tolerate_error)
                num_alpha_crown += abc_dpb.sum().item() # safe-incomplete maps to alpha crown explicitly
                num_abcrown_bab += abc_cert.sum().item() # safe maps to bab complete explicitly
                num_abcrown_pgd_attacked += abc_adv.sum().item() # unsafe-pgd maps to verifier-internal PGD explicitly
                num_bab_rejected += abc_rej.sum().item() # unsafe (BaB) maps to bab rejected explicitly

                print(f"  alpha-CROWN cert: {abc_dpb.sum().item()}")
                print(f"  abCROWN PGD unsafe: {abc_adv.sum().item()}")
                print(f"  Rejected (BaB): {abc_rej.sum().item()}")
                print(f"  alpha-beta-CROWN cert: {abc_cert.sum().item()}")

                # Record certification status
                for sample_idx, verified in zip(kept_idx, abc_dpb | abc_cert):
                    is_cert_accu[sample_idx] = is_cert_accu[sample_idx] | verified

            is_nat_cert_accurate += [f"{int(is_nat_accu[i].item())}{int(is_cert_accu[i].item())}" for i in range(len(is_nat_accu))]
            perf_dict = update_perf(save_root, args, num_cert_ibp, num_nat_accu, num_heuristic_dpb, num_alpha_crown, num_abcrown_bab, num_total, num_autoattack_attacked, num_abcrown_pgd_attacked, num_bab_rejected, is_nat_cert_accurate, certify_start_time, previous_time, batch_idx, test_loader, postfix, end_idx=current_end_idx)

        if num_total > 0:
            perf_dict = update_perf(save_root, args, num_cert_ibp, num_nat_accu, num_heuristic_dpb, num_alpha_crown, num_abcrown_bab, num_total, num_autoattack_attacked, num_abcrown_pgd_attacked, num_bab_rejected, is_nat_cert_accurate, certify_start_time, previous_time, batch_idx, test_loader, postfix, end_idx=current_end_idx)
            write_perf_to_json(perf_dict, save_root, filename=f"complete_cert{postfix}.json")
        else:
            print("Warning: No samples were processed in the given range.")

    if not args.disable_abcrown:
        for temp_created_path in [model_path, yaml_tmp_path]:
            if os.path.exists(temp_created_path):
                try:
                    os.remove(temp_created_path)
                except OSError:
                    pass

def main():
    args = get_args(["basic", "cert"])
    run(args)

if __name__ == '__main__':
    main()
