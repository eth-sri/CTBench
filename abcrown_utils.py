import os
import pickle
import shutil
import subprocess

import torch
import yaml


try:
    from autoattack import AutoAttack
except ImportError:
    AutoAttack = None


def load_abcrown_yaml_and_eps(args):
    abcrown_yaml = None
    if not args.disable_abcrown and args.abcrown_config is not None:
        with open(args.abcrown_config, "r") as f:
            abcrown_yaml = yaml.safe_load(f)

    yaml_eps = None
    if abcrown_yaml is not None:
        yaml_eps = abcrown_yaml.get("specification", {}).get("epsilon", None)

    if args.test_eps is not None:
        eps = args.test_eps
        if abcrown_yaml is not None:
            abcrown_yaml.setdefault("specification", {})["epsilon"] = float(eps)
    elif yaml_eps is not None:
        eps = float(yaml_eps)
        args.test_eps = eps
    else:
        raise ValueError("Epsilon must be specified either via --test-eps or in the YAML config under specification.epsilon.")

    return abcrown_yaml, eps


def prepare_abcrown_config(torch_net, abcrown_yaml, args, device):
    pid = os.getpid()
    tmp_dir = os.path.abspath("../tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    data_path = f"../tmp/ctbench_abcrown_data_{pid}.pt"
    model_path = f"../tmp/ctbench_abcrown_model_{pid}.pt"
    result_file_path = f"../tmp/ctbench_abcrown_results_{pid}.pkl"
    yaml_tmp_path = f"../tmp/ctbench_abcrown_temp_{pid}.yaml"

    torch.save(torch_net, model_path)

    abcrown_yaml.setdefault("attack", {})["pgd_order"] = "skip" if args.disable_abcrown_pgd else "before"

    if hasattr(args, "test_batch"):
        abcrown_batch = args.abcrown_batch if args.abcrown_batch is not None else args.test_batch
        abcrown_yaml.setdefault("solver", {})["batch_size"] = abcrown_batch
    abcrown_yaml.setdefault("general", {})["device"] = device
    if hasattr(args, "dp_only") and args.dp_only:
        abcrown_yaml.setdefault("general", {})["complete_verifier"] = "skip"
        abcrown_yaml.setdefault("solver", {})["bound_prop_method"] = "alpha-crown"

    abcrown_yaml.setdefault("general", {})["results_file"] = result_file_path
    abcrown_yaml.setdefault("general", {})["root_path"] = "../alpha-beta-CROWN"
    abcrown_yaml.setdefault("model", {})["name"] = f'Customized("abcrown_adapter", "get_ctbench_model", model_path="{model_path}")'
    abcrown_yaml.setdefault("data", {})["dataset"] = f'Customized("abcrown_adapter", "get_ctbench_data", data_path="{data_path}")'

    with open(yaml_tmp_path, "w") as f:
        yaml.dump(abcrown_yaml, f)

    return model_path, yaml_tmp_path


def copy_certify_artifacts(src_dir, dst_dir, postfix, complete=False):
    if os.path.abspath(src_dir) == os.path.abspath(dst_dir):
        return

    prefix = "complete_cert" if complete else "cert"
    src_path = os.path.join(src_dir, f"{prefix}{postfix}.json")
    if os.path.isfile(src_path):
        shutil.copy2(src_path, os.path.join(dst_dir, f"{prefix}{postfix}.json"))

    cert_args_path = os.path.join(src_dir, f"cert_args{postfix}.json")
    if os.path.isfile(cert_args_path):
        shutil.copy2(cert_args_path, os.path.join(dst_dir, f"cert_args{postfix}.json"))


def verify_with_abcrown(torch_net, x, y, eps, device, config_path, args, log_file_path=None, tolerate_error=False):
    is_dpb_cert = torch.zeros(len(x), dtype=torch.bool, device=device)
    is_abcrown_cert = torch.zeros(len(x), dtype=torch.bool, device=device)
    is_adv_attacked = torch.zeros(len(x), dtype=torch.bool, device=device)
    is_bab_rejected = torch.zeros(len(x), dtype=torch.bool, device=device)
    is_undecidable = torch.zeros(len(x), dtype=torch.bool, device=device)

    if len(x) == 0:
        return is_dpb_cert, is_abcrown_cert, is_adv_attacked, is_bab_rejected, is_undecidable

    pid = os.getpid()
    data_path = f"../tmp/ctbench_abcrown_data_{pid}.pt"
    torch.save((x, y), data_path)

    result_file_path = f"../tmp/ctbench_abcrown_results_{pid}.pkl"
    if os.path.exists(result_file_path):
        os.remove(result_file_path)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    abcrown_path = "../alpha-beta-CROWN/complete_verifier/abcrown.py"
    try:
        print("Launching alpha-beta-CROWN via subprocess...")
        abcrown_conda_env = getattr(args, "abcrown_conda_env", "unified_ctbench")
        process = subprocess.Popen(["conda", "run", "-n", abcrown_conda_env, "python", abcrown_path, "--config", f"../tmp/ctbench_abcrown_temp_{pid}.yaml"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        in_summary = False
        verbosity = getattr(args, "subprocess_verbosity", "summary")
        log_f = open(log_file_path, "a") if log_file_path else None

        for line in process.stdout:
            if verbosity == "all":
                if log_f:
                    log_f.write(line)
            elif verbosity == "ignore":
                continue
            else:
                if log_f:
                    stripped = line.strip()
                    if stripped and (stripped[0].isdigit() or stripped.startswith("|")) and "it/s]" in line:
                        continue

                    is_relevant = False
                    if "############# Summary #############" in line:
                        in_summary = True
                        is_relevant = True
                    elif in_summary:
                        is_relevant = True
                    elif "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" in line or line.startswith("Result:") or "Result:" in line or "Time out!!!!!!!!" in line:
                        is_relevant = True

                    if is_relevant:
                        log_f.write(line)

        if log_f:
            log_f.close()

        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, process.args)
    except subprocess.CalledProcessError as e:
        if tolerate_error:
            print("alpha-beta-CROWN error! Tolerating and marking as undecidable.")
            is_undecidable[:] = True
            return is_dpb_cert, is_abcrown_cert, is_adv_attacked, is_bab_rejected, is_undecidable
        raise e

    if os.path.exists(result_file_path):
        with open(result_file_path, "rb") as f:
            results = pickle.load(f)

        summary = results.get("summary", {})
        safe_incomplete_idx = summary.get("safe-incomplete", [])
        safe_idx = summary.get("safe", [])
        unsafe_pgd_idx = summary.get("unsafe-pgd", [])
        unsafe_bab_idx = summary.get("unsafe", [])

        for idx in safe_incomplete_idx:
            is_dpb_cert[idx] = True
        for idx in safe_idx:
            is_abcrown_cert[idx] = True
        for idx in unsafe_pgd_idx:
            is_adv_attacked[idx] = True
        for idx in unsafe_bab_idx:
            is_bab_rejected[idx] = True

    for index in range(len(x)):
        if not is_dpb_cert[index] and not is_abcrown_cert[index] and not is_adv_attacked[index] and not is_bab_rejected[index]:
            is_undecidable[index] = True

    for temp_created_path in [data_path, result_file_path]:
        if os.path.exists(temp_created_path):
            try:
                os.remove(temp_created_path)
            except OSError:
                pass

    return is_dpb_cert, is_abcrown_cert, is_adv_attacked, is_bab_rejected, is_undecidable


def run_autoattack(model, x, y, eps, device, args):
    assert AutoAttack is not None, "AutoAttack is not installed. Install it or run without --use-autoattack."
    attack_batch = args.attack_batch if args.attack_batch is not None else len(x)
    adversary = AutoAttack(model, norm="Linf", eps=eps, version=args.autoattack_version, device=device)
    with torch.enable_grad():
        x_adv = adversary.run_standard_evaluation(x, y, bs=attack_batch)
    y_adv = model(x_adv).argmax(dim=1)
    return y_adv != y


def transform_abs_into_torch(abs_net, torch_net):
    abs_state = abs_net.state_dict()
    torch_state = {}
    for key, value in abs_state.items():
        key = key.lstrip("layers.")
        if key == "0.sigma":
            key = "0.std"
        torch_state[key] = value

    torch_net.load_state_dict(torch_state)
    return torch_net
