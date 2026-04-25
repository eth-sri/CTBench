import sys
import os
import json
import glob


def extract_range(path):
    base = os.path.basename(path)
    base = base.replace("complete_cert", "").replace("cert", "").replace(".json", "")
    parts = base.split("_")
    return tuple(int(p) for p in parts if p)


def load_cert_args(results_dir, result_path):
    base = os.path.basename(result_path)
    if base.startswith("complete_cert"):
        args_base = base.replace("complete_cert", "cert_args", 1)
    else:
        args_base = base.replace("cert", "cert_args", 1)
    args_path = os.path.join(results_dir, args_base)
    if not os.path.isfile(args_path):
        return {}
    with open(args_path) as f:
        return json.load(f)


def choose_result_files(results_dir):
    pattern_partial = os.path.join(results_dir, "cert*.json")
    pattern_complete = os.path.join(results_dir, "complete_cert*.json")

    partial_files = sorted(f for f in glob.glob(pattern_partial) if "args" not in f and "complete_" not in os.path.basename(f))
    complete_files = sorted(glob.glob(pattern_complete))

    full_complete = os.path.join(results_dir, "complete_cert.json")
    using_full_complete = os.path.isfile(full_complete)
    ignored_shards = 0
    if using_full_complete:
        ignored_shards = len([f for f in complete_files if f != full_complete]) + len(partial_files)
        complete_files = [full_complete]
        partial_files = []

    complete_ranges = {extract_range(f): f for f in complete_files}
    partial_ranges = {extract_range(f): f for f in partial_files}

    all_ranges = set(complete_ranges.keys()) | set(partial_ranges.keys())
    chosen_files = {}
    for r in all_ranges:
        if r in complete_ranges:
            chosen_files[r] = (complete_ranges[r], "complete")
        else:
            chosen_files[r] = (partial_ranges[r], "partial")

    return chosen_files, using_full_complete, ignored_shards


def summarize_file(results_dir, fpath, status):
    with open(fpath) as f:
        data = json.load(f)
    cert_args = load_cert_args(results_dir, fpath)

    is_legacy_mnbab = 'num_cert_mnbab' in data or 'mnbab_config' in cert_args
    is_abcrown_result = (
        not is_legacy_mnbab
        and (
            'abcrown_config' in cert_args
            or 'num_cert_alpha_crown' in data
            or 'num_cert_abcrown' in data
            or 'num_bab_rejected' in data
        )
    )
    has_split_attacks = (
        'num_autoattack_attacked' in data
        or 'num_abcrown_pgd_attacked' in data
        or 'num_abcrown_pgd_unsafe' in data
    )
    use_autoattack = cert_args.get('use_autoattack', False)

    if has_split_attacks:
        num_autoattack_attacked = data.get('num_autoattack_attacked', 0)
        num_abcrown_pgd_attacked = data.get('num_abcrown_pgd_attacked', data.get('num_abcrown_pgd_unsafe', 0))
    elif is_abcrown_result and use_autoattack:
        num_autoattack_attacked = data.get('num_adv_attacked', 0)
        num_abcrown_pgd_attacked = 0
    else:
        num_autoattack_attacked = 0
        num_abcrown_pgd_attacked = data.get('num_adv_attacked', 0)

    num_cert_alpha_crown = data.get('num_cert_alpha_crown', 0)
    num_cert_abcrown = data.get('num_cert_abcrown', data.get('num_cert_mnbab', 0))
    num_bab_rejected = data.get('num_bab_rejected', 0)
    use_abcrown = (
        not cert_args.get('disable_abcrown', False)
        if cert_args and not is_legacy_mnbab
        else (num_cert_alpha_crown + data.get('num_cert_abcrown', 0) + num_bab_rejected) > 0
    )
    use_abcrown_pgd = use_abcrown and not cert_args.get('disable_abcrown_pgd', False) if cert_args else num_abcrown_pgd_attacked > 0

    num_total = data.get('num_total', 0)
    num_nat_accu = data.get('num_nat_accu', 0)
    num_cert_ibp = data.get('num_cert_ibp', 0)
    num_heuristic_dpb = data.get('num_heuristic_dpb', data.get('num_cert_dpb', 0))
    num_cert_sum = num_cert_ibp + num_heuristic_dpb + num_cert_alpha_crown + num_cert_abcrown
    num_unsafe_sum = num_autoattack_attacked + num_abcrown_pgd_attacked + num_bab_rejected
    num_unknown = max(0, num_nat_accu - num_cert_sum - num_unsafe_sum)

    return {
        "name": os.path.basename(fpath),
        "status": status,
        "num_total": num_total,
        "num_nat_accu": num_nat_accu,
        "num_cert_ibp": num_cert_ibp,
        "num_heuristic_dpb": num_heuristic_dpb,
        "num_cert_alpha_crown": num_cert_alpha_crown,
        "num_cert_abcrown": num_cert_abcrown,
        "num_autoattack_attacked": num_autoattack_attacked,
        "num_abcrown_pgd_attacked": num_abcrown_pgd_attacked,
        "num_bab_rejected": num_bab_rejected,
        "num_cert_sum": num_cert_sum,
        "num_unsafe_sum": num_unsafe_sum,
        "num_unknown": num_unknown,
        "total_cert_rate": data.get('total_cert_rate', 0.0),
        "enable_dpb": cert_args.get('enable_heuristic_dpb', False),
        "use_autoattack": use_autoattack,
        "use_abcrown": use_abcrown,
        "use_abcrown_pgd": use_abcrown_pgd,
        "has_split_attacks": has_split_attacks,
        "has_unsplit_abcrown_attack": is_abcrown_result and not has_split_attacks and num_abcrown_pgd_attacked > 0,
        "has_unsplit_abcrown_autoattack": is_abcrown_result and not has_split_attacks and use_autoattack and num_autoattack_attacked > 0,
        "is_legacy_mnbab": is_legacy_mnbab,
        "has_legacy_unsafe": is_legacy_mnbab and not has_split_attacks and num_abcrown_pgd_attacked > 0,
        "has_mnbab_count": data.get('num_cert_mnbab', 0) > 0,
    }


def update_totals(totals, stage_flags, shard):
    for key in totals:
        totals[key] += shard[key]

    stage_flags["autoattack"] = stage_flags["autoattack"] or (
        shard["use_autoattack"]
        and (shard["has_split_attacks"] or shard["has_unsplit_abcrown_autoattack"])
    )
    stage_flags["deep_poly"] = stage_flags["deep_poly"] or shard["enable_dpb"] or shard["num_heuristic_dpb"] > 0
    stage_flags["abcrown"] = stage_flags["abcrown"] or shard["use_abcrown"] or shard["num_cert_alpha_crown"] > 0
    stage_flags["abcrown_pgd"] = stage_flags["abcrown_pgd"] or shard["use_abcrown_pgd"] or shard["num_abcrown_pgd_attacked"] > 0
    stage_flags["mnbab"] = stage_flags["mnbab"] or shard["is_legacy_mnbab"] or shard["has_mnbab_count"]
    stage_flags["legacy_pgd"] = stage_flags["legacy_pgd"] or shard["has_legacy_unsafe"]
    stage_flags["unsplit_abcrown_attack"] = stage_flags["unsplit_abcrown_attack"] or shard["has_unsplit_abcrown_attack"]
    stage_flags["attack_accuracy"] = stage_flags["attack_accuracy"] or (
        (
            shard["has_split_attacks"]
            and (
                shard["use_autoattack"]
                or shard["use_abcrown_pgd"]
                or shard["num_autoattack_attacked"] > 0
                or shard["num_abcrown_pgd_attacked"] > 0
            )
        )
        or (
            shard["has_unsplit_abcrown_attack"]
            and (
                shard["use_abcrown_pgd"]
                or shard["num_abcrown_pgd_attacked"] > 0
            )
        )
        or shard["has_unsplit_abcrown_autoattack"]
    )


def print_shard_table(shards, using_full_complete, ignored_shards):
    print(f"\nDiscovered {len(shards)} result file(s). Aggregating...\n")
    if using_full_complete and ignored_shards > 0:
        print(f"Using complete_cert.json and ignoring {ignored_shards} shard/partial file(s) in the same directory.\n")

    header = f"{'File':<36} {'Status':<10} {'Total':>6} {'NatAcc':>7} {'Cert':>6} {'Unsafe':>7} {'Unknown':>8} {'CertRate':>9}"
    row_width = len(header)
    print(header)
    print("-" * row_width)

    for shard in shards:
        print(f"  {shard['name']:<34} [{shard['status']:<8}] {shard['num_total']:>6} "
              f"{shard['num_nat_accu']:>7} {shard['num_cert_sum']:>6} {shard['num_unsafe_sum']:>7} {shard['num_unknown']:>8} "
              f"{shard['total_cert_rate']:>8.1f}%")

    print("-" * row_width)


def print_final_summary(totals, stage_flags):
    total = totals["num_total"]
    if total == 0:
        print("No images processed yet.")
        return

    cert_total = totals['num_cert_ibp'] + totals['num_heuristic_dpb'] + totals['num_cert_alpha_crown'] + totals['num_cert_abcrown']
    nat_acc = totals["num_nat_accu"]
    autoattack_total = totals["num_autoattack_attacked"]
    abcrown_pgd_total = totals["num_abcrown_pgd_attacked"]
    rej_total = totals["num_bab_rejected"]
    verifier_unsafe_total = abcrown_pgd_total + rej_total
    nat_misclassified = max(0, total - nat_acc)

    unknown_total = max(0, nat_acc - cert_total - autoattack_total - verifier_unsafe_total)
    attack_total = autoattack_total + abcrown_pgd_total
    attack_adv_correct = nat_acc - attack_total
    attack_adv_acc = attack_adv_correct / total * 100 if stage_flags["attack_accuracy"] else None

    print(f"\n{'='*55}")
    print(f"  FINAL AGGREGATE SUMMARY")
    print(f"{'='*55}")
    print(f"  Total images processed  : {total}")
    print(f"  Natural accuracy        : {nat_acc} / {total}  ({nat_acc/total*100:.2f}%)")
    print(f"  Certified accuracy      : {cert_total} / {total}  ({cert_total/total*100:.2f}%)")
    if attack_adv_acc is not None:
        print(f"  Adversarial accuracy    : {attack_adv_correct} / {total}  ({attack_adv_acc:.2f}%)")
    print(f"{'-'*55}")
    n_width = 5
    p_width = 5
    print(f"  Pipeline breakdown:")
    print(f"      - Natural misclass. : {nat_misclassified:>{n_width}} ({nat_misclassified/total*100:>{p_width}.2f}%)")
    print(f"      - IBP certified     : {totals['num_cert_ibp']:>{n_width}} ({totals['num_cert_ibp']/total*100:>{p_width}.2f}%)")
    if stage_flags["deep_poly"]:
        print(f"      - DeepPoly certified: {totals['num_heuristic_dpb']:>{n_width}} ({totals['num_heuristic_dpb']/total*100:>{p_width}.2f}%)")
    attack_total = autoattack_total + abcrown_pgd_total
    if stage_flags["autoattack"]:
        print(f"      - AutoAttack unsafe : {autoattack_total:>{n_width}} ({autoattack_total/total*100:>{p_width}.2f}%)")
    if stage_flags["abcrown"]:
        if stage_flags["abcrown_pgd"]:
            if stage_flags["unsplit_abcrown_attack"]:
                print(f"      - Attack unsafe     : {abcrown_pgd_total:>{n_width}} ({abcrown_pgd_total/total*100:>{p_width}.2f}%)")
            else:
                print(f"      - abCROWN PGD unsafe: {abcrown_pgd_total:>{n_width}} ({abcrown_pgd_total/total*100:>{p_width}.2f}%)")
        print(f"      - alpha-CROWN cert. : {totals['num_cert_alpha_crown']:>{n_width}} ({totals['num_cert_alpha_crown']/total*100:>{p_width}.2f}%)")
        print(f"      - beta-CROWN cert.  : {totals['num_cert_abcrown']:>{n_width}} ({totals['num_cert_abcrown']/total*100:>{p_width}.2f}%)")
        print(f"      - BaB unsafe/reject : {rej_total:>{n_width}} ({rej_total/total*100:>{p_width}.2f}%)")
    elif stage_flags["mnbab"]:
        print(f"      - MN-BaB certified  : {totals['num_cert_abcrown']:>{n_width}} ({totals['num_cert_abcrown']/total*100:>{p_width}.2f}%)")
        if stage_flags["abcrown_pgd"]:
            print(f"      - Unsafe            : {abcrown_pgd_total:>{n_width}} ({abcrown_pgd_total/total*100:>{p_width}.2f}%)")
    elif stage_flags["abcrown_pgd"]:
        print(f"      - Unsafe            : {abcrown_pgd_total:>{n_width}} ({abcrown_pgd_total/total*100:>{p_width}.2f}%)")
    print(f"      - Unknown           : {unknown_total:>{n_width}} ({unknown_total/total*100:>{p_width}.2f}%)")
    print(f"{'─'*55}")

    # Calculate percentage relative to correctly classified images (Nat Acc)
    if nat_acc > 0:
        print(f"  Correctly classified breakdown:")
        print(f"      - Certified (Safe)  : {cert_total:>{n_width}} / {nat_acc} ({cert_total/nat_acc*100:>{p_width}.2f}%)")
        if stage_flags["autoattack"]:
            print(f"      - AutoAttack found  : {autoattack_total:>{n_width}} / {nat_acc} ({autoattack_total/nat_acc*100:>{p_width}.2f}%)")
        if stage_flags["abcrown_pgd"] and not stage_flags["legacy_pgd"]:
            pgd_label = "Attack found" if stage_flags["unsplit_abcrown_attack"] else "abCROWN PGD found"
            print(f"      - {pgd_label:<18}: {abcrown_pgd_total:>{n_width}} / {nat_acc} ({abcrown_pgd_total/nat_acc*100:>{p_width}.2f}%)")
        if stage_flags["abcrown"]:
            print(f"      - BaB unsafe/reject : {rej_total:>{n_width}} / {nat_acc} ({rej_total/nat_acc*100:>{p_width}.2f}%)")
        elif stage_flags["mnbab"] or stage_flags["abcrown_pgd"]:
            print(f"      - Unsafe            : {verifier_unsafe_total:>{n_width}} / {nat_acc} ({verifier_unsafe_total/nat_acc*100:>{p_width}.2f}%)")
        print(f"      - Unknown           : {unknown_total:>{n_width}} / {nat_acc} ({unknown_total/nat_acc*100:>{p_width}.2f}%)")

    print(f"{'='*55}\n")


def aggregate(results_dir):
    chosen_files, using_full_complete, ignored_shards = choose_result_files(results_dir)
    if not chosen_files:
        print(f"No result files found in {results_dir}")
        return

    totals = {
        "num_total": 0,
        "num_nat_accu": 0,
        "num_cert_ibp": 0,
        "num_heuristic_dpb": 0,
        "num_cert_alpha_crown": 0,
        "num_cert_abcrown": 0,
        "num_autoattack_attacked": 0,
        "num_abcrown_pgd_attacked": 0,
        "num_bab_rejected": 0,
    }
    stage_flags = {
        "deep_poly": False,
        "autoattack": False,
        "abcrown": False,
        "abcrown_pgd": False,
        "mnbab": False,
        "legacy_pgd": False,
        "unsplit_abcrown_attack": False,
        "attack_accuracy": False,
    }

    shards = []
    for rng, (fpath, status) in sorted(chosen_files.items()):
        shard = summarize_file(results_dir, fpath, status)
        shards.append(shard)
        update_totals(totals, stage_flags, shard)

    print_shard_table(shards, using_full_complete, ignored_shards)
    print_final_summary(totals, stage_flags)

if __name__ == "__main__":
    results_dir = sys.argv[1]
    aggregate(results_dir)
