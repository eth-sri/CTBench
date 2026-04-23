import sys
import os
import json
import glob

def aggregate(results_dir):
    pattern_partial = os.path.join(results_dir, "cert*.json")
    pattern_complete = os.path.join(results_dir, "complete_cert*.json")

    partial_files = sorted(f for f in glob.glob(pattern_partial) if "args" not in f and "complete_" not in os.path.basename(f))
    complete_files = sorted(glob.glob(pattern_complete))

    def extract_range(path):
        base = os.path.basename(path)
        base = base.replace("complete_cert", "").replace("cert", "").replace(".json", "")
        parts = base.split("_")
        return tuple(int(p) for p in parts if p)

    complete_ranges = {extract_range(f): f for f in complete_files}
    partial_ranges = {extract_range(f): f for f in partial_files}

    all_ranges = set(complete_ranges.keys()) | set(partial_ranges.keys())
    chosen_files = {}
    for r in all_ranges:
        if r in complete_ranges:
            chosen_files[r] = (complete_ranges[r], "complete")
        else:
            chosen_files[r] = (partial_ranges[r], "partial")

    if not chosen_files:
        print(f"No result files found in {results_dir}")
        return

    print(f"\nDiscovered {len(chosen_files)} result file(s). Aggregating...\n")
    print(f"{'':36} {'':10} {'':>6} {'':>7} {'--- Certified (Safe) ---':^27} {'':>2} {'----- Unsafe -----':^21} {'':>8} {'':>9}")
    print(f"{'File':<36} {'Status':<10} {'Total':>6} {'NatAcc':>7} {'IBP':>5} {'DPB':>5} {'a-CROWN':>8} {'BaB':>5} {'':>2} {'AA':>5} {'aPGD':>5} {'Reject':>7} {'Unknown':>8} {'CertRate':>9}")
    print("-" * 143)

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
    autoattack_reported = False

    for rng, (fpath, status) in sorted(chosen_files.items()):
        with open(fpath) as f:
            data = json.load(f)

        name = os.path.basename(fpath)

        # Get data from JSON
        t_tot = data.get('num_total', 0)
        t_nat = data.get('num_nat_accu', 0)
        t_ibp = data.get('num_cert_ibp', 0)
        t_heu_dpb = data.get('num_heuristic_dpb', data.get('num_cert_dpb', 0))
        t_alpha = data.get('num_cert_alpha_crown', 0)
        t_bab = data.get('num_cert_abcrown', 0)
        has_split_attacks = (
            'num_autoattack_attacked' in data
            or 'num_abcrown_pgd_attacked' in data
            or 'num_abcrown_pgd_unsafe' in data
        )
        t_autoattack = data.get('num_autoattack_attacked', 0)
        t_abcrown_pgd = data.get('num_abcrown_pgd_attacked', data.get('num_abcrown_pgd_unsafe', 0 if has_split_attacks else data.get('num_adv_attacked', 0)))
        t_rej = data.get('num_bab_rejected', 0)
        autoattack_reported = autoattack_reported or data.get('autoattack_adv_accuracy') is not None or 'num_autoattack_attacked' in data

        t_cert_sum = t_ibp + t_heu_dpb + t_alpha + t_bab
        t_unknown = max(0, t_nat - t_cert_sum - t_autoattack - t_abcrown_pgd - t_rej)

        print(f"  {name:<34} [{status:<8}] {t_tot:>6} "
              f"{t_nat:>7} {t_ibp:>5} {t_heu_dpb:>5} {t_alpha:>8} {t_bab:>5} "
              f"{'|':>3} {t_autoattack:>5} {t_abcrown_pgd:>5} {t_rej:>7} {t_unknown:>8} "
              f"{data.get('total_cert_rate',0.0):>8.1f}%")

        totals['num_total'] += t_tot
        totals['num_nat_accu'] += t_nat
        totals['num_cert_ibp'] += t_ibp
        totals['num_heuristic_dpb'] += t_heu_dpb
        totals['num_cert_alpha_crown'] += t_alpha
        totals['num_cert_abcrown'] += t_bab
        totals['num_autoattack_attacked'] += t_autoattack
        totals['num_abcrown_pgd_attacked'] += t_abcrown_pgd
        totals['num_bab_rejected'] += t_rej

    print("-" * 143)
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

    unknown_total = max(0, nat_acc - cert_total - autoattack_total - verifier_unsafe_total)
    autoattack_correct = nat_acc - autoattack_total
    autoattack_adv_acc = autoattack_correct / total * 100 if autoattack_reported else None

    print(f"\n{'='*55}")
    print(f"  FINAL AGGREGATE SUMMARY")
    print(f"{'='*55}")
    print(f"  Total images processed  : {total}")
    print(f"  Natural accuracy        : {nat_acc} / {total}  ({nat_acc/total*100:.2f}%)")
    print(f"{'-'*55}")
    n_width = 5
    p_width = 5
    print(f"  [1] Certified (Safe)    : {cert_total:>{n_width}} ({cert_total/total*100:>{p_width}.2f}%)")
    print(f"      - IBP               : {totals['num_cert_ibp']:>{n_width}} ({totals['num_cert_ibp']/total*100:>{p_width}.2f}%)")
    print(f"      - DeepPoly          : {totals['num_heuristic_dpb']:>{n_width}} ({totals['num_heuristic_dpb']/total*100:>{p_width}.2f}%)")
    print(f"      - abcrown(alpha)    : {totals['num_cert_alpha_crown']:>{n_width}} ({totals['num_cert_alpha_crown']/total*100:>{p_width}.2f}%)")
    print(f"      - abcrown(beta)     : {totals['num_cert_abcrown']:>{n_width}} ({totals['num_cert_abcrown']/total*100:>{p_width}.2f}%)")
    print()
    if autoattack_adv_acc is not None:
        print(f"  [2] AutoAttack adv acc  : {autoattack_correct:>{n_width}} / {total} ({autoattack_adv_acc:>{p_width}.2f}%)")
        print(f"      - AutoAttack found  : {autoattack_total:>{n_width}} ({autoattack_total/total*100:>{p_width}.2f}%)")
        print()
    print(f"  [3] Verifier unsafe     : {verifier_unsafe_total:>{n_width}} ({verifier_unsafe_total/total*100:>{p_width}.2f}%)")
    print(f"      - abCROWN PGD       : {abcrown_pgd_total:>{n_width}} ({abcrown_pgd_total/total*100:>{p_width}.2f}%)")
    print(f"      - BaB rejected      : {rej_total:>{n_width}} ({rej_total/total*100:>{p_width}.2f}%)")
    print()
    print(f"  [4] Unknown             : {unknown_total:>{n_width}} ({unknown_total/total*100:>{p_width}.2f}%)")
    print(f"{'─'*55}")

    # Calculate percentage relative to correctly classified images (Nat Acc)
    if nat_acc > 0:
        print(f"  Correctly classified breakdown:")
        print(f"      - Certified (Safe)  : {cert_total:>{n_width}} / {nat_acc} ({cert_total/nat_acc*100:>{p_width}.2f}%)")
        if autoattack_reported:
            print(f"      - AutoAttack found  : {autoattack_total:>{n_width}} / {nat_acc} ({autoattack_total/nat_acc*100:>{p_width}.2f}%)")
        print(f"      - Verifier unsafe   : {verifier_unsafe_total:>{n_width}} / {nat_acc} ({verifier_unsafe_total/nat_acc*100:>{p_width}.2f}%)")
        print(f"      - Unknown           : {unknown_total:>{n_width}} / {nat_acc} ({unknown_total/nat_acc*100:>{p_width}.2f}%)")

    print(f"{'='*55}\n")

if __name__ == "__main__":
    results_dir = sys.argv[1]
    aggregate(results_dir)
