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
    # Add Unknown(Timeout) column to header and adjust spacing
    print(f"{'File':<36} {'Status':<10} {'Total':>6} {'NatAcc':>7} {'IBP':>5} {'DPB':>5} {'a-CROWN':>8} {'BaB':>5} {'Attacked':>9} {'Unknown':>8} {'CertRate':>9}")
    print("-" * 120)
    
    totals = {
        "num_total": 0,
        "num_nat_accu": 0,
        "num_cert_ibp": 0,
        "num_heuristic_dpb": 0,
        "num_cert_alpha_crown": 0,
        "num_cert_abcrown": 0,
        "num_adv_attacked": 0,
    }
    
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
        t_adv = data.get('num_adv_attacked', 0)
        
        # Calculate Unknown(Timeout) per file: (Correct count) - (Safe proof count) - (Attacked count)
        t_cert_sum = t_ibp + t_heu_dpb + t_alpha + t_bab
        t_unknown = max(0, t_nat - t_cert_sum - t_adv)
        
        print(f"  {name:<34} [{status:<8}] {t_tot:>6} "
              f"{t_nat:>7} {t_ibp:>5} {t_heu_dpb:>5} {t_alpha:>8} {t_bab:>5} "
              f"{t_adv:>9} {t_unknown:>8} "
              f"{data.get('total_cert_rate',0.0):>8.1f}%")
        
        totals['num_total'] += t_tot
        totals['num_nat_accu'] += t_nat
        totals['num_cert_ibp'] += t_ibp
        totals['num_heuristic_dpb'] += t_heu_dpb
        totals['num_cert_alpha_crown'] += t_alpha
        totals['num_cert_abcrown'] += t_bab
        totals['num_adv_attacked'] += t_adv
    
    print("-" * 110)
    total = totals["num_total"]
    if total == 0:
        print("No images processed yet.")
        return
    
    cert_total = totals['num_cert_ibp'] + totals['num_heuristic_dpb'] + totals['num_cert_alpha_crown'] + totals['num_cert_abcrown']
    nat_acc = totals["num_nat_accu"]
    adv_total = totals["num_adv_attacked"]
    
    # Calculate total Unknown(Timeout)
    unknown_total = max(0, nat_acc - cert_total - adv_total)
    
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
    print(f"  [2] Attacked (Unsafe)   : {adv_total:>{n_width}} ({adv_total/total*100:>{p_width}.2f}%)")
    print(f"  [3] Unknown (Timeout)   : {unknown_total:>{n_width}} ({unknown_total/total*100:>{p_width}.1f}%)")
    print(f"{'─'*55}")
    
    # Calculate percentage relative to correctly classified images (Nat Acc)
    if nat_acc > 0:
        print(f"  * Out of {nat_acc} correctly classified images:")
        print(f"    Safe: {cert_total/nat_acc*100:.1f}% | Unsafe: {adv_total/nat_acc*100:.1f}% | Timeout: {unknown_total/nat_acc*100:.1f}%")
        
    print(f"{'='*55}\n")

if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "./results/cifar10/2.255/IBP/"  
    aggregate(results_dir)