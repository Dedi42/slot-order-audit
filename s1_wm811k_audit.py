"""
S1 - Slot-order audit of WM-811K (Section IV-B) and shuffled-slot control (IV-C).

For every lot with >= --min-wafers maps: y = fail fraction per wafer, the three
slot statistics with within-lot permutation p-values (B = --perm), plus analytic
p-values (Welch t-test, asymptotic Spearman) for the comparison in Section IV-C.

Usage:
    python s1_wm811k_audit.py --out out/audit_real
    python s1_wm811k_audit.py --out out/audit_control --shuffle-control
Output: <out>/lot_slot_stats.csv, <out>/summary.json
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from common import SCRIPT_DIR, find_file, load_lswmd, prepare_df, shuffle_slots, lots_from_df
from slot_audit import STATS, STAT_NAMES, lot_stats, analytic_p, bh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=None, help="path to LSWMD.pkl")
    ap.add_argument("--perm", type=int, default=2000, help="permutations per lot (B)")
    ap.add_argument("--min-wafers", type=int, default=20)
    ap.add_argument("--max-lots", type=int, default=0, help="0 = all lots")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "out", "audit_real"))
    ap.add_argument("--shuffle-control", action="store_true",
                    help="negative control: shuffle slot numbers inside every lot first")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    t0 = time.time()
    df = prepare_df(load_lswmd(find_file(a.pkl, "LSWMD.pkl")))
    if a.shuffle_control:
        df = shuffle_slots(df, rng)
        print("NEGATIVE CONTROL: slot numbers shuffled inside each lot")
    lots = lots_from_df(df, a.min_wafers)
    if a.max_lots:
        lots = lots[: a.max_lots]
    print(f"{len(lots):,} lots with >= {a.min_wafers} wafers  ({time.time() - t0:.0f}s)")

    rows = []
    for i, (lot, s, y) in enumerate(lots):
        row = dict(lot=lot, n=len(y), mean_y=float(y.mean()), constant=bool(np.ptp(y) == 0))
        if row["constant"]:
            row.update(d_par=0.0, d_head=0.0, rho=0.0, p_par=1.0, p_head=1.0, p_rho=1.0,
                       pa_par=1.0, pa_head=1.0, pa_rho=1.0)
        else:
            (dp, dh, r), (pp, ph, pr) = lot_stats(s, y, a.perm, rng)
            ap_ = analytic_p(s, y)
            row.update(d_par=dp, d_head=dh, rho=r, p_par=pp, p_head=ph, p_rho=pr,
                       pa_par=ap_[0], pa_head=ap_[1], pa_rho=ap_[2])
        rows.append(row)
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1:,}/{len(lots):,} lots  ({time.time() - t0:.0f}s)")
    res = pd.DataFrame(rows)
    for k in STATS:
        res[f"q_{k}"] = bh(res[f"p_{k}"].to_numpy())
    res.to_csv(os.path.join(a.out, "lot_slot_stats.csv"), index=False)

    summ = dict(negative_control=a.shuffle_control, lots=int(len(res)), permutations=a.perm,
                seed=a.seed, runtime_s=round(time.time() - t0, 1))
    for k in STATS:
        summ[STAT_NAMES[k]] = dict(share_p05=round(float((res[f"p_{k}"] < 0.05).mean()), 4),
                                   n_BH_q05=int((res[f"q_{k}"] < 0.05).sum()),
                                   share_analytic_p05=round(float((res[f"pa_{k}"] < 0.05).mean()), 4))
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
