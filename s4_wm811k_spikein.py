"""
S4 - Recall under known effects (spike-in, Section IV-C).

Slot numbers are shuffled inside every lot (no order information), then a parity,
start-of-lot or trend effect of 0.25, 0.5 and 1.0 within-lot SDs of y is added:
    parity : +delta on odd slots
    head   : +delta on slots 1-3
    trend  : +delta * (s - 1) / 24
Recall = share of injected lots with p < 0.05 for the matching statistic.
False-alarm rate = same share on untouched shuffled lots.
Fixed-threshold rule: |T| above the threshold that gives the SAME false-alarm
rate on the untouched shuffled lots; its recall is reported for comparison.

Usage:  python s4_wm811k_spikein.py --n-lots 2000
Output: out/spikein/recall_table.csv (the nine values for the supplementary table)
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from common import SCRIPT_DIR, find_file, load_lswmd, prepare_df, shuffle_slots, lots_from_df
from slot_audit import STATS, STAT_NAMES, lot_stats

EFFECT = {"par": lambda s: (s % 2 == 1).astype(float),
          "head": lambda s: (s <= 3).astype(float),
          "rho": lambda s: (s - 1) / 24.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=None)
    ap.add_argument("--n-lots", type=int, default=2000, help="lots per configuration")
    ap.add_argument("--sizes", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--min-wafers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "out", "spikein"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    df = shuffle_slots(prepare_df(load_lswmd(find_file(a.pkl, "LSWMD.pkl"))), rng)
    lots = [L for L in lots_from_df(df, a.min_wafers) if np.ptp(L[2]) > 0]
    print(f"{len(lots):,} non-constant shuffled lots")
    t0 = time.time()

    # untouched shuffled lots: false-alarm rate and fixed-threshold calibration
    pick = rng.choice(len(lots), min(a.n_lots, len(lots)), replace=False)
    base_T, base_p = [], []
    for i in pick:
        _, s, y = lots[i]
        obs, pv = lot_stats(s, y, a.perm, rng)
        base_T.append(np.abs(obs)); base_p.append(pv)
    base_T, base_p = np.array(base_T), np.array(base_p)
    fa = (base_p < 0.05).mean(0)
    thr = np.array([np.quantile(base_T[:, j], 1 - fa[j]) for j in range(3)])
    print("false-alarm rate (untouched shuffled):", dict(zip(STATS, fa.round(4).tolist())))

    rows = []
    for j, k in enumerate(STATS):
        for size in a.sizes:
            pick = rng.choice(len(lots), min(a.n_lots, len(lots)), replace=False)
            hit_perm = hit_fix = 0
            for i in pick:
                _, s, y = lots[i]
                y2 = y + size * y.std() * EFFECT[k](s)
                obs, pv = lot_stats(s, y2, a.perm, rng)
                hit_perm += pv[j] < 0.05
                hit_fix += abs(obs[j]) > thr[j]
            rows.append(dict(statistic=STAT_NAMES[k], effect_SD=size, lots=len(pick),
                             recall_permutation=round(float(hit_perm) / len(pick), 4),
                             recall_fixed_threshold=round(float(hit_fix) / len(pick), 4),
                             false_alarm_untouched=round(float(fa[j]), 4)))
            print(rows[-1], f"({time.time() - t0:.0f}s)")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "recall_table.csv"), index=False)
    json.dump(dict(false_alarm=dict(zip(STATS, fa.round(4).tolist())), rows=rows),
              open(os.path.join(a.out, "spikein.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
