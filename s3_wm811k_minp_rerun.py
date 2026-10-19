"""
S3 - Re-run lots at the minimum attainable p with more permutations (Section IV-B).

With B = 2,000 the smallest p is 1/(B+1); BH over ~26,700 lots cannot declare a
lot unless many lots reach it. This script takes every lot at that minimum for
any statistic, re-runs it with B = --perm (default 10^5), and reports how many
lots survive BH at q < 0.05 (column "After BH" of Table III).

Usage:  python s3_wm811k_minp_rerun.py
Output: out/minp_rerun/lot_slot_stats_rerun.csv, minp_rerun.json
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from common import SCRIPT_DIR, find_file, load_lswmd, prepare_df, lots_from_df
from slot_audit import STATS, STAT_NAMES, lot_stats, bh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=os.path.join("out", "audit_real", "lot_slot_stats.csv"))
    ap.add_argument("--pkl", default=None)
    ap.add_argument("--perm-old", type=int, default=2000)
    ap.add_argument("--perm", type=int, default=100_000)
    ap.add_argument("--chunk", type=int, default=20_000, help="permutations per batch (memory)")
    ap.add_argument("--min-wafers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "out", "minp_rerun"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    res = pd.read_csv(find_file(a.real, "lot_slot_stats.csv")); res["lot"] = res["lot"].astype(str)
    pmin = 1 / (a.perm_old + 1)
    at_min = {k: set(res.loc[res[f"p_{k}"] <= pmin + 1e-12, "lot"]) for k in STATS}
    todo = set().union(*at_min.values())
    print({STAT_NAMES[k]: len(v) for k, v in at_min.items()}, f"-> {len(todo)} lots to re-run with B = {a.perm:,}")

    df = prepare_df(load_lswmd(find_file(a.pkl, "LSWMD.pkl")))
    lots = {lot: (s, y) for lot, s, y in lots_from_df(df[df.lotName.isin(todo)], a.min_wafers)}
    t0 = time.time()
    for i, lot in enumerate(sorted(todo)):
        s, y = lots[lot]
        hits, done = np.zeros(3), 0
        while done < a.perm:
            b = min(a.chunk, a.perm - done)
            _, pv = lot_stats(s, y, b, rng)
            hits += np.array(pv) * (b + 1) - 1          # number of |T_b| >= |T| in this batch
            done += b
        p_new = (1 + hits) / (a.perm + 1)
        for j, k in enumerate(STATS):
            if lot in at_min[k]:
                res.loc[res.lot == lot, f"p_{k}"] = p_new[j]
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(todo)}  ({time.time() - t0:.0f}s)")
    summ = {}
    for k in STATS:
        res[f"q_{k}"] = bh(res[f"p_{k}"].to_numpy())
        summ[STAT_NAMES[k]] = dict(n_at_min_p_B_old=len(at_min[k]),
                                   n_BH_q05_after_rerun=int((res[f"q_{k}"] < 0.05).sum()))
    res.to_csv(os.path.join(a.out, "lot_slot_stats_rerun.csv"), index=False)
    json.dump(dict(perm_old=a.perm_old, perm_new=a.perm, **summ),
              open(os.path.join(a.out, "minp_rerun.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
