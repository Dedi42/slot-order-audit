"""
S9 - AUROC with stratified bootstrap 95 % CIs (Section V-D, Table VI, Fig. 6).

Input CSV: one row per lot with a 0/1 column (1 = Bad) and one or more score columns
(e.g. out-of-fold scores of the untrained chamber-deviation score, L1 logistic
regression and random forest).

Usage:  python s9_auroc_bootstrap.py --csv scores.csv --label bad --scores baseline l1_logreg rf
Output: printed table and out/auroc/auroc_ci.csv
"""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from common import SCRIPT_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--label", default="bad")
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "out", "auroc"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    d = pd.read_csv(a.csv)
    y = d[a.label].to_numpy(int)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rows = []
    for s in a.scores:
        x = d[s].to_numpy(float)
        bs = []
        for _ in range(a.boot):                       # resample Bad and Good lots separately
            i = np.r_[rng.choice(pos, pos.size), rng.choice(neg, neg.size)]
            bs.append(roc_auc_score(y[i], x[i]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append(dict(score=s, n_bad=int(pos.size), n_good=int(neg.size),
                         AUROC=round(float(roc_auc_score(y, x)), 3), ci_low=round(float(lo), 3), ci_high=round(float(hi), 3)))
    r = pd.DataFrame(rows); r.to_csv(os.path.join(a.out, "auroc_ci.csv"), index=False)
    print(r.to_string(index=False))


if __name__ == "__main__":
    main()
