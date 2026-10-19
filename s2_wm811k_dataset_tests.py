"""
S2 - Dataset-level inference on WM-811K (Section IV-B/C, Table III, Fig. 3).

Reads the real and the shuffled-control audit and reports, per statistic:
  share p<0.05, exact binomial test vs 5 %, Storey pi1, excess-based lower bound,
  lots at the minimum attainable p = 1/(B+1), lots surviving BH at q<0.05,
  analytic (no permutation) share, fixed-threshold (2 SD, no permutation) share,
  and the strata of Fig. 3(b) (die-size quintile, lot size, mean fail rate).

Usage:  python s2_wm811k_dataset_tests.py
Output: out/dataset_tests/table3.csv, strata.csv, dataset_tests.json, fig3.png
"""
import argparse, json, os
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import SCRIPT_DIR, find_file, load_lswmd, prepare_df
from slot_audit import STATS, STAT_NAMES, storey_pi1


def stat_rows(res, B):
    pmin = 1 / (B + 1)
    rows = []
    for k in STATS:
        p = res[f"p_{k}"].to_numpy()
        n_sig = int((p < 0.05).sum())
        T = res[{"par": "d_par", "head": "d_head", "rho": "rho"}[k]].to_numpy()
        thr = 2 * T.std()
        rows.append(dict(statistic=STAT_NAMES[k], lots=int(p.size), n_p05=n_sig,
                         share_p05=round(n_sig / p.size, 4),
                         binom_p=float(binomtest(n_sig, p.size, 0.05, alternative="greater").pvalue),
                         storey_pi1=round(storey_pi1(p), 4),
                         excess_lower_bound=round((n_sig / p.size - 0.05) / 0.95, 4),
                         n_at_min_p=int((p <= pmin + 1e-12).sum()),
                         n_BH_q05=int((res[f"q_{k}"] < 0.05).sum()),
                         share_analytic_p05=round(float((res[f"pa_{k}"] < 0.05).mean()), 4),
                         share_fixed_2sd=round(float((np.abs(T) > thr).mean()), 4)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=os.path.join("out", "audit_real", "lot_slot_stats.csv"))
    ap.add_argument("--control", default=os.path.join("out", "audit_control", "lot_slot_stats.csv"))
    ap.add_argument("--pkl", default=None)
    ap.add_argument("--perm", type=int, default=2000, help="B used in s1")
    ap.add_argument("--exclude-constant", action="store_true",
                    help="drop lots with a constant fail fraction before computing shares")
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "out", "dataset_tests"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    real = pd.read_csv(find_file(a.real, "lot_slot_stats.csv")); real["lot"] = real["lot"].astype(str)
    ctrl = pd.read_csv(find_file(a.control, "lot_slot_stats.csv")); ctrl["lot"] = ctrl["lot"].astype(str)
    if a.exclude_constant:
        real, ctrl = real[~real.constant], ctrl[~ctrl.constant]

    t3 = pd.DataFrame([dict(data="real", **r) for r in stat_rows(real, a.perm)] +
                      [dict(data="shuffled", **r) for r in stat_rows(ctrl, a.perm)])
    t3.to_csv(os.path.join(a.out, "table3.csv"), index=False)
    anyrow = {}
    for name, d in (("real", real), ("shuffled", ctrl)):
        fixed = np.zeros(len(d), bool)
        for col in ("d_par", "d_head", "rho"):
            T = d[col].to_numpy(); fixed |= np.abs(T) > 2 * T.std()
        anyrow[name] = dict(share_fixed_2sd_any=round(float(fixed.mean()), 4),
                            share_perm_p05_any=round(float((d[["p_par", "p_head", "p_rho"]] < 0.05).any(axis=1).mean()), 4),
                            share_analytic_p05_any=round(float((d[["pa_par", "pa_head", "pa_rho"]] < 0.05).any(axis=1).mean()), 4))
    print("== ANY OF THE THREE STATISTICS ==", json.dumps(anyrow, indent=2))
    pd.set_option("display.width", 220)
    print("== TABLE III and Section IV-B/C ==")
    print(t3.to_string(index=False))

    # strata (Fig. 3(b))
    df = prepare_df(load_lswmd(find_file(a.pkl, "LSWMD.pkl")))
    meta = df.groupby("lotName").agg(dieSize=("dieSize", "first"))
    r = real.merge(meta, left_on="lot", right_index=True, how="left")
    groups = {"dieSize_quintile": pd.qcut(r["dieSize"], 5, duplicates="drop"),
              "lot_size": pd.cut(r["n"], [19, 24, 25], labels=["20-24", "25"]),
              "mean_fail": pd.cut(r["mean_y"], [-0.01, 0.02, 0.05, 0.1, 1.0],
                                  labels=["<2%", "2-5%", "5-10%", ">10%"])}
    srows = []
    for g, cat in groups.items():
        for lev, d in r.groupby(cat, observed=True):
            row = dict(grouping=g, level=str(lev), lots=len(d))
            for k in STATS:
                n = int((d[f"p_{k}"] < 0.05).sum())
                row[f"share_{k}"] = round(n / len(d), 4)
                row[f"binom_p_{k}"] = float(binomtest(n, len(d), 0.05, alternative="greater").pvalue)
            srows.append(row)
    strata = pd.DataFrame(srows)
    strata.to_csv(os.path.join(a.out, "strata.csv"), index=False)
    print("\n== STRATA (Fig. 3(b)) ==")
    print(strata[["grouping", "level", "lots", "share_par", "share_head", "share_rho"]].to_string(index=False))
    json.dump(dict(table3=t3.to_dict("records"), any_of_three=anyrow, strata=strata.to_dict("records")),
              open(os.path.join(a.out, "dataset_tests.json"), "w"), indent=2)

    # Fig. 3
    plt.rcParams.update({"font.family": "serif", "font.weight": "bold", "font.size": 9,
                         "axes.labelweight": "bold", "axes.linewidth": 1.1})
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.3), gridspec_kw=dict(width_ratios=[1, 1.6]))
    x = np.arange(3); w = 0.38
    rr, cc = t3[t3.data == "real"], t3[t3.data == "shuffled"]
    ax[0].bar(x - w / 2, 100 * rr.share_p05, w, color="#B22222", label="Real")
    ax[0].bar(x + w / 2, 100 * cc.share_p05, w, color="#7F7F7F", label="Shuffled")
    ax[0].axhline(5, color="k", ls="--", lw=1.1)
    ax[0].set_xticks(x); ax[0].set_xticklabels(["Odd/even", "Start", "Trend"])
    ax[0].set_ylabel("Lots with p < 0.05 (%)"); ax[0].legend(fontsize=8, frameon=False)
    ax[0].set_title("(a)", fontweight="bold")
    q = strata[strata.grouping == "dieSize_quintile"]
    xq = np.arange(len(q)); w = 0.26
    for j, (k, c) in enumerate(zip(STATS, ["#1F4E9E", "#2E8B2E", "#B22222"])):
        ax[1].bar(xq + (j - 1) * w, 100 * q[f"share_{k}"], w, color=c, label=STAT_NAMES[k])
    ax[1].axhline(5, color="k", ls="--", lw=1.1)
    ax[1].set_xticks(xq); ax[1].set_xticklabels([f"Q{i + 1}" for i in range(len(q))])
    ax[1].set_xlabel("Die-count quintile"); ax[1].legend(fontsize=8, frameon=False, ncol=3)
    ax[1].set_title("(b)", fontweight="bold")
    for axx in ax:
        for s in ("top", "right"):
            axx.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig3.png"), dpi=600); plt.close(fig)
    print(f"\nSaved to {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
