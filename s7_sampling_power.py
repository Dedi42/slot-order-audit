"""
S7 - Sampling-plan design by simulation (Section III-A Step 6, Section VI, Table VII, Fig. 7).

Lots are simulated from model (5) with only the parts that matter for
within-lot contrasts: lot effect u_l ~ N(0, s_u^2) (cancelled by lot fixed
effects) and residual e ~ N(0, s_e^2), s_e = sigma_total * sqrt(1 - ICC).
Odd slots go to the first twin with probability --p-odd-first (0.5 = random).

Effects (delta, in process units):
    first : +delta on slot 1                  regressors: chamber, 1[s=1]
    twin  : +delta on the second chamber      regressors: chamber
    drift : +delta * (s - 1) / 24             regressors: chamber, (s-1)/24 [, 1[s=1] if joint]
Estimation: least squares with lot fixed effects (within-lot demeaning);
power = share of replicates with p < 0.05; MDE = delta at 80 % power (bisection).
"n.i." = not identifiable (regressor has no within-lot variation or is collinear).

Usage:  python s7_sampling_power.py            (defaults reproduce the Table VII design)
Output: out/sampling/table7.csv, lots_for_80pct.csv, fig7.png
"""
import argparse, json, os
import numpy as np
import pandas as pd
from scipy.stats import t as tdist, norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import SCRIPT_DIR

PLANS = {3: [1, 14, 24], 6: [1, 5, 10, 14, 19, 24], 13: list(range(1, 26, 2)), 25: list(range(1, 26))}


def power(effect, delta, slots, N, s_e, s_u, p_odd_first, R, seed, joint=False):
    rng = np.random.default_rng(seed)
    s = np.asarray(slots); m = s.size
    hits, ni = 0, False
    for _ in range(R):
        odd_first = rng.random(N) < p_odd_first                       # odd slots -> first twin
        c = np.where(odd_first[:, None], (s % 2 == 0)[None, :], (s % 2 == 1)[None, :]).astype(float)
        y = rng.normal(0, s_u, (N, 1)) + rng.normal(0, s_e, (N, m))
        first = np.broadcast_to((s == 1).astype(float), (N, m))
        ramp = np.broadcast_to((s - 1) / 24.0, (N, m))
        if effect == "first":
            y = y + delta * first; X = [c, first]
        elif effect == "twin":
            y = y + delta * c; X = [c]
        else:
            y = y + delta * ramp; X = [c, ramp] + ([first] if joint else [])
        Xd = np.stack([x - x.mean(1, keepdims=True) for x in X], -1).reshape(N * m, -1)
        yd = (y - y.mean(1, keepdims=True)).ravel()
        keep = np.abs(Xd).sum(0) > 1e-9
        target = 1 if effect != "twin" else 0
        if not keep[target] or np.linalg.matrix_rank(Xd[:, keep]) < keep.sum():
            ni = True; break
        Xd = Xd[:, keep]; ti = int(keep[:target + 1].sum()) - 1
        beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
        dfree = N * m - N - Xd.shape[1]
        sigma2 = ((yd - Xd @ beta) ** 2).sum() / dfree
        cov = sigma2 * np.linalg.inv(Xd.T @ Xd)
        tval = beta[ti] / np.sqrt(cov[ti, ti])
        hits += 2 * tdist.sf(abs(tval), dfree) < 0.05
    return None if ni else hits / R


def mde(effect, slots, N, s_e, s_u, p_odd_first, R, seed, joint=False, hi=60.0, tol=0.05):
    if power(effect, hi, slots, N, s_e, s_u, p_odd_first, 20, seed, joint) is None:
        return None
    lo = 0.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if power(effect, mid, slots, N, s_e, s_u, p_odd_first, R, seed, joint) >= 0.8:
            hi = mid
        else:
            lo = mid
    return round((lo + hi) / 2, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-total", type=float, default=20.9, help="residual SD (process units)")
    ap.add_argument("--icc", type=float, nargs="+", default=[0.5, 0.3])
    ap.add_argument("--N", type=int, default=100, help="lots for the MDE table")
    ap.add_argument("--R", type=int, default=400, help="replicates per bisection step")
    ap.add_argument("--effect-size", type=float, default=6.0, help="first-wafer effect for Fig. 7")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "out", "sampling"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    def add(label, icc, effect, m, p_odd_first=0.5, joint=False):
        s_e = a.sigma_total * np.sqrt(1 - icc); s_u = a.sigma_total * np.sqrt(icc)
        v = mde(effect, PLANS[m], a.N, s_e, s_u, p_odd_first, a.R, a.seed, joint)
        rows.append(dict(row=label, ICC=icc, sigma_e=round(float(s_e), 2), effect=effect, m=m,
                         p_odd_first=p_odd_first, joint_first_wafer=joint, MDE=v if v is not None else "n.i."))
        print(rows[-1])

    for icc in a.icc:
        for eff in ("first", "twin", "drift"):
            if icc != a.icc[0] and eff != "first":
                continue
            for m in PLANS:
                add(f"{eff}, ICC={icc}", icc, eff, m)
    icc0 = a.icc[0]
    for m in (3, 6):
        add("drift, joint with first wafer", icc0, "drift", m, joint=True)
        add("first, odd slots always same twin", icc0, "first", m, p_odd_first=1.0)
        add("first, 10% of lots swapped", icc0, "first", m, p_odd_first=0.9)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(a.out, "table7.csv"), index=False)

    # lots for 80 % power for a first-wafer effect of --effect-size (normal approximation, SE ~ 1/sqrt(N))
    base = tab[(tab.row == f"first, ICC={icc0}")]
    lots = []
    for _, r in base.iterrows():
        n80 = a.N * (float(r.MDE) / a.effect_size) ** 2
        lots.append(dict(m=int(r.m), MDE=float(r.MDE), N80_exact=int(np.ceil(n80)),
                         N80_rounded5=int(5 * np.ceil(np.ceil(n80) / 5))))
    lots = pd.DataFrame(lots)
    lots.to_csv(os.path.join(a.out, "lots_for_80pct.csv"), index=False)
    print(lots.to_string(index=False))

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                         "font.weight": "bold", "axes.labelweight": "bold", "font.size": 9, "axes.linewidth": 1.2})
    k = norm.ppf(0.975) + norm.ppf(0.8)
    N = np.linspace(10, 200, 400)
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    styles = [("#C00000", "-"), ("#1F4E9E", "--"), ("#2E8B2E", "-."), ("#B8860B", ":")]
    for (c, ls), (_, r) in zip(styles, lots.iterrows()):
        ax.plot(N, norm.cdf(k * np.sqrt(N / (a.N * (r.MDE / a.effect_size) ** 2)) - norm.ppf(0.975)),
                color=c, ls=ls, lw=2.0, label=f"m = {r.m}: {r.N80_exact} → {r.N80_rounded5}")
        ax.plot([r.N80_exact] * 2, [0.1, 0.8], color=c, ls=":", lw=1.1)
    ax.axhline(0.8, color="k", ls="--", lw=1.2)
    ax.set_xlim(5, 205); ax.set_ylim(0.1, 1.03)
    ax.set_xticks([25, 50, 75, 100, 125, 150, 175, 200]); ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Number of lots N", fontsize=9.5, labelpad=1); ax.set_ylabel("Power", fontsize=9.5, labelpad=2)
    ax.tick_params(labelsize=8.5, width=1.1, length=3, pad=1.5)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
    leg = ax.legend(title="Lots for 80% power", loc="lower right", fontsize=7.5, title_fontsize=7.5,
                    framealpha=1, edgecolor="0.6", handlelength=2.2, borderpad=0.4, labelspacing=0.25)
    leg.get_title().set_fontweight("bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.2)
    fig.savefig(os.path.join(a.out, "fig7.png"), dpi=600); plt.close(fig)
    json.dump(dict(args=vars(a), table7=rows, lots=lots.to_dict("records")),
              open(os.path.join(a.out, "sampling.json"), "w"), indent=2)
    print(f"Saved to {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
