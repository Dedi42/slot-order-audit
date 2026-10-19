"""
S8 - Production analysis with a chamber log (Section V-A to V-C, Tables IV-V).

The production data of the paper cannot be shared; this script runs the same
analysis on YOUR OWN per-wafer table. Required CSV columns:
    lot, slot, chamber, tool, time, y
Optional columns:
    pair      twin-pair id of the chamber (e.g. "T2-C"); enables adjusted mismatch per pair
    pmcount   wafers since the last preventive maintenance; enables post-PM analysis

Model (5): y = mu + a_chamber + b_slot + f(time) + u_lot + e  (REML, statsmodels MixedLM;
the tool level is absorbed by the chamber because chambers are nested in tools).

Reports: parity share, Kruskal-Wallis chamber test, raw and adjusted twin mismatch,
b_slot1 - b_slot14 and b_slot24 - b_slot14 with parametric-bootstrap 95 % CIs,
sigma_u / sigma_e, within-chamber slot-1 test (Mann-Whitney, Hodges-Lehmann),
strong deviation of slot 1 (|z| > --z, Fisher), Levene dispersion ratio,
pooled start-of-lot test on residuals of model (5) without b_slot,
Theil-Sen drift per chamber (units per month), lot-wide excursions and post-PM share.

Usage:  python s8_fab_slot_chamber.py --csv my_fab.csv --boot 200
Output: out/fab/fab_results.json, chamber_drift.csv, pair_mismatch.csv, excursions.csv
"""
import argparse, json, os, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import kruskal, mannwhitneyu, levene, fisher_exact, theilslopes, spearmanr
from common import SCRIPT_DIR
from slot_audit import pooled_test

warnings.filterwarnings("ignore")


def fit(df, with_slot=True):
    f = "y ~ C(chamber) + bs(t_month, df=4)"
    if with_slot:
        f += " + C(slotpos, Treatment(reference='14'))"
    return smf.mixedlm(f, df, groups=df["lot"]).fit(reml=True)


def contrast(res, names_plus, names_minus=()):
    v = pd.Series(0.0, index=res.fe_params.index)
    for n in names_plus:
        if n in v.index: v[n] += 1
    for n in names_minus:
        if n in v.index: v[n] -= 1
    return v


def slot_name(s):
    return f"C(slotpos, Treatment(reference='14'))[T.{s}]"


def chamber_name(c, ref):
    return None if c == ref else f"C(chamber)[T.{c}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--boot", type=int, default=200, help="parametric-bootstrap refits for CIs (0 = Wald CIs)")
    ap.add_argument("--z", type=float, default=2.5)
    ap.add_argument("--pm-window", type=int, default=200)
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "out", "fab"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    df = pd.read_csv(a.csv)
    df["lot"] = df["lot"].astype(str); df["chamber"] = df["chamber"].astype(str)
    df["time"] = pd.to_datetime(df["time"])
    df["t_month"] = (df["time"] - df["time"].min()).dt.total_seconds() / (86400 * 30.4375)
    meas = df.dropna(subset=["y"]).copy()
    meas["slotpos"] = meas["slot"].astype(int).astype(str)
    out = dict(lots=int(df.lot.nunique()), wafers=int(len(df)), measured_wafers=int(len(meas)))

    # parity: do odd and even slots go to different chambers?
    par = []
    for lot, d in df.groupby("lot"):
        odd, even = d[d.slot % 2 == 1], d[d.slot % 2 == 0]
        if len(odd) and len(even):
            par.append(set(odd.chamber).isdisjoint(set(even.chamber)))
    out["parity"] = dict(lots_with_odd_and_even=len(par), share_different_chamber=round(float(np.mean(par)), 4) if par else None)
    out["kruskal_chamber_p"] = float(kruskal(*[g.y.values for _, g in meas.groupby("chamber")]).pvalue)

    # model (5)
    res = fit(meas)
    ref = sorted(meas.chamber.unique())[0]
    sig_u, sig_e = float(np.sqrt(res.cov_re.iloc[0, 0])), float(np.sqrt(res.scale))
    out["sigma_u_e"] = [round(sig_u, 2), round(sig_e, 2)]
    C = {"slot1-slot14": contrast(res, [slot_name(1)]), "slot24-slot14": contrast(res, [slot_name(24)])}
    pairs = []
    if "pair" in meas.columns:
        for p, d in meas.groupby("pair"):
            ch = sorted(d.chamber.unique())
            if len(ch) == 2:
                c2, c1 = chamber_name(ch[1], ref), chamber_name(ch[0], ref)
                C[f"mismatch {p}"] = contrast(res, [c2] if c2 else [], [c1] if c1 else [])
                raw = mannwhitneyu(d[d.chamber == ch[1]].y, d[d.chamber == ch[0]].y)
                pairs.append(dict(pair=p, first=ch[0], second=ch[1],
                                  raw_diff=round(float(d[d.chamber == ch[1]].y.median() - d[d.chamber == ch[0]].y.median()), 2),
                                  raw_p=float(raw.pvalue)))
    est = {}
    for k, v in C.items():
        tt = res.t_test(v.values.reshape(1, -1))
        est[k] = dict(estimate=round(float(tt.effect[0]), 2), p=float(tt.pvalue),
                      wald_ci=[round(float(x), 2) for x in tt.conf_int()[0]])
    if a.boot:
        X = res.model.exog; beta = res.fe_params.values
        lots_idx = pd.factorize(meas["lot"])[0]
        draws = {k: [] for k in C}
        for b in range(a.boot):
            ystar = X @ beta + rng.normal(0, sig_u, lots_idx.max() + 1)[lots_idx] + rng.normal(0, sig_e, len(meas))
            r = fit(meas.assign(y=ystar))
            for k, v in C.items():
                draws[k].append(float(v.reindex(r.fe_params.index).fillna(0) @ r.fe_params))
        for k in C:
            d = np.array(draws[k])
            est[k]["boot_ci"] = [round(float(x), 2) for x in np.percentile(d, [2.5, 97.5])]
    out["model5"] = est
    for p in pairs:
        p.update({f"adj_{k}": v for k, v in est[f"mismatch {p['pair']}"].items()})
    pd.DataFrame(pairs).to_csv(os.path.join(a.out, "pair_mismatch.csv"), index=False)

    # model-free slot-1 checks
    meas["r_ch"] = meas.y - meas.groupby("chamber").y.transform("median")
    s1 = meas[meas.slot == 1].r_ch; oo = meas[(meas.slot % 2 == 1) & (meas.slot != 1)].r_ch
    if len(s1) and len(oo):
        hl = float(np.median(np.subtract.outer(s1.values, oo.values)))
        out["slot1_vs_other_odd_same_chamber"] = dict(n_slot1=int(len(s1)), n_other_odd=int(len(oo)),
                                                      hodges_lehmann=round(hl, 2),
                                                      p=float(mannwhitneyu(s1, oo).pvalue))
    resid = meas.y - res.fittedvalues
    z = (resid - resid.median()) / (1.4826 * np.median(np.abs(resid - resid.median())))
    is1 = (meas.slot == 1).values
    tab = [[int((np.abs(z[is1]) > a.z).sum()), int((np.abs(z[is1]) <= a.z).sum())],
           [int((np.abs(z[~is1]) > a.z).sum()), int((np.abs(z[~is1]) <= a.z).sum())]]
    out["strong_deviation"] = dict(z=a.z, share_slot1=round(tab[0][0] / max(sum(tab[0]), 1), 4),
                                   share_other=round(tab[1][0] / max(sum(tab[1]), 1), 4),
                                   fisher_p=float(fisher_exact(tab)[1]))
    out["levene_slot1"] = dict(sd_ratio=round(float(resid[is1].std() / resid[~is1].std()), 2),
                               p=float(levene(resid[is1], resid[~is1]).pvalue))

    # pooled start-of-lot test on residuals of model (5) without b_slot
    r0 = fit(meas, with_slot=False)
    meas["r0"] = meas.y - r0.fittedvalues
    lots = [(d.slot.to_numpy(int), d.r0.to_numpy(float)) for _, d in meas.groupby("lot") if len(d) >= 2]
    out["pooled_head_adjusted"] = pooled_test(lots, "head", a.perm, rng)

    # drift per chamber (Theil-Sen, units per month)
    dr = []
    for c, d in meas.groupby("chamber"):
        if len(d) >= 10:
            sl, ic, lo, hi = theilslopes(d.y, d.t_month)
            rho, p = spearmanr(d.t_month, d.y)
            dr.append(dict(chamber=c, n=len(d), spearman_rho=round(float(rho), 3), p=float(p),
                           theil_sen_per_month=round(float(sl), 3), ci95=[round(float(lo), 3), round(float(hi), 3)]))
    pd.DataFrame(dr).to_csv(os.path.join(a.out, "chamber_drift.csv"), index=False)

    # lot-wide excursions and post-PM share
    fixed = res.model.exog @ res.fe_params.values
    zf = pd.Series(meas.y.values - fixed, index=meas.index)
    zf = (zf - zf.median()) / (1.4826 * np.median(np.abs(zf - zf.median())))
    meas["zf"] = zf
    exc = []
    for lot, d in meas.groupby("lot"):
        if len(d) >= 2 and ((d.zf > a.z).all() or (d.zf < -a.z).all()):
            exc.append(dict(lot=lot, kind="thick" if d.zf.mean() > 0 else "thin", tool=d.tool.iloc[0],
                            time=str(d.time.min().date()),
                            pmcount=float(d.pmcount.min()) if "pmcount" in d else np.nan))
    exc = pd.DataFrame(exc)
    exc.to_csv(os.path.join(a.out, "excursions.csv"), index=False)
    out["excursions"] = dict(lots_with_2plus_measured=int((meas.groupby("lot").size() >= 2).sum()),
                             n=int(len(exc)), thin=int((exc.kind == "thin").sum()) if len(exc) else 0,
                             thick=int((exc.kind == "thick").sum()) if len(exc) else 0)
    if "pmcount" in meas.columns and len(exc):
        lot_pm = meas.groupby("lot").pmcount.min()
        share_all = float((lot_pm <= a.pm_window).mean())
        post = lambda k: exc[exc.kind == k].pmcount.le(a.pm_window)
        tab = [[int(post("thin").sum()), int((~post("thin")).sum())],
               [int(post("thick").sum()), int((~post("thick")).sum())]]
        thin_share = tab[0][0] / max(sum(tab[0]), 1)
        out["post_pm"] = dict(window=a.pm_window, share_all_lots=round(share_all, 4),
                              thin_post_pm=tab[0], thick_post_pm=tab[1], fisher_p=float(fisher_exact(tab)[1]),
                              thin_over_representation=round(thin_share / share_all, 2) if share_all else None)
    json.dump(out, open(os.path.join(a.out, "fab_results.json"), "w"), indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
