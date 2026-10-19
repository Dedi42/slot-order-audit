"""
Core of the outcome-agnostic slot-order audit (Section III-A, Steps 2-5).

Every function works on one lot given as slot numbers s (1..25, possibly a
subset = masked slots) and one scalar outcome y per observed slot.

    d_par  = mean y(odd) - mean y(even)          (2)  twin chamber
    d_head = mean y(s<=3) - mean y(s>3)          (2)  start-of-lot / first wafer
    rho    = Spearman(s, y)                      (3)  drift
    d_k    = max_c mean y(s mod k = c) - min_c   (3)  k-chamber rotation

Permutation p-values follow (4): p = (1 + #{|T_b| >= |T|}) / (B + 1).
"""
import numpy as np
from scipy.stats import rankdata, spearmanr, ttest_ind

STATS = ("par", "head", "rho")
STAT_NAMES = {"par": "odd/even", "head": "start-of-lot", "rho": "trend"}


# ---------------------------------------------------------------- statistics
def _stats(Y, RY, odd, head, rs_c):
    """Vectorised statistics; Y, RY have shape (..., n)."""
    zero = np.zeros(Y.shape[:-1])
    d_par = Y[..., odd].mean(-1) - Y[..., ~odd].mean(-1) if odd.any() and (~odd).any() else zero
    d_head = Y[..., head].mean(-1) - Y[..., ~head].mean(-1) if head.any() and (~head).any() else zero
    ryc = RY - RY.mean(-1, keepdims=True)
    den = np.sqrt((ryc ** 2).sum(-1) * (rs_c ** 2).sum())
    rho = np.where(den > 0, (ryc * rs_c).sum(-1) / np.where(den > 0, den, 1), 0.0)
    return d_par, d_head, rho


def lot_stats(slots, y, B, rng, return_null=False):
    """Observed (d_par, d_head, rho) and within-lot permutation p-values (4)."""
    slots = np.asarray(slots); y = np.asarray(y, float)
    odd, head = slots % 2 == 1, slots <= 3
    ry = rankdata(y); rs = rankdata(slots); rs_c = rs - rs.mean()
    obs = _stats(y, ry, odd, head, rs_c)
    idx = np.argsort(rng.random((B, y.size)), axis=1)
    null = _stats(y[idx], ry[idx], odd, head, rs_c)
    pv = [(1 + (np.abs(n) >= abs(o) - 1e-12).sum()) / (B + 1) for o, n in zip(obs, null)]
    obs = [float(o) for o in obs]
    return (obs, pv, null) if return_null else (obs, pv)


def period_stat(slots, y, k):
    s = np.asarray(slots); y = np.asarray(y, float)
    means = [y[s % k == c].mean() for c in range(k) if (s % k == c).any()]
    return max(means) - min(means) if len(means) > 1 else 0.0


def period_test(slots, y, k, B, rng):
    """Period-k contrast d_k with its within-lot permutation p-value."""
    y = np.asarray(y, float)
    obs = period_stat(slots, y, k)
    null = np.array([period_stat(slots, y[rng.permutation(y.size)], k) for _ in range(B)])
    return obs, (1 + (null >= obs - 1e-12).sum()) / (B + 1)


def analytic_p(slots, y):
    """Analytic p-values without permutation (Welch t-test, asymptotic Spearman)."""
    s = np.asarray(slots); y = np.asarray(y, float)
    odd, head = s % 2 == 1, s <= 3
    out = []
    for g in (odd, head):
        if g.sum() >= 2 and (~g).sum() >= 2 and np.ptp(y) > 0:
            p = ttest_ind(y[g], y[~g], equal_var=False).pvalue
            out.append(float(p) if np.isfinite(p) else 1.0)
        else:
            out.append(1.0)
    p = spearmanr(s, y).pvalue if np.ptp(y) > 0 else 1.0
    out.append(float(p) if np.isfinite(p) else 1.0)
    return out


# ------------------------------------------------------- pooled sparse test
def pooled_test(lots, stat, B, rng, squared=None):
    """Pooled permutation test for sparsely measured lots (box 3-2 in Fig. 1).

    lots    : list of (slots, y) with at least 2 observed slots each
    stat    : 'par', 'head' or 'rho'
    squared : pool (T_l / sd_l)^2 instead of T_l / sd_l. Default True for 'par'
              (sign depends on which twin gets the odd slots), False otherwise.
    Every lot is permuted independently in every replicate.
    """
    k = STATS.index(stat)
    squared = (stat == "par") if squared is None else squared
    obs_sum, null_sum, used = 0.0, np.zeros(B), 0
    for slots, y in lots:
        y = np.asarray(y, float)
        if y.size < 2 or np.ptp(y) == 0:
            continue
        obs, _, null = lot_stats(slots, y, B, rng, return_null=True)
        sd = null[k].std()
        if sd <= 0:
            continue
        t, tn = obs[k] / sd, null[k] / sd
        obs_sum += t ** 2 if squared else t
        null_sum += tn ** 2 if squared else tn
        used += 1
    if squared:
        p = (1 + (null_sum >= obs_sum - 1e-12).sum()) / (B + 1)
    else:
        p = (1 + (np.abs(null_sum) >= abs(obs_sum) - 1e-12).sum()) / (B + 1)
    return dict(stat=stat, squared=bool(squared), lots_used=used, T_pool=float(obs_sum), p=float(p))


# ---------------------------------------------------- dataset-level inference
def bh(p):
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, float); n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.minimum(q, 1.0)
    return out


def storey_pi1(p, lam=0.5):
    """Storey's estimate of the share of lots with a true effect."""
    p = np.asarray(p)
    return float(1 - min(1.0, (p > lam).mean() / (1 - lam)))
