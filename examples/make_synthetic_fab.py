"""Write a SYNTHETIC per-wafer table in the format of s8_fab_slot_chamber.py (for testing only).

True effects: chamber offsets, tool T2 +10, slot-1 effect -6, residual SD 15, lot SD 14.
    python examples/make_synthetic_fab.py --out data/synthetic_fab.csv
"""
import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/synthetic_fab.csv")
ap.add_argument("--lots", type=int, default=300)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
t0 = pd.Timestamp("2023-04-01")
ch_eff = {f"T{t}-{p}{i}": rng.normal(0, 5) + (10 if t == 2 else 0) for t in (1, 2) for p in "ABC" for i in (1, 2)}
rows = []
for l in range(a.lots):
    tool, pair = int(rng.integers(1, 3)), str(rng.choice(list("ABC")))
    odd_first = rng.random() < 0.62
    u, day, pm = rng.normal(0, 14), int(rng.integers(0, 480)), int(rng.integers(0, 3000))
    for s in range(1, 26):
        c = f"T{tool}-{pair}{1 if (s % 2 == 1) == odd_first else 2}"
        y = np.nan
        if s in (1, 14, 24) or rng.random() < 0.1:
            y = 3000 + ch_eff[c] + u + (-6 if s == 1 else 0) + rng.normal(0, 15)
        rows.append(dict(lot=f"L{l:04d}", slot=s, chamber=c, tool=f"T{tool}", pair=f"T{tool}-{pair}",
                         time=t0 + pd.Timedelta(days=day), pmcount=pm + s, y=y))
pd.DataFrame(rows).to_csv(a.out, index=False)
print(f"wrote {a.out}")
