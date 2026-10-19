"""
S5 - Null data set 1: Infineon/KAI simulated analog wafer-test lots (Zenodo 2542504), Section IV-C.

Second source WITH lot + wafer-number structure (5 lots x ~200 wafers, analog
electrical tests, simulated patterns). Used to show that the pipeline also runs on
ANALOG test data, not only on pass/fail bin maps:

  1. analog value -> robust z-score per test -> die "fails" when |z| > --z
     (gives the same 0/1/2 map format as WM-811K: 0 = off, 1 = pass, 2 = fail)
  2. the slot audit of wm811k_slot_audit.py (odd/even, head, Spearman trend,
     within-lot permutation test) on the per-wafer fail fraction, per test column
  3. example wafer maps for the report

Data are SIMULATED: results show that the code generalises, not that real fabs
have slot effects. Column names are detected automatically (lot, wafer, x, y);
if detection fails the script prints the columns it found.

Slot mapping (each lot has ~200 wafers, a carrier has 25 slots):
  --slot-map first25 : wafers sorted by wafer number, the first 25 become slots 1-25 (default)
  --slot-map blocks  : consecutive blocks of 25 wafers, each block is one lot with slots 1-25

Usage : python s5_kai_null.py --dir data/Wafer --max-tests 4
Output: kai_out/slot_audit_by_test.csv, kai_summary.json, wafer_examples.png, columns_found.json
"""
import argparse, glob, json, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from common import SCRIPT_DIR
from slot_audit import lot_stats


def find_dir(user):
    cands = [user] if user else []
    cands += [p for p in glob.glob(os.path.join(SCRIPT_DIR, "**", "Wafer"), recursive=True) if os.path.isdir(p)]
    for c in cands:
        if c and os.path.isdir(c):
            return c
    raise SystemExit("Folder 'Wafer' (Infineon/KAI) not found. Use --dir <path>.")


def sniff_sep(path):
    with open(path, "r", errors="ignore") as f:
        line = f.readline()
    for s in (";", "\t", ","):
        if line.count(s) >= 2:
            return s
    return r"\s+"


def pick(cols, patterns):
    for p in patterns:
        for c in cols:
            if re.fullmatch(p, c.lower().strip()):
                return c
    return None


def load(folder):
    files = [f for f in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
             if os.path.isfile(f) and f.lower().endswith((".csv", ".txt", ".dat"))
             and "descr" not in os.path.basename(f).lower()]
    if not files:
        raise SystemExit(f"No .csv/.txt/.dat files in {folder}")
    dfs = []
    for f in sorted(files):
        sep = sniff_sep(f)
        df = pd.read_csv(f, sep=sep, engine="c" if sep != r"\s+" else "python", low_memory=False)
        for c in df.columns:  # European files: decimal comma ("0,123")
            if not pd.api.types.is_numeric_dtype(df[c]):
                conv = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
                if conv.notna().mean() > 0.9:
                    df[c] = conv
        df["__file"] = os.path.basename(f)
        dfs.append(df)
        print(f"   {os.path.basename(f)}: {len(df):,} rows, {df.shape[1] - 1} columns")
    df = pd.concat(dfs, ignore_index=True)
    cols = [c for c in df.columns if c != "__file"]
    found = dict(lot=pick(cols, [r"lot.*", r".*lot"]), wafer=pick(cols, [r"wafer.*", r"wf.*", r".*wafer.*"]),
                 x=pick(cols, [r"x", r"die_?x", r"x_?coord.*", r"x.*"]), y=pick(cols, [r"y", r"die_?y", r"y_?coord.*", r"y.*"]))
    if found["lot"] is None:  # one file per lot
        df["__lot"] = df["__file"]; found["lot"] = "__lot"
    meta = set(found.values()) | {"__file"}
    tests = [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])
             and not re.search(r"label|class|pattern|cluster", c.lower())]
    # columns *_pure are the noise-free version of the same pattern -> reference only
    found["pure_reference"] = [c for c in tests if c.lower().endswith("_pure")]
    tests = [c for c in tests if not c.lower().endswith("_pure")]
    found["tests"] = tests
    found["all_columns"] = cols
    return df, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--z", type=float, default=3.0, help="robust z threshold for a failing die")
    ap.add_argument("--max-tests", type=int, default=4)
    ap.add_argument("--slot-map", default="first25", choices=["first25", "blocks"])
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "out", "kai_null"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    df, f = load(find_dir(a.dir))
    json.dump({k: v for k, v in f.items()}, open(os.path.join(a.out, "columns_found.json"), "w"), indent=2)
    print("detected:", {k: v for k, v in f.items() if k != "all_columns" and k != "tests"},
          f"| {len(f['tests'])} test columns")
    if not f["tests"]:
        raise SystemExit("No numeric test columns. See columns_found.json.")
    if f["wafer"] is None or f["x"] is None or f["y"] is None:
        raise SystemExit("Wafer / x / y columns not detected. See columns_found.json "
                         "and adapt the column patterns in load().")
    L, W, X, Y = f["lot"], f["wafer"], f["x"], f["y"]
    df[W] = pd.to_numeric(df[W], errors="coerce")
    df = df.dropna(subset=[W])
    df[W] = df[W].astype(int)
    df["__xi"] = (df[X] - df[X].min()).round().astype(int)
    df["__yi"] = (df[Y] - df[Y].min()).round().astype(int)
    H, Wd = df["__yi"].max() + 1, df["__xi"].max() + 1

    rows, examples = [], []
    for t in f["tests"][: a.max_tests]:
        v = df[t].to_numpy(float)
        med = np.nanmedian(v); mad = 1.4826 * np.nanmedian(np.abs(v - med))
        if not np.isfinite(mad) or mad == 0:
            continue
        df["__fail"] = np.abs((v - med) / mad) > a.z
        per = df.groupby([L, W])["__fail"].mean().rename("y").reset_index()
        units = []
        for lot, d in per.groupby(L):
            d = d.sort_values(W).reset_index(drop=True)
            if a.slot_map == "first25":
                units.append((lot, d.iloc[:25]))
            else:
                for b in range(len(d) // 25):
                    units.append((f"{lot}_b{b + 1}", d.iloc[25 * b:25 * (b + 1)]))
        for lot, d in units:
            s, y = np.arange(1, len(d) + 1), d["y"].to_numpy(float)
            if len(d) < 10 or np.ptp(y) == 0:
                continue
            (dp, dh, r), (pp, ph, pr) = lot_stats(s, y, a.perm, rng)
            rows.append(dict(test=t, lot=lot, wafers=len(d), mean_fail=round(float(y.mean()), 4),
                             d_par=dp, d_head=dh, rho=r, p_par=pp, p_head=ph, p_rho=pr))
        if len(examples) < 2:
            lot0 = per[L].iloc[0]
            for w in sorted(per[per[L] == lot0][W].unique())[:4]:
                d = df[(df[L] == lot0) & (df[W] == w)]
                m = np.zeros((H, Wd), np.uint8)
                m[d["__yi"], d["__xi"]] = np.where(d["__fail"], 2, 1)
                examples.append((t, lot0, w, m))
    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("No test/lot could be analysed.")
    res.to_csv(os.path.join(a.out, "slot_audit_by_test.csv"), index=False)
    summ = dict(dataset="Infineon/KAI simulated (Zenodo 2542504)", slot_map=a.slot_map, lots=int(res.lot.nunique()),
                tests_analysed=int(res.test.nunique()), lot_test_pairs=int(len(res)), z_threshold=a.z,
                share_p05={k: round(float((res[f"p_{k}"] < 0.05).mean()), 4) for k in ("par", "head", "rho")},
                note="simulated data: shows the pipeline runs on analog test data; 5% = chance")
    json.dump(summ, open(os.path.join(a.out, "kai_summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))

    cmap = ListedColormap(["white", "#BFD7EA", "#C00000"])
    fig, ax = plt.subplots(1, len(examples), figsize=(2.6 * len(examples), 2.9), squeeze=False)
    for k, (t, lot, w, m) in enumerate(examples):
        ax[0, k].imshow(m, cmap=cmap, vmin=0, vmax=2, interpolation="nearest"); ax[0, k].axis("off")
        ax[0, k].set_title(f"{t}\nlot {lot}, wafer {w}", fontsize=8)
    fig.suptitle(f"Infineon/KAI: analog test -> fail map (|robust z| > {a.z})", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "wafer_examples.png"), dpi=140); plt.close(fig)
    print(f"Saved to {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
