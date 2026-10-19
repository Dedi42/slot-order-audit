"""
S6 - Null data set 2: PHM 2018 ion-mill sensor data in run order (Section IV-C).
Do process sensors trend along run order?

This is NOT a wafer-map dataset. It is used only to show, on public fab data,
that the effect behind the "ramp / drift" slot pattern (slide 3) is real.

Per tool: rows are aggregated to one value per run (= one wafer pass,
key Tool + Lot + runnum when present), runs are ordered by time, then
    rho = Spearman(run order, run-mean of each sensor)
Consumable counters (ETCHSOURCEUSAGE, ...) rise by construction and are
reported separately as the "usage clock"; the drift claim uses the process sensors.

Files: phm_data_challenge_2018/train/*.csv (fault / ttf files are skipped).
Files are large (millions of rows) and are read in chunks.

Usage : python s6_phm2018_drift.py --dir data/phm_data_challenge_2018
        (quick test: --max-files 2 --max-rows 2000000)
Output: phm18_out/run_level.csv, drift_by_tool.csv, drift_summary.json, drift_trends.png
"""
import argparse, glob, json, os, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import SCRIPT_DIR

KEYS = ["tool", "lot", "runnum"]
USAGE = ["etchsourceusage", "etchauxsourcetimer", "etchaux2sourcetimer"]
SKIP = ["time", "tool", "stage", "lot", "runnum", "recipe", "recipe_step"]


def find_dir(user):
    cands = [user] if user else []
    cands += glob.glob(os.path.join(SCRIPT_DIR, "**", "phm_data_challenge_2018"), recursive=True)
    for c in cands:
        if c and os.path.isdir(c):
            return c
    raise SystemExit("Folder phm_data_challenge_2018 not found. Use --dir <path>.")


def aggregate_file(path, max_rows, chunk=1_000_000):
    """Return one row per run with the mean of every numeric sensor and the first time stamp."""
    acc, cols, rows = [], None, 0
    for df in pd.read_csv(path, chunksize=chunk, low_memory=False):
        df.columns = [c.strip().lower() for c in df.columns]
        if cols is None:
            keys = [k for k in KEYS if k in df.columns]
            if not keys:
                raise SystemExit(f"{os.path.basename(path)}: no Tool/Lot/runnum columns. Columns: {list(df.columns)}")
            sensors = [c for c in df.columns if c not in SKIP and pd.api.types.is_numeric_dtype(df[c])]
            cols = (keys, sensors)
        keys, sensors = cols
        g = df.groupby(keys, sort=False)
        part = g[sensors].agg(["sum", "count"])
        part.columns = [f"{a}__{b}" for a, b in part.columns]
        if "time" in df.columns:
            part["t0"] = g["time"].min()
        acc.append(part)
        rows += len(df)
        if max_rows and rows >= max_rows:
            break
    allp = pd.concat(acc)
    keys, sensors = cols
    agg = {c: "sum" for c in allp.columns if c != "t0"}
    if "t0" in allp.columns:
        agg["t0"] = "min"
    allp = allp.groupby(level=list(range(len(keys)))).agg(agg)
    out = pd.DataFrame(index=allp.index)
    for s in sensors:
        out[s] = allp[f"{s}__sum"] / allp[f"{s}__count"].replace(0, np.nan)
    if "t0" in allp.columns:
        out["t0"] = allp["t0"]
    out = out.reset_index()
    if "tool" not in out.columns:
        out["tool"] = os.path.basename(path).split("_")[1] if "_" in os.path.basename(path) else os.path.basename(path)
    out["source_file"] = os.path.basename(path)
    return out, sensors, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0, help="rows per file (0 = all)")
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "out", "phm2018"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = find_dir(a.dir)
    files = sorted(f for f in glob.glob(os.path.join(d, a.split, "*.csv"))
                   if "fault" not in os.path.basename(f).lower() and "ttf" not in os.path.basename(f).lower())
    if a.max_files:
        files = files[: a.max_files]
    if not files:
        raise SystemExit(f"No CSV files in {os.path.join(d, a.split)}")
    print(f"{len(files)} files in {os.path.join(d, a.split)}")

    t0, parts, sensors = time.time(), [], None
    for f in files:
        r, s, n = aggregate_file(f, a.max_rows)
        sensors = sensors or s
        parts.append(r)
        print(f"   {os.path.basename(f)}: {n:,} rows -> {len(r):,} runs  ({time.time() - t0:.0f}s)")
    runs = pd.concat(parts, ignore_index=True)
    runs.to_csv(os.path.join(a.out, "run_level.csv"), index=False)

    order_col = "t0" if "t0" in runs.columns else "runnum"
    proc = [s for s in sensors if s not in USAGE and s not in ("t0",)]
    use = [s for s in sensors if s in USAGE]
    rows = []
    for tool, g in runs.groupby("tool"):
        g = g.sort_values(order_col)
        if len(g) < 30:
            continue
        idx = np.arange(len(g))
        for s in proc + use:
            v = g[s].to_numpy(float)
            ok = np.isfinite(v)
            if ok.sum() < 30 or np.nanstd(v) == 0:
                continue
            rho, p = spearmanr(idx[ok], v[ok])
            rows.append(dict(tool=tool, sensor=s, kind="usage counter" if s in use else "process sensor",
                             runs=int(ok.sum()), rho=round(float(rho), 4), p=float(p)))
    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("Not enough runs per tool (need >= 30).")
    res.to_csv(os.path.join(a.out, "drift_by_tool.csv"), index=False)

    ps = res[res.kind == "process sensor"]
    strong = ps[(ps.rho.abs() >= 0.3) & (ps.p < 0.001)]
    summ = dict(files=len(files), tools=int(res.tool.nunique()), runs=int(len(runs)),
                process_sensor_tool_pairs=int(len(ps)),
                n_pairs_with_drift=int(len(strong)),
                share_pairs_with_drift=round(float(len(strong) / max(len(ps), 1)), 4),
                criterion="|Spearman rho(run order, run mean)| >= 0.3 and p < 0.001",
                tools_with_any_drifting_sensor=int(strong.tool.nunique()),
                top_drifting_sensors=strong.assign(a=strong.rho.abs()).groupby("sensor").a.median()
                .sort_values(ascending=False).head(6).round(3).to_dict())
    json.dump(summ, open(os.path.join(a.out, "drift_summary.json"), "w"), indent=2)
    print("\n== DRIFT SUMMARY =="); print(json.dumps(summ, indent=2))

    top = list(summ["top_drifting_sensors"])[:4] or proc[:4]
    tool = (strong.tool.value_counts().index[0] if len(strong) else runs.tool.iloc[0])
    g = runs[runs.tool == tool].sort_values(order_col)
    fig, ax = plt.subplots(1, len(top), figsize=(4 * len(top), 3.2), squeeze=False)
    for k, s in enumerate(top):
        v = g[s].to_numpy(float)
        ax[0, k].plot(v, lw=0.6, color="#A6A6A6")
        ax[0, k].plot(pd.Series(v).rolling(max(5, len(v) // 50), min_periods=1).mean(), color="#C00000", lw=2)
        rho = res[(res.tool == tool) & (res.sensor == s)].rho
        ax[0, k].set_title(f"{s}\nrho = {rho.iloc[0] if len(rho) else float('nan'):.2f}", fontsize=9)
        ax[0, k].set_xlabel("run order")
    fig.suptitle(f"PHM 2018 tool {tool}: run-mean of process sensors (red = rolling mean)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "drift_trends.png"), dpi=150); plt.close(fig)
    print(f"Saved to {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
