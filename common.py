"""Shared helpers: file lookup and WM-811K loading."""
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_file(user_path, default_name):
    """Return an existing path for user_path (or default_name), searching the repo folder."""
    names = [user_path] if user_path else []
    names += [default_name, os.path.join("data", os.path.basename(default_name)),
              os.path.join("datasets", os.path.basename(default_name))]
    for n in names:
        for c in (n, os.path.join(SCRIPT_DIR, n)):
            if c and os.path.exists(c):
                return os.path.abspath(c)
    raise SystemExit(f"File not found: {user_path or default_name}. "
                     f"Put it in {SCRIPT_DIR}/data/ or pass the path explicitly.")


def load_lswmd(path):
    """Read LSWMD.pkl (written with an old pandas) under recent pandas versions."""
    import pandas.core.indexes as _idx
    sys.modules.setdefault("pandas.indexes", _idx)
    for old, new in (("pandas.indexes.base", "pandas.core.indexes.base"),
                     ("pandas.indexes.numeric", "pandas.core.indexes.numeric")):
        try:
            sys.modules.setdefault(old, __import__(new, fromlist=["_"]))
        except Exception:
            pass
    return pd.read_pickle(path)


def norm_label(v):
    try:
        a = np.asarray(v, dtype=object).ravel()
        return str(a[0]) if a.size else "unlabeled"
    except Exception:
        return "unlabeled"


def fail_rate(m):
    """Fail fraction of a wafer map (0 = off wafer, 1 = pass, 2 = fail)."""
    m = np.asarray(m)
    dies = (m > 0).sum()
    return float((m == 2).sum()) / dies if dies else np.nan


def prepare_df(df):
    """One row per (lot, slot) with y = fail fraction, slots 1-25 only."""
    df = df.copy()
    df["lotName"] = df["lotName"].astype(str)
    df["slot"] = pd.to_numeric(df["waferIndex"], errors="coerce")
    df["label"] = df["failureType"].map(norm_label)
    df["y"] = df["waferMap"].map(fail_rate)
    df = df.dropna(subset=["slot", "y"])
    df["slot"] = df["slot"].astype(int)
    df = df[(df["slot"] >= 1) & (df["slot"] <= 25)]
    df = df.drop_duplicates(["lotName", "slot"])
    if "dieSize" in df.columns:
        df["dieSize"] = pd.to_numeric(df["dieSize"], errors="coerce")
    return df


def shuffle_slots(df, rng):
    """Negative control: permute slot numbers inside every lot."""
    df = df.copy()
    df["slot"] = df.groupby("lotName")["slot"].transform(lambda x: rng.permutation(x.to_numpy()))
    return df


def lots_from_df(df, min_wafers):
    """List of (lot, slots, y) for lots with at least min_wafers wafers."""
    out = []
    for lot, d in df.groupby("lotName"):
        if len(d) >= min_wafers:
            d = d.sort_values("slot")
            out.append((lot, d["slot"].to_numpy(int), d["y"].to_numpy(float)))
    return out
