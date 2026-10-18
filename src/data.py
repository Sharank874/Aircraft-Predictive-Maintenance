"""
Data pipeline for Predictive Maintenance & Risk Scoring (NASA C-MAPSS turbofan data).

Design decisions (all leakage-aware):
  * Split by ENGINE UNIT, never by row/window: windows from one engine's degradation
    trajectory never appear in both train and validation.
  * Scaler is fitted on training units only, then applied to val / test / streaming data.
  * Each window holds only cycles <= t (the prediction time), so the BiLSTM's backward
    pass never sees the future relative to the prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

# Sensors that are constant (or near-constant) in FD001 carry no information. They are
# identified from the training data below (std ~ 0), not hard-coded from the literature.
RUL_CAP = 125      # piecewise-linear RUL: engine is treated as "healthy" above 125 cycles
HORIZON = 30       # fault label = "failure within the next 30 cycles"
WINDOW = 30        # input sequence length (cycles)


def load_raw(subset: str = "FD001"):
    rd = lambda f: pd.read_csv(DATA_DIR / f, sep=r"\s+", header=None, names=COLS)
    train = rd(f"train_{subset}.txt")
    test = rd(f"test_{subset}.txt")
    rul = pd.read_csv(DATA_DIR / f"RUL_{subset}.txt", header=None, names=["rul_end"])
    rul["unit"] = np.arange(1, len(rul) + 1)
    return train, test, rul


def add_targets(df: pd.DataFrame, rul_end: pd.DataFrame | None = None) -> pd.DataFrame:
    """True RUL at every cycle. Train units run to failure; test units are truncated and
    their true RUL at the last observed cycle comes from RUL_FDxxx.txt."""
    df = df.copy()
    last = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = last - df["cycle"]
    if rul_end is not None:
        df = df.merge(rul_end, on="unit", how="left")
        df["rul"] = df["rul"] + df["rul_end"]
        df = df.drop(columns="rul_end")
    df["rul_capped"] = df["rul"].clip(upper=RUL_CAP)
    # Risk index in [0, 1]: 0 = healthy plateau, 1 = at failure. A monotone, normalised
    # transform of capped RUL -> a defined, documented regression target (NOT a probability).
    df["risk"] = 1.0 - df["rul_capped"] / RUL_CAP
    df["fault"] = (df["rul"] <= HORIZON).astype(int)
    return df


def informative_sensors(train: pd.DataFrame, min_unique: int = 5) -> list[str]:
    """Drop sensors that are constant or take only a couple of discrete values in the
    training data (they carry no degradation signal). Decided from training data only."""
    cand = [c for c in COLS if c.startswith("s")]
    nu = train[cand].nunique()
    return [c for c in cand if nu[c] > min_unique]


class Scaler:
    """Z-score scaler fitted on training units only."""
    def fit(self, df, cols):
        self.cols, self.mu, self.sd = cols, df[cols].mean(), df[cols].std().replace(0, 1)
        return self

    def transform(self, df):
        df = df.copy()
        df[self.cols] = (df[self.cols] - self.mu) / self.sd
        return df


def make_windows(df: pd.DataFrame, feats: list[str], window: int = WINDOW, last_only=False):
    """Sliding windows (stride 1). Window ending at cycle t -> targets at cycle t."""
    X, y_fault, y_risk, y_rul, units, cycles = [], [], [], [], [], []
    for u, g in df.groupby("unit"):
        a = g[feats].to_numpy(np.float32)
        if len(a) < window:          # left-pad very short sequences with first row
            a = np.vstack([np.repeat(a[:1], window - len(a), 0), a])
            g = pd.concat([g.iloc[[0]]] * (window - len(g)) + [g])
        ends = [len(a)] if last_only else range(window, len(a) + 1)
        for e in ends:
            X.append(a[e - window:e])
            row = g.iloc[e - 1]
            y_fault.append(row["fault"]); y_risk.append(row["risk"]); y_rul.append(row["rul"])
            units.append(u); cycles.append(row["cycle"])
    return (np.stack(X), np.array(y_fault, np.float32), np.array(y_risk, np.float32),
            np.array(y_rul, np.float32), np.array(units), np.array(cycles))


def prepare(subset="FD001", val_frac=0.15, rtf_frac=0.15, seed=42, test_subset=None):
    """Returns dict of arrays (+ metadata).
      train / val : 70% / 15% of the run-to-failure ENGINES in train_FD001 (unit-level split)
      rtf         : 15% of run-to-failure engines never used for training, early stopping or
                    threshold selection -> used for lead-time / false-alarm analysis
      test        : official NASA test set (100 unseen, truncated engines), every window
      test_last   : last window of each test engine (the standard C-MAPSS RUL benchmark)
      shift       : optional other subset (e.g. FD003 = extra fan fault mode) for drift tests"""
    train, test, rul = load_raw(subset)
    train, test = add_targets(train), add_targets(test, rul)
    feats = informative_sensors(train)

    rng = np.random.default_rng(seed)
    units = rng.permutation(train["unit"].unique())
    n_va, n_rtf = int(len(units) * val_frac), int(len(units) * rtf_frac)
    val_units, rtf_units = units[:n_va], units[n_va:n_va + n_rtf]
    tr = train[~train.unit.isin(np.concatenate([val_units, rtf_units]))]
    va = train[train.unit.isin(val_units)]
    rt = train[train.unit.isin(rtf_units)]

    scaler = Scaler().fit(tr, feats)
    out = {"feats": feats, "scaler": scaler, "val_units": sorted(val_units.tolist()),
           "rtf_units": sorted(rtf_units.tolist()), "n_train_units": tr.unit.nunique()}
    for name, d in [("train", tr), ("val", va), ("rtf", rt), ("test", test)]:
        out[name] = make_windows(scaler.transform(d), feats)
    out["test_last"] = make_windows(scaler.transform(test), feats, last_only=True)

    if test_subset:  # shifted-distribution evaluation (e.g., FD003: extra fan fault mode)
        _, t2, r2 = load_raw(test_subset)
        t2 = add_targets(t2, r2)
        out["shift"] = make_windows(scaler.transform(t2), feats)
    out["raw_train"] = train
    return out


if __name__ == "__main__":
    d = prepare(test_subset="FD003")
    print("features:", d["feats"])
    for k in ["train", "val", "rtf", "test", "test_last", "shift"]:
        X, yf, yr, rul, u, c = d[k]
        print(f"{k:9s} X={X.shape} units={len(np.unique(u))} fault_rate={yf.mean():.3f}")
