"""Basket completion: which category is missing from this basket?

Each test basket has every studied category hidden in turn; a score ranks how likely the hidden
category is to be in it, given the household's history and the other categories present. The
association between categories comes from training weeks only. Four scores compete:

- popularity: how often the category is bought at all;
- household habit: how often this household bought it in training (smoothed);
- habit + lift: habit plus the mean log lift with the categories in the basket;
- habit + taste-adjusted association: habit plus the mean residual co-purchase, after removing
  household tastes and trip size.

The weight on the association term is chosen on validation weeks and the comparison is made on
test weeks the weight never saw. Area under the ROC curve is computed over all (basket,
category) pairs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import mba


def auc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = sums[inv] / cnt[inv]
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _matrix(df: pd.DataFrame, col: str, cats: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(cats)}
    A = np.zeros((len(cats), len(cats)))
    for r in df.itertuples():
        i, j = idx[r.j], idx[r.k]
        A[i, j] = A[j, i] = getattr(r, col)
    return A


def scores(y: pd.DataFrame, b: pd.DataFrame, train: np.ndarray, target: np.ndarray, assoc: np.ndarray | None,
           w: float) -> np.ndarray:
    Y = y.to_numpy(dtype=float)
    hh = b["household_id"].to_numpy()
    tr = pd.DataFrame(Y[train]).groupby(hh[train]).agg(["sum", "count"])
    K = Y.shape[1]
    pop = Y[train].mean(axis=0)
    sums = tr.xs("sum", axis=1, level=1)
    cnt = tr.xs("count", axis=1, level=1).iloc[:, 0]
    prior = 5.0
    habit = (sums.add(prior * pop, axis=1)).div(cnt + prior, axis=0)
    H = habit.reindex(hh[target]).to_numpy()
    H = np.where(np.isnan(H), pop, H)
    S = np.log(H / (1 - H))
    if assoc is not None and w != 0:
        Yt = Y[target]
        present = Yt.sum(axis=1, keepdims=True)
        # mean association with the other categories present (the hidden one excluded)
        tot = Yt @ assoc
        own = Yt * np.diag(assoc)
        others = np.maximum(present - Yt, 1)
        S = S + w * (tot - own) / others
    return S


def evaluate(y: pd.DataFrame, b: pd.DataFrame, train_weeks=(1, 26), valid_weeks=(27, 39), test_weeks=(40, 53)) -> pd.DataFrame:
    wk = b["week"].to_numpy()
    tr = (wk >= train_weeks[0]) & (wk <= train_weeks[1])
    va = (wk >= valid_weeks[0]) & (wk <= valid_weeks[1])
    te = (wk >= test_weeks[0]) & (wk <= test_weeks[1])
    cats = list(y.columns)
    yt, bt = y[tr], b[tr]
    lift = mba.pair_table(yt)
    lift["log_lift"] = np.log(lift["lift"])
    resid = mba.residual_correlation(yt, bt["household_id"], bt["items"])
    A_lift, A_res = _matrix(lift, "log_lift", cats), _matrix(resid, "resid_corr", cats)
    Y = y.to_numpy()
    rows = []
    for name, A in (("habit + lift", A_lift), ("habit + taste-adjusted association", A_res)):
        best, bw = -1, 0.0
        for w in (0.25, 0.5, 1, 2, 4, 8, 16):
            a = auc(Y[va].ravel(), scores(y, b, tr | False, va, A, w).ravel())
            if a > best:
                best, bw = a, w
        rows.append({"model": name, "weight": bw, "auc_valid": best,
                     "auc_test": auc(Y[te].ravel(), scores(y, b, tr | va, te, A, bw).ravel())})
    pop = np.tile(Y[tr | va].mean(axis=0), (te.sum(), 1))
    rows.insert(0, {"model": "popularity", "weight": 0, "auc_valid": np.nan, "auc_test": auc(Y[te].ravel(), pop.ravel())})
    rows.insert(1, {"model": "household habit", "weight": 0, "auc_valid": np.nan,
                    "auc_test": auc(Y[te].ravel(), scores(y, b, tr | va, te, None, 0).ravel())})
    return pd.DataFrame(rows)
