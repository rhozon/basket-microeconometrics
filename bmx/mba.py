"""Market basket analysis, and what lift actually measures.

Lift(j, k) = P(j and k in a basket) / (P(j) P(k)). Above one, the pair is bought together more
often than chance. Three things push it up, and only one of them helps a promotion: true
complementarity, big trips that contain everything, and households that like both. The two
adjusted measures below remove the second and the third.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pair_table(y: pd.DataFrame) -> pd.DataFrame:
    Y = y.to_numpy(dtype=float)
    n = len(Y)
    p = Y.mean(axis=0)
    co = (Y.T @ Y) / n
    rows = []
    cats = list(y.columns)
    for j in range(len(cats)):
        for k in range(j + 1, len(cats)):
            rows.append({"j": cats[j], "k": cats[k], "support": co[j, k], "p_j": p[j], "p_k": p[k],
                         "lift": co[j, k] / (p[j] * p[k]), "conf_j_to_k": co[j, k] / p[j], "conf_k_to_j": co[j, k] / p[k]})
    return pd.DataFrame(rows)


def lift_by_size(y: pd.DataFrame, size: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Lift within deciles of basket size, averaged with weights equal to joint purchases."""
    q = pd.qcut(size.rank(method="first"), bins, labels=False)
    Y = y.to_numpy(dtype=float)
    K = Y.shape[1]
    num = np.zeros((K, K))
    den = np.zeros((K, K))
    for g in range(bins):
        Yg = Y[(q == g).to_numpy()]
        if len(Yg) == 0:
            continue
        p = Yg.mean(axis=0)
        co = (Yg.T @ Yg) / len(Yg)
        expected = np.outer(p, p)
        num += co * len(Yg)
        den += expected * len(Yg)
    lift = num / np.where(den > 0, den, np.nan)
    cats = list(y.columns)
    return pd.DataFrame([{"j": cats[j], "k": cats[k], "lift_size_adj": lift[j, k]}
                         for j in range(K) for k in range(j + 1, K)])


def residual_correlation(y: pd.DataFrame, household: pd.Series, size: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Correlation of purchases after removing each household's average and the trip size."""
    q = pd.qcut(size.rank(method="first"), bins, labels=False)
    Y = y.astype(float)
    for _ in range(10):
        Y = Y - Y.groupby(household.values).transform("mean")
        Y = Y - Y.groupby(q.values).transform("mean")
    R = np.corrcoef(Y.to_numpy().T)
    cats = list(y.columns)
    K = len(cats)
    return pd.DataFrame([{"j": cats[j], "k": cats[k], "resid_corr": R[j, k]} for j in range(K) for k in range(j + 1, K)])
