"""Category purchase incidence as a function of every studied category's price and promotions.

For each category j, a linear probability model on all baskets:

    1[j in basket b] = household effect + month effect + holiday weeks
                       + sum_k beta_jk log price_k + sum_k gamma_jk log(1 + displays_k)
                       + sum_k lambda_jk log(1 + mailer items_k) + error

beta_jk for k != j is the cross-price effect: negative means a higher price of k makes j less
likely (complements), positive means more likely (substitutes). Household effects remove the
lasting tastes that make lift high for pairs that are merely liked by the same people. Errors
are clustered by household. All equations share one design matrix, so they are solved together.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import HOLIDAY_WEEKS


def controls(b: pd.DataFrame) -> pd.DataFrame:
    c = pd.DataFrame(index=b.index)
    month = np.minimum((b["week"] - 1) // 4.42, 11).astype(int)
    for m in range(1, 12):
        c[f"m{m}"] = (month == m).astype(float)
    for w, name in HOLIDAY_WEEKS.items():
        c[name] = (b["week"] == w).astype(float)
    return c


def within(a: np.ndarray, groups: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(a)
    return (df - df.groupby(groups).transform("mean")).to_numpy()


def fit(y: pd.DataFrame, X: pd.DataFrame, b: pd.DataFrame) -> dict:
    C = controls(b)
    Z = pd.concat([X, C], axis=1)
    g = b["household_id"].to_numpy()
    Zw = within(Z.to_numpy(dtype=float), g)
    Yw = within(y.to_numpy(dtype=float), g)
    keep = Zw.std(axis=0) > 0
    Zw, names = Zw[:, keep], list(Z.columns[keep])
    XtX_inv = np.linalg.pinv(Zw.T @ Zw)
    B = XtX_inv @ Zw.T @ Yw
    E = Yw - Zw @ B
    se = np.zeros_like(B)
    codes, uniq = pd.factorize(g)
    n, k = Zw.shape
    G = len(uniq)
    adj = G / (G - 1) * (n - 1) / (n - k - G)
    for j in range(Yw.shape[1]):
        S = np.zeros((G, k))
        np.add.at(S, codes, Zw * E[:, [j]])
        V = XtX_inv @ (S.T @ S) @ XtX_inv * adj
        se[:, j] = np.sqrt(np.clip(np.diag(V), 0, None))
    coef = pd.DataFrame(B, index=names, columns=y.columns)
    ses = pd.DataFrame(se, index=names, columns=y.columns)
    return {"coef": coef, "se": ses, "mean_y": y.mean(), "n": int(n), "households": int(G)}


def cross_table(res: dict, cats: list[str]) -> pd.DataFrame:
    """beta_jk and its standard error for every ordered pair, with the implied elasticity."""
    rows = []
    for j, cj in enumerate(cats):
        for k, ck in enumerate(cats):
            name = f"lp_{k}"
            if name not in res["coef"].index:
                continue
            b, s = res["coef"].loc[name, cj], res["se"].loc[name, cj]
            rows.append({"j": cj, "k": ck, "own": j == k, "beta": b, "se": s, "t": b / s if s > 0 else np.nan,
                         "elasticity": b / res["mean_y"][cj],
                         "display": res["coef"].loc[f"disp_{k}", cj] if f"disp_{k}" in res["coef"].index else np.nan,
                         "display_se": res["se"].loc[f"disp_{k}", cj] if f"disp_{k}" in res["se"].index else np.nan})
    return pd.DataFrame(rows)
