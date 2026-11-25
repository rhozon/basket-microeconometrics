"""Baskets, categories and store-week marketing from The Complete Journey (84.51 degrees).

One row per shopping trip (basket): who shopped, where, which week, and whether each of the
studied categories was in the basket. The marketing each basket faced is measured in its store
and week on the *other* baskets, so a shopper's own purchases never enter the price they face:
the depth of the loyalty-card discount on the category (or, as a robustness check, the price
paid relative to each product's usual price), and from the promotions table the number of the
category's products on display and in the weekly mailer, observed whether or not anyone bought.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(os.environ.get("BMX_RAW", "C:/Users/rodri/data/completejourney"))
EXCLUDE = {"COUPON/MISC ITEMS", "GIFT & GOODS", "FUEL", "COUPON"}
EXTRA = ["PASTA SAUCE", "HOT DOGS", "PNT BTR/JELLY/JAMS", "DINNER MXS:DRY", "BACON", "FROZEN BREAD/DOUGH"]
# Weeks with national holidays that move demand and promotions together (2017 calendar).
HOLIDAY_WEEKS = {5: "super_bowl", 27: "july_4", 47: "thanksgiving", 52: "christmas", 1: "new_year", 21: "memorial_day", 36: "labor_day"}


def load(raw: Path = RAW) -> dict[str, pd.DataFrame]:
    tx = pd.read_csv(raw / "transactions.csv.gz")
    pr = pd.read_csv(raw / "products.csv.gz")
    promo = pd.read_csv(raw / "promotions_by_category.csv.gz")
    return {"tx": tx, "products": pr, "promo": promo}


def choose_categories(tx: pd.DataFrame, products: pd.DataFrame, k: int = 40) -> list[str]:
    t = tx.merge(products[["product_id", "product_category"]], on="product_id", how="left")
    c = t.groupby("product_category")["basket_id"].nunique().sort_values(ascending=False)
    c = c[~c.index.isin(EXCLUDE)]
    top = list(c.index[:k])
    return top + [e for e in EXTRA if e in c.index and e not in top]


def baskets(tx: pd.DataFrame, products: pd.DataFrame, cats: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Basket table and a 0/1 matrix of category presence, plus spend per category when bought."""
    t = tx.merge(products[["product_id", "product_category"]], on="product_id", how="left")
    b = t.groupby("basket_id").agg(household_id=("household_id", "first"), store_id=("store_id", "first"),
                                   week=("week", "first"), items=("quantity", "sum"),
                                   spend=("sales_value", "sum")).reset_index()
    inc = (t[t["product_category"].isin(cats)].groupby(["basket_id", "product_category"])["sales_value"].sum()
           .unstack(fill_value=0.0).reindex(index=b["basket_id"], columns=cats, fill_value=0.0))
    y = (inc > 0).astype(np.int8)
    y.index = b.index
    spend = inc.where(inc > 0).mean()  # mean spend on the category when it is bought
    return b, y, spend


def _leave_out(tx: pd.DataFrame, products: pd.DataFrame, cats: list[str]) -> pd.DataFrame:
    t = tx[(tx["quantity"] > 0) & (tx["sales_value"] > 0)].merge(
        products[["product_id", "product_category"]], on="product_id")
    t = t[t["product_category"].isin(cats)].copy()
    t["disc"] = t["retail_disc"].abs()
    t["shelf"] = t["sales_value"] + t["disc"]
    t["unit"] = t["sales_value"] / t["quantity"]
    t["usual"] = t.groupby("product_id")["unit"].transform("median")
    t["wq"] = t["unit"] / t["usual"] * t["quantity"]
    return t


def store_week_marketing(b: pd.DataFrame, tx: pd.DataFrame, products: pd.DataFrame, promo: pd.DataFrame,
                         cats: list[str], measure: str = "discount") -> pd.DataFrame:
    """Marketing each basket faced in its store and week, leaving the basket's own purchases out.

    measure="discount": loyalty-card discount as a share of shelf value of the category's sales
    (0 = nothing on discount); measure="price": log of the price paid relative to each product's
    usual price. Displays and mailers come from the promotions table.
    """
    t = _leave_out(tx, products, cats)
    num, den = ("disc", "shelf") if measure == "discount" else ("wq", "quantity")
    bc = t.groupby(["basket_id", "product_category"])[[num, den]].sum().reset_index()
    sw = t.groupby(["product_category", "store_id", "week"])[[num, den]].sum().reset_index()
    sw = sw.rename(columns={num: "N", den: "D"})
    X = pd.DataFrame(index=b.index)
    key = b[["basket_id", "store_id", "week"]]
    for j, c in enumerate(cats):
        k = key.merge(sw[sw["product_category"] == c], on=["store_id", "week"], how="left") \
               .merge(bc[bc["product_category"] == c][["basket_id", num, den]], on="basket_id", how="left") \
               .fillna({num: 0.0, den: 0.0})
        v = (k["N"] - k[num]) / (k["D"] - k[den]).replace(0, np.nan)
        if measure == "discount":
            v = v.clip(0, 0.9)
            X[f"lp_{j}"] = v.fillna(v.mean()).to_numpy()
        else:
            X[f"lp_{j}"] = np.log(v.fillna(1.0).clip(0.3, 2.0)).to_numpy()
        pm = promo[promo["product_category"] == c][["store_id", "week", "n_display", "n_mailer"]]
        m2 = key.merge(pm, on=["store_id", "week"], how="left").fillna({"n_display": 0, "n_mailer": 0})
        X[f"disp_{j}"] = np.log1p(m2["n_display"].to_numpy())
        X[f"mail_{j}"] = np.log1p(m2["n_mailer"].to_numpy())
    return X
