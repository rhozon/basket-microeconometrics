"""Command line.

    Rscript r/fetch.R $BMX_RAW   # download The Complete Journey (completejourney package, CC0)
    python -m bmx.cli analyse      # tables in results/
    python -m bmx.cli export F     # JSON for the interactive page
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import data, demand, mba, rec

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
KEY_PAIRS = [("DRY NOODLES/PASTA", "PASTA SAUCE"), ("BAG SNACKS", "SOFT DRINKS"), ("HOT DOGS", "BAKED BREAD/BUNS/ROLLS"),
             ("COLD CEREAL", "FLUID MILK PRODUCTS"), ("PNT BTR/JELLY/JAMS", "BAKED BREAD/BUNS/ROLLS"),
             ("TOMATOES", "ONIONS"), ("BEERS/ALES", "BAG SNACKS")]


def bh(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg: which p-values are discoveries at false discovery rate q."""
    o = np.argsort(p)
    n = len(p)
    ok = p[o] <= q * np.arange(1, n + 1) / n
    k = ok.nonzero()[0].max() + 1 if ok.any() else 0
    out = np.zeros(n, bool)
    out[o[:k]] = True
    return out


def effects(res: dict, cats: list[str], var: str = "lp") -> pd.DataFrame:
    co, se = res["coef"], res["se"]
    rows = []
    for k, ck in enumerate(cats):
        name = f"{var}_{k}"
        if name not in co.index:
            continue
        for j, cj in enumerate(cats):
            b, s = co.loc[name, cj], se.loc[name, cj]
            rows.append({"k": ck, "j": cj, "own": j == k, "var": var, "coef": b, "se": s, "t": b / s,
                         "p": 2 * (1 - norm.cdf(abs(b / s))), "base_rate": float(res["mean_y"][cj])})
    e = pd.DataFrame(rows)
    for own in (True, False):
        m = e["own"] == own
        e.loc[m, "bh_discovery"] = bh(e.loc[m, "p"].to_numpy())
    return e


def cmd_analyse() -> dict:
    RESULTS.mkdir(exist_ok=True)
    d = data.load()
    cats = data.choose_categories(d["tx"], d["products"])
    b, y, spend = data.baskets(d["tx"], d["products"], cats)
    summary = {"households": int(b["household_id"].nunique()), "baskets": int(len(b)), "stores": int(b["store_id"].nunique()),
               "transactions": int(len(d["tx"])), "categories": len(cats), "weeks": int(b["week"].max())}

    pairs = mba.pair_table(y).merge(mba.lift_by_size(y, b["items"]), on=["j", "k"]) \
        .merge(mba.residual_correlation(y, b["household_id"], b["items"]), on=["j", "k"])
    pairs.to_csv(RESULTS / "pairs.csv", index=False)
    top_raw = set(map(tuple, pairs.nlargest(50, "lift")[["j", "k"]].to_numpy()))
    top_res = set(map(tuple, pairs.nlargest(50, "resid_corr")[["j", "k"]].to_numpy()))
    summary.update({"mean_lift": float(pairs["lift"].mean()), "mean_lift_size_adj": float(pairs["lift_size_adj"].mean()),
                    "share_lift_above_1": float((pairs["lift"] > 1).mean()),
                    "share_size_adj_above_1": float((pairs["lift_size_adj"] > 1).mean()),
                    "top50_overlap_raw_vs_taste_adjusted": len(top_raw & top_res),
                    "spearman_lift_vs_resid": float(pairs["lift"].rank().corr(pairs["resid_corr"].rank()))})

    out = []
    for measure in ("discount", "price"):
        X = data.store_week_marketing(b, d["tx"], d["products"], d["promo"], cats, measure)
        res = demand.fit(y, X, b)
        for var in ("lp", "disp", "mail"):
            e = effects(res, cats, var)
            e["measure"] = measure
            out.append(e)
        summary[f"n_obs_{measure}"] = res["n"]
    eff = pd.concat(out)
    eff["var"] = eff["var"].replace({"lp": "discount depth"}).where(eff["measure"] == "discount",
                                                                    eff["var"].replace({"lp": "log relative price"}))
    eff.to_csv(RESULTS / "effects.csv", index=False)
    main = eff[(eff["measure"] == "discount")]
    for var in main["var"].unique():
        m = main[main["var"] == var]
        summary[f"{var}: own discoveries"] = int(m.loc[m["own"], "bh_discovery"].sum())
        summary[f"{var}: own positive at 5%"] = int(((m["own"]) & (m["t"] > 1.96)).sum())
        summary[f"{var}: cross discoveries"] = int(m.loc[~m["own"], "bh_discovery"].sum())
        summary[f"{var}: cross pairs"] = int((~m["own"]).sum())
    kp = main[main["var"] == "discount depth"]
    kp = kp[kp.apply(lambda r: (r["j"], r["k"]) in KEY_PAIRS or (r["k"], r["j"]) in KEY_PAIRS, axis=1)].copy()
    # effect of a 10-point deeper discount on k, in percentage points of j's purchase probability
    kp["effect_10pt_pp"] = kp["coef"] * 0.10 * 100
    kp["ci_lo_pp"], kp["ci_hi_pp"] = (kp["coef"] - 1.96 * kp["se"]) * 10, (kp["coef"] + 1.96 * kp["se"]) * 10
    kp["mde80_pp"] = 2.8 * kp["se"] * 10
    kp.merge(pairs[["j", "k", "lift"]].rename(columns={"j": "k", "k": "j"}), on=["j", "k"], how="left") \
      .to_csv(RESULTS / "key_pairs.csv", index=False)

    r = rec.evaluate(y, b)
    r.to_csv(RESULTS / "recommender.csv", index=False)
    pd.DataFrame({"category": cats, "base_rate": y.mean().to_numpy(), "spend_when_bought": spend.to_numpy()}) \
      .to_csv(RESULTS / "categories.csv", index=False)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _records(df: pd.DataFrame) -> list[dict]:
    df = df.round(6)
    return df.astype(object).where(df.notna(), None).to_dict("records")


def cmd_export(path: str) -> None:
    eff = pd.read_csv(RESULTS / "effects.csv")
    eff = eff[(eff["measure"] == "discount") & (eff["var"] == "discount depth")]
    out = {"summary": json.loads((RESULTS / "summary.json").read_text(encoding="utf-8")),
           "categories": _records(pd.read_csv(RESULTS / "categories.csv")),
           "pairs": _records(pd.read_csv(RESULTS / "pairs.csv")[["j", "k", "support", "lift", "lift_size_adj", "resid_corr"]]),
           "effects": _records(eff[["k", "j", "own", "coef", "se", "bh_discovery"]]),
           "key_pairs": _records(pd.read_csv(RESULTS / "key_pairs.csv")),
           "recommender": _records(pd.read_csv(RESULTS / "recommender.csv"))}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"), ensure_ascii=False)
    print("wrote", path)


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    cmd = argv[0] if argv else "analyse"
    if cmd == "analyse":
        print(json.dumps(cmd_analyse(), indent=2))
    elif cmd == "export":
        cmd_export(argv[1])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
