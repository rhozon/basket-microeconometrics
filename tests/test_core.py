import numpy as np
import pandas as pd
import pytest

from bmx import demand, mba, rec
from bmx.cli import bh


def test_lift_of_independent_categories_is_one():
    rng = np.random.default_rng(0)
    y = pd.DataFrame({"a": rng.random(200_000) < 0.3, "b": rng.random(200_000) < 0.2}).astype(int)
    t = mba.pair_table(y)
    assert t["lift"].iloc[0] == pytest.approx(1.0, abs=0.03)


def test_trip_size_alone_creates_lift_that_the_size_adjustment_removes():
    rng = np.random.default_rng(1)
    n = 200_000
    size = rng.integers(1, 40, n)
    p = np.clip(size / 50, 0, 1)
    y = pd.DataFrame({"a": rng.random(n) < p, "b": rng.random(n) < p}).astype(int)
    raw = mba.pair_table(y)["lift"].iloc[0]
    adj = mba.lift_by_size(y, pd.Series(size), bins=39)["lift_size_adj"].iloc[0]
    assert raw > 1.2
    assert adj == pytest.approx(1.0, abs=0.03)


def test_household_taste_correlation_is_removed():
    rng = np.random.default_rng(2)
    hh = np.repeat(np.arange(500), 100)
    taste = rng.random(500)[hh]          # same taste drives both categories
    y = pd.DataFrame({"a": rng.random(len(hh)) < taste, "b": rng.random(len(hh)) < taste}).astype(int)
    size = pd.Series(np.ones(len(hh)))
    r = mba.residual_correlation(y, pd.Series(hh), size, bins=1)["resid_corr"].iloc[0]
    assert abs(r) < 0.02
    assert np.corrcoef(y["a"], y["b"])[0, 1] > 0.2


def test_fit_recovers_a_cross_effect_with_household_effects():
    rng = np.random.default_rng(3)
    n_h, per = 400, 60
    hh = np.repeat(np.arange(n_h), per)
    week = np.tile(np.arange(1, per + 1) % 53 + 1, n_h)
    x = rng.random(len(hh)) * 0.3
    base = rng.random(n_h)[hh] * 0.4
    y = pd.DataFrame({"a": (rng.random(len(hh)) < base + 0.5 * x).astype(int),
                      "b": (rng.random(len(hh)) < 0.2).astype(int)})
    X = pd.DataFrame({"lp_0": x, "lp_1": rng.random(len(hh)) * 0.3})
    b = pd.DataFrame({"household_id": hh, "week": week})
    res = demand.fit(y, X, b)
    assert res["coef"].loc["lp_0", "a"] == pytest.approx(0.5, abs=0.06)
    assert abs(res["coef"].loc["lp_1", "a"]) < 3 * res["se"].loc["lp_1", "a"] + 0.02


def test_auc_matches_known_values():
    assert rec.auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert rec.auc(np.array([0, 1, 0, 1]), np.array([1.0, 1.0, 1.0, 1.0])) == 0.5


def test_benjamini_hochberg():
    p = np.array([0.001, 0.009, 0.04, 0.2, 0.5])
    assert bh(p, 0.05).tolist() == [True, True, False, False, False]
