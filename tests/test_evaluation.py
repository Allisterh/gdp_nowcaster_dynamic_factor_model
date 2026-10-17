"""
Tests for src/backtest/evaluation.py, the new baselines in src/model/baseline.py,
and the first-release GDP construction in src/backtest/vintage.py.
All unit tests: synthetic data, no API calls.
"""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.backtest.evaluation import (
    accuracy_table,
    block_bootstrap_rmse_diff,
    comparison_tests,
    dm_test,
    rmse,
)
from src.model.baseline import fit_ar1, fit_ar1_robust, mean_nowcast


class TestDieboldMariano:

    def test_identical_errors_give_no_evidence(self):
        e = pd.Series(np.random.default_rng(0).normal(size=40))
        out = dm_test(e, e)
        assert out["pvalue"] == 1.0 and out["stat"] == 0.0

    def test_detects_clearly_better_model(self):
        rng = np.random.default_rng(1)
        good, bad = rng.normal(0, 1, 60), rng.normal(0, 3, 60)
        out = dm_test(good, bad)
        assert out["stat"] < 0, "negative statistic = model has smaller loss"
        assert out["pvalue"] < 0.01

    def test_antisymmetric(self):
        rng = np.random.default_rng(2)
        a, b = rng.normal(size=40), rng.normal(size=40) * 1.3
        assert dm_test(a, b)["stat"] == pytest.approx(-dm_test(b, a)["stat"])
        assert dm_test(a, b)["pvalue"] == pytest.approx(dm_test(b, a)["pvalue"])

    def test_hln_matches_hand_calculation(self):
        """h = 1: DM = mean(d)/sqrt(var_pop(d)/n), scaled by sqrt((n−1)/n), t(n−1)."""
        a = np.array([1.0, -2.0, 0.5, 1.5, -1.0, 0.2])
        b = np.array([2.0, -1.0, 1.0, 3.0, -2.5, 1.0])
        d = a ** 2 - b ** 2
        n = len(d)
        expected = d.mean() / np.sqrt(d.var() / n) * np.sqrt((n - 1) / n)
        assert dm_test(a, b)["stat"] == pytest.approx(expected)

    def test_drops_unpaired_nans(self):
        a = pd.Series([1.0, np.nan, 0.5, 1.5, -1.0])
        b = pd.Series([2.0, 1.0, np.nan, 3.0, -2.5])
        assert dm_test(a, b)["n"] == 3


class TestBlockBootstrap:

    def test_point_estimate_is_rmse_difference(self):
        rng = np.random.default_rng(3)
        a, b = rng.normal(size=40), rng.normal(size=40) * 2
        out = block_bootstrap_rmse_diff(a, b, n_boot=500)
        assert out["diff"] == pytest.approx(rmse(a) - rmse(b))
        assert out["lo"] <= out["diff"] <= out["hi"]

    def test_interval_excludes_zero_for_large_gap(self):
        rng = np.random.default_rng(4)
        out = block_bootstrap_rmse_diff(rng.normal(0, 1, 80), rng.normal(0, 4, 80), n_boot=2000)
        assert out["hi"] < 0

    def test_reproducible_with_seed(self):
        rng = np.random.default_rng(5)
        a, b = rng.normal(size=30), rng.normal(size=30)
        assert block_bootstrap_rmse_diff(a, b, seed=7) == block_bootstrap_rmse_diff(a, b, seed=7)


class TestTables:

    @staticmethod
    def _df():
        rng = np.random.default_rng(6)
        q = pd.date_range("2015-01-01", periods=44, freq="QS")
        actual = rng.normal(2.5, 1.5, 44)
        return pd.DataFrame({
            "target_quarter":         q,
            "is_2020":                q.year == 2020,
            "actual_ann":             actual,
            "first_release_ann":      actual + rng.normal(0, 0.5, 44),
            "dfm_nowcast_ann":        actual + rng.normal(0, 1.0, 44),
            "ar1_nowcast_ann":        np.full(44, 2.5),
            "ar1_robust_nowcast_ann": np.full(44, 2.4),
            "mean_nowcast_ann":       np.full(44, 2.3),
        })

    def test_accuracy_table_covers_all_cells(self):
        tab = accuracy_table(self._df())
        assert len(tab) == 2 * 3 * 4                      # targets × samples × models
        ex = tab[(tab["sample"] == "ex-2020") & (tab["target"] == "current vintage")]
        assert set(ex["n"]) == {40}

    def test_bias_sign_convention(self):
        df = self._df()
        df["dfm_nowcast_ann"] = df["actual_ann"] + 1.0     # always 1 pp too high
        tab = accuracy_table(df)
        row = tab[(tab.model == "DFM") & (tab.target == "current vintage") & (tab["sample"] == "full")]
        assert float(row["bias"].iloc[0]) == pytest.approx(1.0)

    def test_comparison_tests_shape(self):
        tests = comparison_tests(self._df())
        assert len(tests) == 2 * 2 * 3                    # targets × samples × benchmarks
        assert (tests["ci_lo"] <= tests["ci_hi"]).all()


class TestNewBaselines:

    def test_mean_nowcast(self):
        s = pd.Series([0.5, 1.0, np.nan, 0.0])
        out = mean_nowcast(s)
        assert out["nowcast_qtr_pct"] == pytest.approx(0.5)
        assert out["nowcast_ann_pct"] == pytest.approx(2.0)

    @staticmethod
    def _calm_ar1(n=95, beta=0.2, seed=7):
        rng = np.random.default_rng(seed)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 + beta * (y[t - 1] - 0.5) + rng.normal(0, 0.5)
        return y

    def test_robust_ar1_resists_a_crash(self):
        """
        Calm AR(1) (β = 0.2) ending in a −1.3% then −9% quarter — what the
        2020 Q3 vintage looked like. OLS is dragged toward β ≈ 1 (forecast:
        another crash); Huber stays near the true β.
        """
        s = pd.Series(np.r_[self._calm_ar1(), -1.3, -9.0])
        assert fit_ar1(s).params["gdp_lag1"] > 0.6
        assert abs(fit_ar1_robust(s).params["gdp_lag1"] - 0.2) < 0.15

    def test_robust_ar1_better_but_not_immune_after_rebound(self):
        """
        Add the +7.5% rebound (2020 Q4 vintage). The (−9 → +7.5) pair is a
        high-LEVERAGE point; Huber down-weights large residuals, not leverage,
        so it is pulled too — just less than OLS.
        """
        s = pd.Series(np.r_[self._calm_ar1(), -1.3, -9.0, 7.5])
        b_ols = fit_ar1(s).params["gdp_lag1"]
        b_rob = fit_ar1_robust(s).params["gdp_lag1"]
        assert abs(b_rob - 0.2) < abs(b_ols - 0.2)


class TestFirstRelease:

    def test_growth_uses_levels_from_the_same_vintage(self, tmp_path):
        """
        Q2's first release (Jul 30) must be divided by Q1 *as published on
        Jul 30* (after a benchmark revision on Jul 1), not Q1's first print.
        """
        from src.backtest import vintage

        releases = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-04-01"]),
            "realtime_start": pd.to_datetime(["2020-04-29", "2020-07-01", "2020-07-30"]),
            "value": [100.0, 110.0, 111.0],
        })
        with patch.object(vintage, "ALFRED_CACHE_DIR", tmp_path):
            releases.to_parquet(tmp_path / "GDPC1_all_releases.parquet")
            out = vintage.fetch_gdp_first_release([pd.Timestamp("2020-04-01")])

        assert out.loc["2020-04-01", "first_release_date"] == pd.Timestamp("2020-07-30")
        assert out.loc["2020-04-01", "first_release_ann"] == pytest.approx(400 * np.log(111 / 110))
