"""
Tests for src/model/dfm.py — DynamicFactorMQ wrapper.

Unit tests: use tiny synthetic panels (no API, no EM fitting).
Integration tests: fit the real model on a short window; marked @integration.
"""
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.model.dfm import extract_nowcast, prepare_dfm_data


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_monthly(n_months=36, n_series=3, start="2015-01-01", seed=0):
    """Synthetic monthly panel: small values around zero (like log_diffs)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_months, freq="MS")
    cols = [f"M{i}" for i in range(n_series)]
    return pd.DataFrame(rng.normal(0, 0.3, (n_months, n_series)), index=idx, columns=cols)


def _make_quarterly(n_quarters=12, start="2015-01-01", seed=1):
    """
    Synthetic quarterly GDP DataFrame with a proper quarterly DatetimeIndex
    (freq='QS'), as DynamicFactorMQ expects for endog_quarterly.
    The last 2 quarters are NaN to simulate unreleased data.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_quarters, freq="QS")
    values = list(rng.normal(0.5, 0.3, n_quarters - 2)) + [float("nan"), float("nan")]
    s = pd.Series(values, index=idx, name="GDPC1")
    return pd.DataFrame({"GDPC1": s})


# ── Unit tests: prepare_dfm_data ──────────────────────────────────────────────

def _fake_gdp_raw():
    """
    Return a synthetic raw GDPC1 Series with quarterly DatetimeIndex ending
    one quarter before today, so prepare_dfm_data will extend it to the current
    quarter (adding the NaN nowcast target row).
    """
    prev_qtr_end = (pd.Timestamp.today().to_period("Q") - 1).to_timestamp()
    idx = pd.date_range("2000-01-01", end=prev_qtr_end, freq="QS")
    return pd.Series(
        np.linspace(18_000, 20_000, len(idx)),
        index=idx,
        name="GDPC1",
    )


def _fake_monthly_panel(start="1999-11-01", n_months=30, n_cols=11):
    idx = pd.date_range(start, periods=n_months, freq="MS")
    cols = ["INDPRO", "PAYEMS", "UNRATE", "PCEC96", "RSAFS",
            "HOUST", "DGORDER", "DSPIC96", "CPIAUCSL",
            "GACDISA066MSFRBNY", "GACDFSA066MSFRBPHI"][:n_cols]
    data = np.random.default_rng(0).normal(0, 0.2, (n_months, n_cols))
    df = pd.DataFrame(data, index=idx, columns=cols)
    df.iloc[0, :] = np.nan   # first row is NaN from log_diff
    return df


class TestPrepareDFMData:

    @patch("src.data.fred_loader.fetch_series")
    @patch("src.model.dfm.build_panel")
    def test_splits_by_frequency(self, mock_build, mock_fetch):
        """
        prepare_dfm_data must return (monthly_df, quarterly_df) where:
          - quarterly_df contains only GDPC1 with a quarterly-frequency index
          - monthly_df contains all other series with a monthly-frequency index
        """
        mock_build.return_value = _fake_monthly_panel()  # no GDPC1 in monthly panel
        mock_fetch.return_value = _fake_gdp_raw()        # raw GDPC1 for GDP branch

        monthly_df, quarterly_df = prepare_dfm_data(
            start="2000-01-01", use_cache=False
        )

        assert "GDPC1" not in monthly_df.columns, \
            "GDPC1 (quarterly) should not appear in the monthly DataFrame"
        assert "GDPC1" in quarterly_df.columns, \
            "GDPC1 should appear in the quarterly DataFrame"

    @patch("src.data.fred_loader.fetch_series")
    @patch("src.model.dfm.build_panel")
    def test_no_all_nan_rows_in_monthly(self, mock_build, mock_fetch):
        """
        After prepare_dfm_data, no row of the monthly DataFrame should be
        entirely NaN — the first log-diff row (which is all-NaN) is dropped.
        """
        mock_build.return_value = _fake_monthly_panel()
        mock_fetch.return_value = _fake_gdp_raw()

        monthly_df, _ = prepare_dfm_data(start="2000-01-01", use_cache=False)

        all_nan_rows = monthly_df.isna().all(axis=1).sum()
        assert all_nan_rows == 0, (
            f"{all_nan_rows} fully-NaN rows remain in monthly_df; "
            f"the leading all-NaN row from log_diff should have been dropped."
        )

    @patch("src.data.fred_loader.fetch_series")
    @patch("src.model.dfm.build_panel")
    def test_different_frequencies(self, mock_build, mock_fetch):
        """
        monthly_df must have monthly freq ('MS') and quarterly_df must have
        quarterly freq (starting with 'Q'). DynamicFactorMQ uses these
        separate grids — they should NOT share the same index.
        """
        mock_build.return_value = _fake_monthly_panel()
        mock_fetch.return_value = _fake_gdp_raw()

        monthly_df, quarterly_df = prepare_dfm_data(
            start="2000-01-01", use_cache=False
        )

        monthly_freq = monthly_df.index.freqstr
        quarterly_freq = quarterly_df.index.freqstr
        assert monthly_freq == "MS", (
            f"monthly_df should have freq='MS', got '{monthly_freq}'"
        )
        assert quarterly_freq.startswith("Q"), (
            f"quarterly_df should have freq starting with 'Q', got '{quarterly_freq}'"
        )


# ── Unit tests: extract_nowcast ────────────────────────────────────────────────

def _synthetic_nowcast_problem(seed: int = 3):
    """
    A small mixed-frequency dataset with a KNOWN answer, built so that the
    three historical extraction mistakes each give a visibly wrong number.

      * Latent monthly factor f_t (AR(1)), forced to −2.5 in Jul–Sep 2019 and
        +2.5 in Oct–Dec 2019, so Q3 2019 is weak and Q4 2019 (the target) is
        strong. Reading a cell dominated by Q3 shows up as a big miss.
      * Quarterly GDP = 2.0 + MM(f) in percent, where MM is the Mariano–
        Murasawa 1-2-3-2-1 aggregation ending at the quarter's LAST month.
        Mean 2.0 means a z-score (≈ 0) is far from the right answer.
      * Four monthly indicators load on f. Q4 2019 GDP is masked (NaN).

    Returns (monthly_df, quarterly_df, truth_q4, truth_q3).
    """
    rng = np.random.default_rng(seed)
    midx = pd.date_range("2005-01-01", "2019-12-01", freq="MS")
    f = np.zeros(len(midx))
    for t in range(1, len(f)):
        f[t] = 0.7 * f[t - 1] + rng.normal()
    f[-6:-3], f[-3:] = -2.5, 2.5

    loadings = [1.0, 0.8, -0.6, 0.9]
    monthly = pd.DataFrame(
        {f"M{i}": lam * f + 0.3 * rng.normal(size=len(f)) for i, lam in enumerate(loadings)},
        index=midx,
    )

    w = np.array([1, 2, 3, 2, 1]) / 9.0
    mm = pd.Series(np.convolve(f, w, mode="full")[: len(f)], index=midx)
    q_end_months = midx[midx.month % 3 == 0]
    gdp = 2.0 + mm.loc[q_end_months] + 0.05 * rng.normal(size=len(q_end_months))
    gdp.index = q_end_months - pd.DateOffset(months=2)        # quarter-START dates
    quarterly = pd.DataFrame({"GDPC1": gdp.iloc[1:]}).asfreq("QS")
    truth_q4, truth_q3 = float(quarterly["GDPC1"].iloc[-1]), float(quarterly["GDPC1"].iloc[-2])
    quarterly.iloc[-1, 0] = np.nan
    return monthly, quarterly, truth_q4, truth_q3


@pytest.fixture(scope="module")
def synthetic_fit():
    """Fit a real 1-factor DynamicFactorMQ once for all extraction tests (~2 s)."""
    import warnings
    from src.model.dfm import fit_dfm

    monthly, quarterly, truth_q4, truth_q3 = _synthetic_nowcast_problem()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # EM maxiter warning is fine here
        results = fit_dfm(monthly, quarterly, n_factors=1, maxiter=200)
    return results, monthly, quarterly, truth_q4, truth_q3


class TestExtractNowcastRealFit:
    """
    Regression tests for the 2026-10 alignment bug, using a REAL fitted model.

    The old tests mocked `fittedvalues` on a quarterly index, which hid three
    problems: statsmodels' fittedvalues are monthly-indexed, in standardized
    units (z-scores), and one-step-ahead; and the quarter-start month holds a
    rolling window that is mostly the previous quarter.
    """

    def test_impact_month_is_quarter_end(self, synthetic_fit):
        """The target month is December 2019 (quarter end), not October."""
        results, _, quarterly, *_ = synthetic_fit
        out = extract_nowcast(results, quarterly)
        assert out["quarter"] == "2019 Q4"
        assert out["date"] == pd.Timestamp("2019-10-01")
        assert out["impact_month"] == pd.Period("2019-12", freq="M")

    def test_value_is_smoothed_prediction_at_quarter_end(self, synthetic_fit):
        """The nowcast is the original-scale smoothed estimate at Dec 2019."""
        results, _, quarterly, *_ = synthetic_fit
        out = extract_nowcast(results, quarterly)
        expected = results.get_prediction(
            start="2019-12", end="2019-12", information_set="smoothed",
        ).predicted_mean["GDPC1"].iloc[0]
        assert out["nowcast_qtr_pct"] == pytest.approx(float(expected), abs=1e-12)

    def test_tracks_current_quarter_in_percent_units(self, synthetic_fit):
        """
        The nowcast must be close to the TRUE Q4 value (strong quarter, ~2.8%)
        — not to Q3 (weak), and not a z-score near zero.
        """
        results, _, quarterly, truth_q4, truth_q3 = synthetic_fit
        nowcast = extract_nowcast(results, quarterly)["nowcast_qtr_pct"]
        assert abs(nowcast - truth_q4) < 0.5, f"nowcast {nowcast:.2f} vs truth {truth_q4:.2f}"
        assert abs(nowcast - truth_q4) < abs(nowcast - truth_q3)

    def test_old_extraction_would_fail_this_problem(self, synthetic_fit):
        """
        Guard on the test design itself: the pre-fix reading (fittedvalues at
        the quarter-START month) must be far from the truth here, or the test
        above would not be able to catch a regression.
        """
        results, _, quarterly, truth_q4, _ = synthetic_fit
        old = float(results.fittedvalues["GDPC1"].loc[pd.Period("2019-10", freq="M")])
        assert abs(old - truth_q4) > 1.5, f"old reading {old:.2f} is too close to {truth_q4:.2f}"

    def test_annualized_is_4x_quarterly(self, synthetic_fit):
        results, _, quarterly, *_ = synthetic_fit
        out = extract_nowcast(results, quarterly)
        assert out["nowcast_ann_pct"] == pytest.approx(4 * out["nowcast_qtr_pct"], abs=1e-12)

    def test_raises_when_target_month_outside_model(self, synthetic_fit):
        """No silent fallback: a target the model never saw is an error."""
        results, _, quarterly, *_ = synthetic_fit
        extra = pd.DataFrame({"GDPC1": [np.nan]}, index=pd.DatetimeIndex(["2020-01-01"]))
        wrong = pd.concat([quarterly, extra]).asfreq("QS")
        with pytest.raises(ValueError, match="not in the model's index"):
            extract_nowcast(results, wrong)


class TestExtractNowcastNoFit:

    def test_returns_none_when_no_missing(self):
        """If all GDP quarters have actual data, there is nothing to nowcast."""
        idx = pd.date_range("2020-01-01", periods=8, freq="QS")
        quarterly_df = pd.DataFrame({"GDPC1": pd.Series(0.5, index=idx)})
        assert extract_nowcast(object(), quarterly_df) is None

    def test_quarter_end_month_handles_datetime_index(self):
        """quarter_end_month() returns the matching key for a DatetimeIndex model."""
        from types import SimpleNamespace
        from src.model.dfm import quarter_end_month

        idx = pd.date_range("2019-01-01", "2019-12-01", freq="MS")
        fake = SimpleNamespace(model=SimpleNamespace(_index=idx))
        assert quarter_end_month(fake, pd.Timestamp("2019-10-01")) == pd.Timestamp("2019-12-01")
        assert quarter_end_month(fake, pd.Timestamp("2019-01-01")) == pd.Timestamp("2019-03-01")


# ── Integration tests (require FRED API + model fit) ─────────────────────────

@pytest.mark.integration
class TestDFMIntegration:
    """
    Fits a single-factor DFM on a 5-year window and checks convergence and
    nowcast plausibility. This takes ~30–90 seconds.
    """

    @pytest.fixture(autouse=True)
    def require_api_key(self):
        if not os.getenv("FRED_API_KEY"):
            pytest.skip("FRED_API_KEY not set")

    def test_dfm_fits_and_converges(self):
        """
        The EM algorithm should produce a finite log-likelihood and
        the factor estimates should be a well-shaped array.

        We use a short window (2015–2020) to keep runtime under ~60s.
        """
        from src.model.dfm import fit_dfm

        # Monthly and quarterly DataFrames are on SEPARATE grids — do NOT
        # align them to the same index. DynamicFactorMQ handles alignment.
        monthly_df   = _make_monthly(n_months=72, n_series=4, start="2015-01-01")
        quarterly_df = _make_quarterly(n_quarters=24, start="2015-01-01")
        monthly_df.columns = ["INDPRO", "PAYEMS", "UNRATE", "PCEC96"]

        results = fit_dfm(monthly_df, quarterly_df, n_factors=1, maxiter=200)

        assert np.isfinite(results.llf), \
            f"Log-likelihood is not finite: {results.llf}"
        factors = results.factors.smoothed
        assert factors.shape[0] == len(monthly_df), \
            f"Factor array has {factors.shape[0]} rows, expected {len(monthly_df)}"

    def test_real_vintage_nowcast_2019q4(self):
        """
        Real ALFRED vintage as of 2019-12-31, blocked 2-factor model. Q4 2019
        GDP was first published at +2.1% annualized. The pre-fix extraction
        reported −0.35 here (a z-score ×4 from the October cell); a correct
        nowcast for a calm quarter should land in a normal growth range.
        """
        from src.backtest.vintage import build_vintage_dfm_data
        from src.model.dfm import fit_dfm, make_block_structure

        m, q = build_vintage_dfm_data(as_of_date="2019-12-31", start="2000-01-01")
        res = fit_dfm(m, q, factor_blocks=make_block_structure(m, q), maxiter=500)
        out = extract_nowcast(res, q)
        assert out["quarter"] == "2019 Q4"
        assert 0.5 < out["nowcast_ann_pct"] < 4.5, out

    def test_nowcast_in_plausible_range(self):
        """
        Run the full pipeline on recent real data and verify the annualized
        nowcast is in a plausible range for the US economy.

        We use a short estimation window (2018–present) for speed. The nowcast
        should be between −20% and +15% annualized — this range would catch
        catastrophic failures while accommodating historical outliers (e.g.,
        COVID-19 Q2 2020 was −31%).
        """
        monthly_df, quarterly_df = prepare_dfm_data(
            start="2018-01-01", use_cache=True
        )

        from src.model.dfm import fit_dfm

        results  = fit_dfm(monthly_df, quarterly_df, n_factors=1, maxiter=300)
        nowcast  = extract_nowcast(results, quarterly_df)

        assert nowcast is not None, "No nowcast produced — check for missing GDP quarters"
        assert -20.0 < nowcast["nowcast_ann_pct"] < 15.0, (
            f"Annualized nowcast {nowcast['nowcast_ann_pct']:.2f}% is outside "
            f"the plausible range (−20%, +15%). This may indicate a data or "
            f"model setup error."
        )


class TestExtractNowcastTargetAndSE:

    def test_se_matches_smoother(self, synthetic_fit):
        results, _, quarterly, *_ = synthetic_fit
        out = extract_nowcast(results, quarterly)
        se = results.get_prediction(start="2019-12", end="2019-12",
                                    information_set="smoothed").se_mean["GDPC1"].iloc[0]
        assert out["nowcast_se_ann"] == pytest.approx(4 * float(se))
        assert out["nowcast_se_ann"] > 0

    def test_explicit_target_must_be_unreleased(self, synthetic_fit):
        results, _, quarterly, *_ = synthetic_fit
        with pytest.raises(ValueError, match="unreleased"):
            extract_nowcast(results, quarterly, target_quarter="2019-07-01")

    def test_explicit_earlier_target_when_two_quarters_unreleased(self):
        """Q3 and Q4 2019 both unpublished: target Q3 reads September, not December."""
        import warnings
        from src.model.dfm import fit_dfm
        monthly, quarterly, _, truth_q3 = _synthetic_nowcast_problem()
        quarterly = quarterly.copy()
        quarterly.loc["2019-07-01", "GDPC1"] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = fit_dfm(monthly, quarterly, n_factors=1, maxiter=200)
        out = extract_nowcast(res, quarterly, target_quarter="2019-07-01")
        assert out["quarter"] == "2019 Q3"
        assert out["impact_month"] == pd.Period("2019-09", freq="M")
        assert abs(out["nowcast_qtr_pct"] - truth_q3) < 0.5
        assert extract_nowcast(res, quarterly)["quarter"] == "2019 Q4"     # default unchanged
