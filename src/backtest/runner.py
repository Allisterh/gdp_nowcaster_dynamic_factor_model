"""
runner.py — Quarterly real-time vintage backtest for GDP nowcasting.

What this module does, in plain English:
  We simulate what our DFM model would have predicted in real-time for each
  past quarter. For Q1 2020, for example:

    1. Evaluation date = March 31, 2020 (January and February data are in,
       March only for the regional surveys; the advance GDP release is still
       ~30 days away).
    2. Fetch ALFRED vintage data as of March 31, 2020 — the same dataset
       the model would have actually seen on that date.
    3. Fit the blocked DFM on that vintage dataset.
    4. Extract the nowcast for Q1 2020 (the NaN quarter in the panel).
    5. Compare to actual Q1 2020 GDP growth (current-vintage, i.e., the
       best available estimate of economic truth).

  Repeat for every quarter from start_qs to end_qs, then compute RMSE.

Why two RMSE numbers (with and without 2020):
  Q2 2020 GDP fell at a −31.4% annualized rate — an unprecedented pandemic
  shock. Our 11-indicator DFM underestimates this badly, as does every small-scale
  DFM (even the 127-variable Fulton model over-corrects to −37%). The COVID
  recession was a supply shock outside the model's identifying assumptions
  (normal business-cycle factor dynamics). Including it inflates aggregate RMSE
  dramatically and obscures the model's genuine performance in normal times.

  We report both numbers, following the standard in the academic forecasting
  literature: a headline RMSE, plus the same stat excluding 2020. This is not
  cherry-picking — it is transparency about what the model is and isn't for.

LOOK-AHEAD GUARANTEE:
  For every backtest point, assert_no_lookahead() verifies that evaluation_date
  is before the GDP advance release date. This is a hard assertion, not a
  comment — the code will raise if violated.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest.vintage import (
    assert_before_release,
    assert_no_lookahead,
    build_vintage_dfm_data,
    fetch_gdp_first_release,
)
from src.data.fred_loader import fetch_series, transform_series
from src.model.baseline import ar1_nowcast, fit_ar1, fit_ar1_robust, mean_nowcast
from src.model.dfm import extract_nowcast, fit_dfm, make_block_structure

logger = logging.getLogger(__name__)

BACKTEST_RESULTS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "backtest_results.csv"
)


def backtest_quarter_range(
    start_qs: str = "2015-01-01",
    end_qs:   str = "2025-10-01",
) -> list[tuple[pd.Timestamp, str]]:
    """
    Generate (target_quarter_start, evaluation_date) pairs for the backtest.

    Evaluation date convention:
      evaluation_date = last calendar day of the target quarter.
      At this point:
        ✓ Months 1 and 2 of the quarter are published for nearly every
          series; month 3 only for the regional Fed surveys (monthly data
          comes out 2–6 weeks after month end).
        ✗ GDP advance release has NOT happened (~28 days after quarter end).
      This is the latest date that is guaranteed to have zero GDP look-ahead
      for every quarter (see scripts/diagnose_information_timing.py for what
      the missing third month costs).

    Parameters
    ----------
    start_qs : str
        First target quarter start date, e.g., "2015-01-01" for Q1 2015.
    end_qs : str
        Last target quarter start date.

    Returns
    -------
    list of (target_quarter_start: Timestamp, evaluation_date: str)
    """
    quarters = pd.date_range(start=start_qs, end=end_qs, freq="QS")
    result = []
    for q in quarters:
        qtr_end   = q + pd.DateOffset(months=3) - pd.DateOffset(days=1)
        eval_date = qtr_end.strftime("%Y-%m-%d")
        result.append((q, eval_date))
    return result


def horizon_evaluation_dates(
    quarter_start: pd.Timestamp,
    release_date: pd.Timestamp,
    step_days: int = 14,
) -> list[pd.Timestamp]:
    """
    Nowcast dates for the through-the-quarter backtest: every `step_days` from
    the quarter's first day, plus its last day (links to the quarter-end
    backtest), all strictly before the quarter's first GDP release.
    """
    q_end = quarter_start + pd.offsets.QuarterEnd(0)
    dates = set(pd.date_range(quarter_start, release_date - pd.Timedelta(days=1),
                              freq=f"{step_days}D"))
    if q_end < release_date:
        dates.add(q_end)
    return sorted(d for d in dates if d < release_date)


def _baseline_nowcasts_vintage(
    quarterly_df: pd.DataFrame,
    target_quarter: pd.Timestamp | None = None,
) -> dict[str, float]:
    """
    Real-time benchmark nowcasts from the vintage quarterly GDP panel.

    All three use only GDP growth published by the evaluation date, so they
    face the same information constraint as the DFM:
      ar1_nowcast_qtr        : OLS AR(1) (the original baseline)
      ar1_robust_nowcast_qtr : Huber-robust AR(1) — not dragged to beta ≈ 1 by
                               the 2020 crash-and-rebound pair
      mean_nowcast_qtr       : expanding mean of growth since the sample start

    If target_quarter is h > 1 quarters after the last published GDP (early in
    a quarter, before the previous quarter's release), the AR(1)s iterate h
    steps: y_{t+h} = a + b·y_{t+h-1}. Default target: the next quarter.

    Returns quarterly % (not annualized); NaN if fewer than 4 quarters exist.
    """
    gdp = quarterly_df["GDPC1"].dropna()
    if len(gdp) < 4:
        return {k: float("nan") for k in
                ("ar1_nowcast_qtr", "ar1_robust_nowcast_qtr", "mean_nowcast_qtr")}
    last_q = pd.Timestamp(gdp.index[-1])
    h = 1
    if target_quarter is not None:
        tq = pd.Timestamp(target_quarter).to_period("Q")
        h = tq.ordinal - last_q.to_period("Q").ordinal
        if h < 1:
            raise ValueError(f"Target {tq} is already published in this vintage.")

    def iterate(fit) -> float:
        y = float(gdp.iloc[-1])
        for _ in range(h):
            y = float(ar1_nowcast(fit, y)["nowcast_qtr_pct"])
        return y

    return {
        "ar1_nowcast_qtr":        iterate(fit_ar1(gdp)),
        "ar1_robust_nowcast_qtr": iterate(fit_ar1_robust(gdp)),
        "mean_nowcast_qtr":       mean_nowcast(gdp)["nowcast_qtr_pct"],
    }


def run_single_quarter(
    target_quarter:  pd.Timestamp,
    evaluation_date: str,
    factor_blocks:   dict | None = None,
    n_factors:       int = 1,
    start_data:      str = "2000-01-01",
    use_cache:       bool = True,
    release_date:    str | pd.Timestamp | None = None,
) -> dict:
    """
    Run one backtest point: fit the DFM on vintage data and return the nowcast.

    Two evaluation designs:
      * release_date=None (quarter-end backtest): evaluation_date must be on or
        before the target quarter's last day (assert_no_lookahead).
      * release_date given (through-the-quarter backtest): evaluation_date may
        fall after the quarter ends but must precede the GDP release
        (assert_before_release); the target is passed explicitly because two
        quarters can be unpublished at once.

    LOOK-AHEAD ASSERTION: assert_no_lookahead() runs first. If it raises, the
    calling code has a bug and the backtest must be fixed before proceeding.

    Parameters
    ----------
    target_quarter : pd.Timestamp
        Quarter-start date of the quarter being nowcast (e.g., 2020-01-01).
    evaluation_date : str
        ALFRED vintage cutoff — must be the last day of target_quarter.
    factor_blocks : dict or None
        Block structure from make_block_structure(); auto-built if None.
    n_factors : int
        Factors to use if factor_blocks is None (default 1).
    start_data : str
        Estimation start date for DFM (default "2000-01-01").
    use_cache : bool
        Use ALFRED cache (default True).

    Returns
    -------
    dict with:
        target_quarter    : pd.Timestamp
        evaluation_date   : str
        quarter_label     : str (e.g. "2020 Q1")
        dfm_nowcast_qtr   : float (quarterly %, or NaN)
        ar1_nowcast_qtr   : float (quarterly %, or NaN)
        converged         : bool
    """
    # Hard look-ahead check — this must pass before any data is loaded
    if release_date is None:
        assert_no_lookahead(evaluation_date, str(target_quarter.date()))
    else:
        assert_before_release(evaluation_date, release_date)
    explicit = target_quarter if release_date is not None else None

    qtr_num   = (target_quarter.month - 1) // 3 + 1
    qtr_label = f"{target_quarter.year} Q{qtr_num}"
    logger.info("Backtest: %s (eval %s)", qtr_label, evaluation_date)

    try:
        monthly_df, quarterly_df = build_vintage_dfm_data(
            as_of_date=evaluation_date,
            start=start_data,
            use_cache=use_cache,
            target_quarter=explicit,
        )
    except AssertionError:
        raise                       # look-ahead errors must never become NaN rows
    except Exception as exc:
        logger.error("build_vintage_dfm_data failed for %s: %s", qtr_label, exc)
        return {
            "target_quarter":  target_quarter,
            "evaluation_date": evaluation_date,
            "quarter_label":   qtr_label,
            "dfm_nowcast_qtr": float("nan"),
            "dfm_nowcast_se_ann": float("nan"),
            "ar1_nowcast_qtr": float("nan"),
            "ar1_robust_nowcast_qtr": float("nan"),
            "mean_nowcast_qtr": float("nan"),
            "converged":       False,
        }

    # Benchmarks from the same vintage data
    baselines = _baseline_nowcasts_vintage(quarterly_df, target_quarter=explicit)

    # DFM — build block structure from the vintage panel
    fb = factor_blocks
    if fb is None:
        fb = make_block_structure(monthly_df, quarterly_df) if n_factors > 1 else None

    try:
        results    = fit_dfm(
            monthly_df, quarterly_df,
            factor_blocks=fb, n_factors=n_factors, maxiter=500,
        )
        nowcast    = extract_nowcast(results, quarterly_df, target_quarter=explicit)
        dfm_qtr    = nowcast["nowcast_qtr_pct"] if nowcast is not None else float("nan")
        dfm_se     = nowcast["nowcast_se_ann"] if nowcast is not None else float("nan")
        converged  = True
    except Exception as exc:
        logger.error("fit_dfm failed for %s: %s", qtr_label, exc)
        dfm_qtr   = float("nan")
        dfm_se    = float("nan")
        converged = False

    return {
        "target_quarter":  target_quarter,
        "evaluation_date": evaluation_date,
        "quarter_label":   qtr_label,
        "dfm_nowcast_qtr": dfm_qtr,
        "dfm_nowcast_se_ann": dfm_se,
        **baselines,
        "converged":       converged,
    }


def result_row(
    res: dict,
    actual_gdp: pd.Series,
    first_release: pd.DataFrame,
) -> dict:
    """
    One backtest CSV row from a run_single_quarter() result.

    Shared by run_backtest() and scripts/rerun_failed_quarters.py so a re-run
    quarter always gets exactly the same columns. NaN propagates through * and
    −, so a failed fit or a missing actual simply yields NaN errors (excluded
    from every statistic).
    """
    target_q   = res["target_quarter"]
    actual_ann = float(actual_gdp.get(target_q, float("nan"))) * 4
    dfm_ann    = res["dfm_nowcast_qtr"] * 4
    ar1_ann    = res["ar1_nowcast_qtr"] * 4
    fr = first_release.loc[target_q] if target_q in first_release.index else None
    return {
        "quarter_label":          res["quarter_label"],
        "target_quarter":         target_q,
        "evaluation_date":        res["evaluation_date"],
        "dfm_nowcast_ann":        dfm_ann,
        "dfm_nowcast_se_ann":     res.get("dfm_nowcast_se_ann", float("nan")),
        "ar1_nowcast_ann":        ar1_ann,
        "ar1_robust_nowcast_ann": res["ar1_robust_nowcast_qtr"] * 4,
        "mean_nowcast_ann":       res["mean_nowcast_qtr"] * 4,
        "actual_ann":             actual_ann,
        "first_release_ann":      float(fr["first_release_ann"]) if fr is not None else float("nan"),
        "first_release_date":     fr["first_release_date"] if fr is not None else pd.NaT,
        "dfm_error_ann":          dfm_ann - actual_ann,
        "ar1_error_ann":          ar1_ann - actual_ann,
        "is_2020":                target_q.year == 2020,
        "converged":              res["converged"],
    }


def load_first_release(quarters, use_cache: bool = True) -> pd.DataFrame:
    """First-release GDP for the given quarters; empty (with a warning) if unavailable."""
    try:
        return fetch_gdp_first_release(list(quarters), use_cache=use_cache)
    except Exception as exc:
        logger.warning("First-release GDP unavailable (%s); column left NaN.", exc)
        return pd.DataFrame(columns=["first_release_ann", "first_release_date"])


def run_backtest(
    start_qs:      str = "2015-01-01",
    end_qs:        str = "2025-10-01",
    factor_blocks: dict | None = None,
    n_factors:     int = 1,
    start_data:    str = "2000-01-01",
    use_cache:     bool = True,
    save_results:  bool = True,
    results_path:  str | None = None,
) -> pd.DataFrame:
    """
    Run the full quarterly ALFRED vintage backtest.

    Design:
      For each target quarter, we evaluate on the last day of that quarter
      (evaluation_date). This guarantees the GDP advance release is not yet
      in the ALFRED data. Each DFM fit uses the 2-factor blocked structure by
      default (pass n_factors=1 and factor_blocks=None for the faster 1-factor
      backtest).

    Handling 2020:
      The is_2020 column flags 2020 Q1-Q4. compute_rmse_stats() reports RMSE
      both with and without 2020. This is NOT cherry-picking — it clearly
      documents where the model's assumptions break down (a pandemic lockdown
      vs. a normal business cycle).

    Progress output:
      Prints a line per quarter because each DFM fit takes ~15-30 seconds.
      Total backtest time for 44 quarters: approximately 15-20 minutes.
      Results are cached to data/backtest_results.csv after completion.

    Parameters
    ----------
    start_qs : str
        First backtest quarter start (default Q1 2015).
    end_qs : str
        Last backtest quarter start (default Q4 2025).
    factor_blocks : dict or None
        Pre-built block structure. If None and n_factors > 1, auto-built per
        vintage panel. If None and n_factors == 1, uses 1-factor model.
    n_factors : int
        Factors to use if block structure is not provided (default 1 for speed).
    start_data : str
        Model estimation window start date (default "2000-01-01").
    use_cache : bool
        Use ALFRED data cache (default True).
    save_results : bool
        Write DataFrame to data/backtest_results.csv on completion (default True).

    Returns
    -------
    pd.DataFrame
        Columns:
          quarter_label    : "2020 Q1"
          target_quarter   : pd.Timestamp
          evaluation_date  : str
          dfm_nowcast_ann        : float  (annualized %)
          ar1_nowcast_ann        : float  (annualized %, OLS AR(1))
          ar1_robust_nowcast_ann : float  (annualized %, Huber AR(1))
          mean_nowcast_ann       : float  (annualized %, expanding mean)
          actual_ann             : float  (annualized %, current-vintage GDP)
          first_release_ann      : float  (annualized %, as first published)
          first_release_date     : date the first estimate was published
          dfm_error_ann          : float  (dfm_nowcast_ann − actual_ann)
          ar1_error_ann          : float  (ar1_nowcast_ann − actual_ann)
          is_2020          : bool
          converged        : bool
    """
    # Current-vintage GDP actuals (the "truth" we compare nowcasts against).
    # Using latest revision is deliberate: we're measuring how close the
    # real-time model got to the best available estimate of economic reality.
    actual_gdp = _fetch_current_gdp_actuals(start=start_data)

    quarter_list = backtest_quarter_range(start_qs, end_qs)

    # Second scoring target: GDP growth as first published (advance estimate).
    first_release = load_first_release([q for q, _ in quarter_list], use_cache=use_cache)
    n = len(quarter_list)
    rows = []

    print(f"\nRunning ALFRED vintage backtest: {n} quarters  "
          f"({start_qs[:7]} → {end_qs[:7]})")
    print(f"  Model: {'blocked 2-factor DFM' if n_factors > 1 else '1-factor DFM'}")
    print(f"  Expected runtime: ~{n * 20 // 60} min {(n * 20) % 60} sec\n")

    for i, (target_q, eval_date) in enumerate(quarter_list):
        qtr_num = (target_q.month - 1) // 3 + 1
        print(f"  [{i+1:3d}/{n}] {target_q.year} Q{qtr_num}  eval={eval_date}  ", end="", flush=True)

        res = run_single_quarter(
            target_quarter=target_q,
            evaluation_date=eval_date,
            factor_blocks=factor_blocks,
            n_factors=n_factors,
            start_data=start_data,
            use_cache=use_cache,
        )

        row = result_row(res, actual_gdp, first_release)
        status = "✓" if res["converged"] else "FAILED"
        print(f"DFM={row['dfm_nowcast_ann']:>+6.1f}%  actual={row['actual_ann']:>+6.1f}%  {status}")
        rows.append(row)

    df = pd.DataFrame(rows)

    if save_results:
        out_path = Path(results_path) if results_path else BACKTEST_RESULTS_PATH
        df.to_csv(out_path, index=False)
        logger.info("Backtest results saved to %s", out_path)

    return df


def _fetch_current_gdp_actuals(start: str = "2000-01-01") -> pd.Series:
    """
    Fetch current-vintage (fully-revised) GDP growth for comparison.

    This is the 'truth' we compare nowcasts against. Using current vintage
    (rather than the first-release value) is a deliberate choice — we want
    to measure how well the model ultimately tracked economic reality, not
    just how close it got to a preliminary number that was itself revised.

    Note: this is a standard choice in the forecasting literature, though
    some papers use the first-release as the target instead. Both are valid;
    we document the choice here for reproducibility.
    """
    raw    = fetch_series("GDPC1", start=start, use_cache=True)
    growth = transform_series(raw, "log_diff").dropna()
    growth.index = pd.DatetimeIndex(growth.index).to_period("Q").to_timestamp()
    growth = growth.asfreq("QS")
    return growth


def compute_rmse_stats(results_df: pd.DataFrame) -> dict:
    """
    Compute RMSE statistics with and without the 2020 COVID period.

    Returns a dict with keys:
        dfm_full_rmse     : DFM RMSE over the full sample (annualized pp)
        dfm_ex2020_rmse   : DFM RMSE excluding 2020 Q1–Q4 (primary metric)
        dfm_covid_rmse    : DFM RMSE for 2020 Q1–Q4 only (crisis metric)
        ar1_full_rmse     : AR(1) RMSE over full sample
        ar1_ex2020_rmse   : AR(1) RMSE excluding 2020
        dfm_skill_ex2020  : RMSE reduction vs AR(1), ex-2020 (positive = DFM wins)
        dfm_mean_error    : DFM mean error (bias); positive = model over-predicts
        n_full            : number of quarters with valid errors
        n_ex_2020         : number of quarters excluding 2020
        n_covid           : number of 2020 quarters
    """
    def _rmse(series: pd.Series) -> float:
        s = series.dropna()
        return float(np.sqrt((s**2).mean())) if len(s) > 0 else float("nan")

    df = results_df.copy()
    mask_2020 = df["is_2020"]

    full_dfm  = df["dfm_error_ann"].dropna()
    ex20_dfm  = df.loc[~mask_2020, "dfm_error_ann"].dropna()
    cov_dfm   = df.loc[ mask_2020, "dfm_error_ann"].dropna()

    full_ar1  = df["ar1_error_ann"].dropna()
    ex20_ar1  = df.loc[~mask_2020, "ar1_error_ann"].dropna()

    dfm_ex20  = _rmse(ex20_dfm)
    ar1_ex20  = _rmse(ex20_ar1)

    return {
        "dfm_full_rmse":    _rmse(full_dfm),
        "dfm_ex2020_rmse":  dfm_ex20,
        "dfm_covid_rmse":   _rmse(cov_dfm),
        "ar1_full_rmse":    _rmse(full_ar1),
        "ar1_ex2020_rmse":  ar1_ex20,
        "dfm_skill_ex2020": ar1_ex20 - dfm_ex20,   # positive = DFM wins
        "dfm_mean_error":   float(full_dfm.mean()),
        "n_full":           len(full_dfm),
        "n_ex_2020":        len(ex20_dfm),
        "n_covid":          len(cov_dfm),
    }


def print_backtest_summary(results_df: pd.DataFrame) -> None:
    """Print a human-readable backtest summary to stdout."""
    stats = compute_rmse_stats(results_df)
    df    = results_df.dropna(subset=["actual_ann"])

    # Quarter range label
    dates     = results_df["target_quarter"]
    def _ql(ts):
        return f"{ts.year} Q{(ts.month - 1) // 3 + 1}"
    date_range = f"{_ql(dates.min())}  →  {_ql(dates.max())}"

    print()
    print("=" * 65)
    print("  GDP Nowcast Backtest — ALFRED Vintage, End-of-Quarter Eval")
    print("=" * 65)
    print(f"  Sample  : {date_range}  ({stats['n_full']} quarters)")
    print(f"  Model   : blocked 2-factor DFM (global + real-activity factor)")
    print()
    print(f"  {'Metric':<35}  {'DFM':>8}  {'AR(1)':>8}")
    print(f"  {'-'*35}  {'-'*8}  {'-'*8}")
    print(f"  {'RMSE — full sample (pp ann)':<35}  "
          f"{stats['dfm_full_rmse']:>8.2f}  {stats['ar1_full_rmse']:>8.2f}")
    print(f"  {'RMSE — ex-2020 (primary)':<35}  "
          f"{stats['dfm_ex2020_rmse']:>8.2f}  {stats['ar1_ex2020_rmse']:>8.2f}")
    print(f"  {'RMSE — 2020 only (crisis)':<35}  "
          f"{stats['dfm_covid_rmse']:>8.2f}  {'—':>8}")
    print()
    skill = stats["dfm_skill_ex2020"]
    sign  = "↓" if skill > 0 else "↑"
    print(f"  DFM skill vs AR(1), ex-2020: "
          f"{sign}{abs(skill):.2f} pp  "
          f"({'DFM wins' if skill > 0 else 'AR(1) wins'})")
    print(f"  DFM mean error (bias) : {stats['dfm_mean_error']:>+.2f} pp "
          f"({'over-predicts' if stats['dfm_mean_error'] > 0 else 'under-predicts'})")
    print()
    print("  NOTE: 2020 Q1–Q4 flagged as COVID outlier — pandemic supply shock")
    print("  is outside the DFM's business-cycle identifying assumptions.")
    print("  Full-sample RMSE should be viewed alongside the ex-2020 figure.")
    print("=" * 65)


def plot_backtest_results(
    results_df: pd.DataFrame,
    save_path:  str | None = None,
) -> None:
    """
    Plot the backtest results: predicted vs actual GDP growth over time.

    Two panels:
      Top: time-series of DFM nowcast, AR(1) nowcast, and actual GDP growth.
           2020 quarters are shaded in red to flag the COVID outlier period.
      Bottom: scatter of DFM nowcast vs actual, with 45-degree line.
              2020 dots are plotted in red with a distinct marker.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of run_backtest().
    save_path : str or None
        If provided, save the figure here (default: None = display only).
    """
    df    = results_df.dropna(subset=["actual_ann", "dfm_nowcast_ann"])
    mask  = df["is_2020"]
    stats = compute_rmse_stats(results_df)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    fig.suptitle(
        "GDP Nowcast Backtest — ALFRED Vintage (End-of-Quarter Evaluation)",
        fontsize=13, fontweight="bold",
    )

    # ── Top panel: time series ────────────────────────────────────────────────
    ax = axes[0]
    quarters = df["target_quarter"]

    # Shade 2020 Q1-Q4
    if mask.any():
        q_2020 = df.loc[mask, "target_quarter"]
        ax.axvspan(
            q_2020.min() - pd.DateOffset(months=1),
            q_2020.max() + pd.DateOffset(months=2),
            alpha=0.12, color="red", label="2020 (COVID)",
        )

    ax.plot(quarters, df["actual_ann"],       "k-",  lw=2,   label="Actual GDP")
    ax.plot(quarters, df["dfm_nowcast_ann"],  "b--", lw=1.5, label="DFM nowcast")
    ax.plot(quarters, df["ar1_nowcast_ann"],  "g:",  lw=1.5, label="AR(1) baseline")
    ax.axhline(0, color="gray", lw=0.7, ls="--")

    ax.set_ylabel("Annualized GDP growth (%)")
    ax.set_title("Nowcast vs Actual GDP Growth — Real-Time Vintage")
    ax.legend(loc="lower left", fontsize=9)

    rmse_label = (
        f"DFM RMSE (ex-2020): {stats['dfm_ex2020_rmse']:.2f} pp  |  "
        f"AR(1): {stats['ar1_ex2020_rmse']:.2f} pp"
    )
    ax.set_xlabel(rmse_label, fontsize=9, labelpad=6)

    # ── Bottom panel: scatter ─────────────────────────────────────────────────
    ax2   = axes[1]
    norm  = df[~mask]
    covid = df[ mask]

    ax2.scatter(norm["actual_ann"],  norm["dfm_nowcast_ann"],
                color="steelblue", s=40, alpha=0.75, label="Non-COVID quarters")
    if not covid.empty:
        ax2.scatter(covid["actual_ann"], covid["dfm_nowcast_ann"],
                    color="red", s=60, marker="D", zorder=5, label="2020 quarters")

    lims = [
        min(df["actual_ann"].min(), df["dfm_nowcast_ann"].min()) - 2,
        max(df["actual_ann"].max(), df["dfm_nowcast_ann"].max()) + 2,
    ]
    ax2.plot(lims, lims, "k--", lw=0.8, label="45° (perfect forecast)")
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel("Actual GDP growth (%, annualized)")
    ax2.set_ylabel("DFM nowcast (%, annualized)")
    ax2.set_title("DFM Nowcast vs Actual (scatter)")
    ax2.legend(fontsize=9)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Backtest plot saved to %s", save_path)
    else:
        plt.show()
