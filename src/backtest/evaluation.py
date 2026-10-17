"""
evaluation.py — scoring backtest results honestly.

What this module adds on top of a single RMSE number:

  * Two scoring targets. "Current vintage" GDP (today's revised numbers) and
    "first release" GDP (the advance estimate, what was actually published a
    month after the quarter). A real-time nowcast is arguably aiming at the
    latter; reporting both shows how much the verdict depends on the choice.

  * Several benchmarks, not one. The OLS AR(1), a Huber-robust AR(1), and the
    expanding mean of past growth ("predict the average").

  * Uncertainty about the comparison. With ~40 normal quarters, a 0.2 pp RMSE
    gap can easily be luck. Two standard tools:
      - Diebold–Mariano (1995) test of equal squared-error loss, with the
        Harvey–Leybourne–Newbold (1997) small-sample correction;
      - a circular block bootstrap confidence interval for the RMSE gap
        (blocks of 4 quarters keep any short-run error correlation intact).

Conventions: errors are forecast − actual (positive = over-prediction), in
annualized percentage points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Model / benchmark nowcast columns in the backtest CSV, and display names.
NOWCAST_COLUMNS: dict[str, str] = {
    "DFM":           "dfm_nowcast_ann",
    "AR(1)":         "ar1_nowcast_ann",
    "Robust AR(1)":  "ar1_robust_nowcast_ann",
    "Expanding mean": "mean_nowcast_ann",
    "GDPNow":        "gdpnow_nowcast_ann",   # optional; see gdpnow_to_log_units()
}

TARGET_COLUMNS: dict[str, str] = {
    "current vintage": "actual_ann",
    "first release":   "first_release_ann",
}


def gdpnow_to_log_units(gdpnow_saar: pd.Series) -> pd.Series:
    """
    Convert GDPNow's compounded annual rate to this project's units.

    BEA and GDPNow report (Y_t/Y_{t-1})^4 − 1; this project uses
    400·ln(Y_t/Y_{t-1}). Equivalent: 100·ln(1 + g/100). The two differ by
    < 0.1 pp at normal growth rates but by ~10 pp in 2020 Q2.
    """
    return 100.0 * np.log1p(gdpnow_saar / 100.0)


def rmse(errors: pd.Series | np.ndarray) -> float:
    e = pd.Series(errors).dropna()
    return float(np.sqrt((e ** 2).mean())) if len(e) else float("nan")


def sample_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """The three reporting samples used throughout the project."""
    return {
        "ex-2020":   ~df["is_2020"],
        "full":      pd.Series(True, index=df.index),
        "2020 only": df["is_2020"],
    }


def dm_test(e_model, e_bench, h: int = 1) -> dict:
    """
    Diebold–Mariano test of equal predictive accuracy (squared-error loss).

    d_t = e_model_t² − e_bench_t². H0: E[d] = 0. A negative statistic means
    the model's squared errors are smaller. For h-step forecasts the variance
    of mean(d) uses autocovariances up to lag h−1 (none for h = 1, a nowcast).
    The Harvey–Leybourne–Newbold correction rescales the statistic and uses a
    t(n−1) reference distribution, which matters at n ≈ 40.

    Returns dict(stat, pvalue, mean_loss_diff, n). Two-sided p-value.
    """
    pair = pd.DataFrame({"m": e_model, "b": e_bench}).dropna()
    d = (pair["m"] ** 2 - pair["b"] ** 2).to_numpy()
    n = len(d)
    if n < 3:
        return {"stat": float("nan"), "pvalue": float("nan"), "mean_loss_diff": float("nan"), "n": n}

    d_bar = d.mean()
    dc = d - d_bar
    lrv = dc @ dc / n
    for k in range(1, h):
        lrv += 2 * (dc[k:] @ dc[:-k]) / n
    if lrv <= 0:
        return {"stat": 0.0, "pvalue": 1.0, "mean_loss_diff": float(d_bar), "n": n}

    dm = d_bar / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = float(dm * hln)
    pvalue = float(2 * stats.t.sf(abs(stat), df=n - 1))
    return {"stat": stat, "pvalue": pvalue, "mean_loss_diff": float(d_bar), "n": n}


def block_bootstrap_rmse_diff(
    e_model, e_bench, block_len: int = 4, n_boot: int = 10_000, seed: int = 0,
) -> dict:
    """
    Circular block bootstrap CI for RMSE(model) − RMSE(benchmark).

    Quarters are resampled in blocks of `block_len` consecutive quarters (with
    wrap-around), keeping each quarter's pair of errors together. Negative
    values favour the model. Returns dict(diff, lo, hi) with a 95% interval.
    """
    pair = pd.DataFrame({"m": e_model, "b": e_bench}).dropna().to_numpy()
    n = len(pair)
    point = rmse(pair[:, 0]) - rmse(pair[:, 1])
    if n < block_len:
        return {"diff": point, "lo": float("nan"), "hi": float("nan")}

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_len)) % n
    idx = idx.reshape(n_boot, -1)[:, :n]
    sq = pair[idx] ** 2                                   # (n_boot, n, 2)
    diffs = np.sqrt(sq[:, :, 0].mean(axis=1)) - np.sqrt(sq[:, :, 1].mean(axis=1))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(point), "lo": float(lo), "hi": float(hi)}


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    RMSE, MAE and bias for every (target, sample, model) combination.

    Long format: columns target, sample, model, n, rmse, mae, bias.
    Models or targets whose columns are absent from df are skipped.
    """
    rows = []
    for tname, tcol in TARGET_COLUMNS.items():
        if tcol not in df:
            continue
        for sname, mask in sample_masks(df).items():
            for mname, mcol in NOWCAST_COLUMNS.items():
                if mcol not in df:
                    continue
                err = (df.loc[mask, mcol] - df.loc[mask, tcol]).dropna()
                rows.append({
                    "target": tname, "sample": sname, "model": mname, "n": len(err),
                    "rmse": rmse(err), "mae": float(err.abs().mean()), "bias": float(err.mean()),
                })
    return pd.DataFrame(rows)


def comparison_tests(df: pd.DataFrame, model: str = "DFM") -> pd.DataFrame:
    """
    DFM versus each benchmark: RMSE gap with bootstrap CI, and DM test.

    Long format: target, sample, benchmark, n, rmse_model, rmse_bench,
    rmse_diff, ci_lo, ci_hi, dm_stat, dm_pvalue. rmse_diff < 0 favours the
    model. The 2020-only sample (n = 4) is excluded — too few points to test.
    """
    mcol = NOWCAST_COLUMNS[model]
    rows = []
    for tname, tcol in TARGET_COLUMNS.items():
        if tcol not in df:
            continue
        for sname, mask in sample_masks(df).items():
            if sname == "2020 only":
                continue
            sub = df.loc[mask]
            for bname, bcol in NOWCAST_COLUMNS.items():
                if bname == model or bcol not in df:
                    continue
                # Score both on the quarters where both exist (GDPNow starts 2016 Q2).
                paired = sub[[mcol, bcol, tcol]].dropna()
                e_m = paired[mcol] - paired[tcol]
                e_b = paired[bcol] - paired[tcol]
                boot = block_bootstrap_rmse_diff(e_m, e_b)
                dm = dm_test(e_m, e_b)
                rows.append({
                    "target": tname, "sample": sname, "benchmark": bname, "n": dm["n"],
                    "rmse_model": rmse(e_m), "rmse_bench": rmse(e_b),
                    "rmse_diff": boot["diff"], "ci_lo": boot["lo"], "ci_hi": boot["hi"],
                    "dm_stat": dm["stat"], "dm_pvalue": dm["pvalue"],
                })
    return pd.DataFrame(rows)
