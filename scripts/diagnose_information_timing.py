"""
diagnose_information_timing.py — how much of the nowcast error is just timing?

At the backtest's evaluation date (the last day of the target quarter), the
third month's hard data has not been published yet: on average only the two
regional Fed surveys are available for month 3. And the Mariano–Murasawa
aggregation gives the previous quarter's last two months a third of the
weight in a quarter's growth. So a quarter-end nowcast legitimately leans on
last quarter's data.

This script quantifies that. For each target quarter it compares:
  * the real-time nowcast (from data/backtest_results_2factor.csv), and
  * a "complete-month" estimate: the 2-factor DFM fitted on the vintage at
    the END OF THE NEXT QUARTER — when all three months of the target
    quarter are published — with the target quarter's GDP hidden.

The complete-month estimate is NOT a real-time forecast (it uses data from
up to three months later and is fitted on a longer sample); it is a
diagnostic of how much information month 3 carries. Uses only cached ALFRED
vintages; the last target quarter is skipped when the next vintage is not
cached.

Run:  python scripts/diagnose_information_timing.py
"""
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.runner import backtest_quarter_range  # noqa: E402
from src.backtest.vintage import ALFRED_CACHE_DIR, build_vintage_dfm_data  # noqa: E402
from src.model.dfm import fit_dfm, make_block_structure, quarter_end_month  # noqa: E402

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)


def complete_month_estimate(target_q: pd.Timestamp, next_eval: str) -> float:
    """Annualized DFM estimate of target_q's GDP from the next quarter's vintage."""
    m, q = build_vintage_dfm_data(as_of_date=next_eval, start="2000-01-01")
    q = q.copy()
    q.loc[target_q, "GDPC1"] = np.nan          # hide the (released) target quarter
    res = fit_dfm(m, q, factor_blocks=make_block_structure(m, q), maxiter=500)
    key = quarter_end_month(res, target_q)
    pred = res.get_prediction(start=key, end=key, information_set="smoothed")
    return float(pred.predicted_mean["GDPC1"].iloc[0]) * 4


def main() -> None:
    bt = pd.read_csv(ROOT / "data" / "backtest_results_2factor.csv",
                     parse_dates=["target_quarter"]).set_index("target_quarter")
    pairs = backtest_quarter_range("2015-01-01", "2026-01-01")
    rows = []
    for (q, _), (_, next_eval) in zip(pairs[:-1], pairs[1:]):
        if not (ALFRED_CACHE_DIR / f"GDPC1_{next_eval}.parquet").exists():
            print(f"skip {q.date()}: vintage {next_eval} not cached")
            continue
        rows.append({"target_quarter": q, "complete_month_ann": complete_month_estimate(q, next_eval)})
        print(f"  {q.year} Q{q.quarter}: {rows[-1]['complete_month_ann']:+6.2f}%", flush=True)

    df = bt.join(pd.DataFrame(rows).set_index("target_quarter"), how="inner")
    lag = df["actual_ann"].shift(1)
    out = []
    for label, mask in [("full", np.ones(len(df), bool)), ("ex-2020", ~df["is_2020"].values)]:
        d, lg = df[mask], lag[mask]
        for col, name in [("dfm_nowcast_ann", "real-time (quarter end)"),
                          ("complete_month_ann", "complete-month (diagnostic)")]:
            err = d[col] - d["actual_ann"]
            out.append({"sample": label, "estimate": name,
                        "RMSE": np.sqrt((err ** 2).mean()),
                        "corr(est, actual_t)": d[col].corr(d["actual_ann"]),
                        "corr(est, actual_t-1)": d[col].corr(lg)})
    print("\n", pd.DataFrame(out).round(2).to_string(index=False))
    ex = df[~df["is_2020"]]
    print(f"\nFor reference, corr(actual_t, actual_t-1) ex-2020 = "
          f"{ex['actual_ann'].corr(df['actual_ann'].shift(1)[~df['is_2020']]):.2f} "
          f"(n = {len(ex)})")


if __name__ == "__main__":
    main()
