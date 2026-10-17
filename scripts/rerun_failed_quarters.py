"""
Re-run only the backtest quarters that failed, and merge them into the CSV.

A quarter "failed" if its DFM nowcast is missing or EM did not converge —
typically because an ALFRED download hit FRED's rate limit. Re-running just
those quarters is much faster than the full backtest, because every other
vintage is already cached in data/alfred/.

Usage:
  python scripts/rerun_failed_quarters.py              # 1-factor results
  python scripts/rerun_failed_quarters.py --two-factor # 2-factor results

Rows are rebuilt with the same helper run_backtest() uses, so re-run quarters
get exactly the same columns. Re-run scripts/evaluate_backtest.py afterwards.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")

import pandas as pd

from src.backtest.runner import (
    BACKTEST_RESULTS_PATH,
    _fetch_current_gdp_actuals,
    load_first_release,
    plot_backtest_results,
    print_backtest_summary,
    result_row,
    run_single_quarter,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--two-factor", action="store_true",
                    help="operate on data/backtest_results_2factor.csv")
    args = ap.parse_args()

    if args.two_factor:
        path, n_factors = ROOT / "data" / "backtest_results_2factor.csv", 2
        fig = ROOT / "docs" / "figures" / "phase4_backtest_2factor.png"
    else:
        path, n_factors = BACKTEST_RESULTS_PATH, 1
        fig = ROOT / "docs" / "figures" / "phase4_backtest.png"

    existing = pd.read_csv(path, parse_dates=["target_quarter", "first_release_date"],
                           float_precision="round_trip")   # rewrite untouched rows exactly
    failed = existing[existing["dfm_nowcast_ann"].isna() | ~existing["converged"].astype(bool)]
    if failed.empty:
        print(f"No failed quarters in {path.name}. Nothing to do.")
        return

    actual_gdp = _fetch_current_gdp_actuals(start="2000-01-01")
    first_release = load_first_release(failed["target_quarter"])

    print(f"\nRe-running {len(failed)} failed quarter(s) in {path.name}...\n")
    new_rows = []
    for _, row in failed.iterrows():
        print(f"  {row['quarter_label']}  eval={row['evaluation_date']}  ", end="", flush=True)
        res = run_single_quarter(
            target_quarter=row["target_quarter"],
            evaluation_date=str(row["evaluation_date"])[:10],
            n_factors=n_factors,
            start_data="2000-01-01",
        )
        new = result_row(res, actual_gdp, first_release)
        print(f"DFM={new['dfm_nowcast_ann']:>+6.1f}%  {'✓' if res['converged'] else 'FAILED'}")
        new_rows.append(new)

    new_df = pd.DataFrame(new_rows)
    kept = existing[~existing["quarter_label"].isin(new_df["quarter_label"])]
    full = (pd.concat([kept, new_df], ignore_index=True)
              .sort_values("target_quarter").reset_index(drop=True))
    full.to_csv(path, index=False)
    print(f"\nSaved merged results to {path}")

    print_backtest_summary(full)
    plot_backtest_results(full, save_path=str(fig))

    still = full.loc[full["dfm_nowcast_ann"].isna(), "quarter_label"].tolist()
    if still:
        print(f"\n⚠  Still failed after re-run: {still}")


if __name__ == "__main__":
    main()
