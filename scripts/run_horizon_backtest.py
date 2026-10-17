"""
run_horizon_backtest.py — nowcasts at many dates through each quarter.

The quarter-end backtest (run_phase4_backtest*.py) asks one question per
quarter. Real nowcasts are updated continuously, so the standard way to judge
them is how accuracy improves as data arrives. For every target quarter
2015Q1–2025Q4 this script nowcasts on:

  * every 14 days from the quarter's first day, and
  * the quarter's last day (links each quarter to the quarter-end backtest),

stopping the day before that quarter's GDP was first published (ALFRED
first-release date). Each date uses the ALFRED vintage of that day —
assert_before_release() enforces no look-ahead.

Per row: 2-factor and 1-factor DFM nowcasts with Kalman-smoother standard
errors, three real-time baselines, GDPNow as published that day (converted to
4×log units), and both scoring targets.

Data: instead of ~5,000 per-date ALFRED requests (~2 s each, rate-limited),
each series' full revision history is fetched once and every vintage is
rebuilt locally (src/backtest/vintage.py: materialize_vintages). The rebuilt
vintages were checked against 1,036 directly-fetched ones: all identical.

Output: data/horizon_backtest.csv. Resumable — rows already present are
skipped, and results are saved after every quarter.

Run:  python scripts/run_horizon_backtest.py
"""
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.evaluation import gdpnow_to_log_units  # noqa: E402
from src.backtest.runner import (  # noqa: E402
    _fetch_current_gdp_actuals,
    horizon_evaluation_dates as evaluation_dates,
    load_first_release,
    run_single_quarter,
)
from src.backtest.vintage import fetch_gdpnow_vintage, materialize_vintages  # noqa: E402
from src.data.fred_loader import SERIES_CATALOG  # noqa: E402

OUTPUT = ROOT / "data" / "horizon_backtest.csv"

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")


def main() -> None:
    quarters = pd.date_range("2015-01-01", "2025-10-01", freq="QS")
    first_release = load_first_release(quarters)
    actual = _fetch_current_gdp_actuals(start="2000-01-01")

    done = pd.DataFrame()
    if OUTPUT.exists():
        done = pd.read_csv(OUTPUT, parse_dates=["target_quarter", "evaluation_date", "release_date"],
                           float_precision="round_trip")
    done_keys = set(zip(done.get("target_quarter", []), done.get("evaluation_date", [])))

    grid = {q: evaluation_dates(q, pd.Timestamp(first_release.loc[q, "first_release_date"]))
            for q in quarters}
    jobs = [(s, str(d.date())) for q in reversed(quarters) for d in grid[q]
            for s in list(SERIES_CATALOG) + (["GDPNOW"] if d >= pd.Timestamp("2016-05-17") else [])]
    # One revision-history request per series rebuilds every needed vintage
    # (validated equal to direct per-date ALFRED fetches on 1,036 files).
    print(f"Materialized {materialize_vintages(jobs)} vintage files from revision histories.",
          flush=True)

    rows = []
    for q in quarters:
        release = pd.Timestamp(first_release.loc[q, "first_release_date"])
        dates = [d for d in evaluation_dates(q, release) if (q, d) not in done_keys]
        if not dates:
            continue
        label = f"{q.year} Q{q.quarter}"
        print(f"{label}: {len(dates)} dates (release {release.date()})", flush=True)
        for d in dates:
            ev = str(d.date())
            r2 = run_single_quarter(q, ev, n_factors=2, release_date=release)
            r1 = run_single_quarter(q, ev, n_factors=1, release_date=release)
            gn = fetch_gdpnow_vintage(q, ev)
            rows.append({
                "quarter_label":          label,
                "target_quarter":         q,
                "evaluation_date":        d,
                "release_date":           release,
                "days_to_release":        (release - d).days,
                "dfm_nowcast_ann":        r2["dfm_nowcast_qtr"] * 4,
                "dfm_nowcast_se_ann":     r2["dfm_nowcast_se_ann"],
                "dfm1_nowcast_ann":       r1["dfm_nowcast_qtr"] * 4,
                "dfm1_nowcast_se_ann":    r1["dfm_nowcast_se_ann"],
                "ar1_nowcast_ann":        r2["ar1_nowcast_qtr"] * 4,
                "ar1_robust_nowcast_ann": r2["ar1_robust_nowcast_qtr"] * 4,
                "mean_nowcast_ann":       r2["mean_nowcast_qtr"] * 4,
                "gdpnow_nowcast_ann":     float(gdpnow_to_log_units(pd.Series([gn])).iloc[0]),
                "actual_ann":             float(actual.get(q, np.nan)) * 4,
                "first_release_ann":      float(first_release.loc[q, "first_release_ann"]),
                "is_2020":                q.year == 2020,
                "converged":              bool(r2["converged"] and r1["converged"]),
            })
            print(f"   {ev}  ({(release - d).days:3d}d)  DFM {rows[-1]['dfm_nowcast_ann']:+6.2f}"
                  f" ± {rows[-1]['dfm_nowcast_se_ann']:.2f}   GDPNow {rows[-1]['gdpnow_nowcast_ann']:+6.2f}",
                  flush=True)
        out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        out.sort_values(["target_quarter", "evaluation_date"]).to_csv(OUTPUT, index=False)

    final = pd.read_csv(OUTPUT)
    print(f"\n{len(final)} rows in {OUTPUT.name}; "
          f"{int((~final['converged'].astype(bool)).sum())} with a failed fit.")


if __name__ == "__main__":
    main()
