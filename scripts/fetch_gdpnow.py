"""
fetch_gdpnow.py — pull Atlanta Fed GDPNow nowcasts for the backtest comparison.

What this script does (plain English):
  The Atlanta Fed publishes its own GDP nowcast, "GDPNow", and FRED archives
  every historical version of it in ALFRED. For each of the 44 quarters in our
  Phase 4 backtest, we fetch GDPNow **as it stood on the same end-of-quarter
  evaluation date our own model was scored on**, and record its nowcast for
  that target quarter. That makes the comparison apples-to-apples:
  both models, same information cutoff, no look-ahead bias.

Output:
  data/gdpnow_eoq.csv with columns:
    quarter_label, target_quarter, evaluation_date, gdpnow_ann

Availability caveat (verified 2026-06-11):
  The first GDPNow vintage in ALFRED is 2016-05-17 — GDPNow existed earlier,
  but FRED only began archiving it then. The first 5 backtest quarters
  (2015 Q1 – 2016 Q1) therefore have no end-of-quarter value and are stored
  as NaN. Comparisons with GDPNow therefore start at 2016 Q2.

Units:
  GDPNOW is a compounded annual rate, (Y_t/Y_{t-1})^4 − 1, as BEA reports.
  This project's columns are 4×log growth. Stored here as published; convert
  with src.backtest.evaluation.gdpnow_to_log_units() before comparing (the gap
  is < 0.1 pp in normal quarters but ~10 pp in 2020 Q2).

Run:  source .venv/bin/activate && python scripts/fetch_gdpnow.py
      (~40 ALFRED calls on first run, throttled; cached in data/alfred/ after.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.backtest.vintage import fetch_gdpnow_vintage

BACKTEST_CSV = Path("data/backtest_results_2factor.csv")
OUTPUT_CSV   = Path("data/gdpnow_eoq.csv")

def fetch_gdpnow_for_quarter(target_quarter: str, evaluation_date: str) -> float:
    """GDPNow's nowcast for `target_quarter` as published on `evaluation_date` (NaN if none)."""
    return fetch_gdpnow_vintage(target_quarter, evaluation_date)


def main() -> pd.DataFrame:
    bt = pd.read_csv(BACKTEST_CSV)

    rows = []
    for _, r in bt.iterrows():
        gdpnow = fetch_gdpnow_for_quarter(r["target_quarter"], r["evaluation_date"])
        rows.append({
            "quarter_label":   r["quarter_label"],
            "target_quarter":  r["target_quarter"],
            "evaluation_date": r["evaluation_date"],
            "gdpnow_ann":      gdpnow,
        })
        status = f"{gdpnow:+.2f}%" if not np.isnan(gdpnow) else "n/a (pre-ALFRED)"
        print(f"  {r['quarter_label']}: GDPNow @ {r['evaluation_date']} = {status}")

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)

    n_ok = out["gdpnow_ann"].notna().sum()
    print(f"\nSaved {OUTPUT_CSV} — {n_ok}/{len(out)} quarters with GDPNow values.")
    return out


if __name__ == "__main__":
    main()
