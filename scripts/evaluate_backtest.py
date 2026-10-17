"""
evaluate_backtest.py — the full scorecard for the saved backtests.

Reads data/backtest_results_2factor.csv and data/backtest_results.csv (run the
two run_phase4_backtest*.py scripts first) and writes:

  data/evaluation_accuracy.csv  RMSE / MAE / bias for every model, both
                                scoring targets (current vintage, first
                                release) and samples (ex-2020, full, 2020)
  data/evaluation_tests.csv     DFM vs each benchmark: RMSE gap, 95% block-
                                bootstrap CI, Diebold–Mariano (HLN) p-value

GDPNow (data/gdpnow_eoq.csv, end-of-quarter ALFRED vintages, 2016 Q2 on) is
included as an outside reference after converting its compounded annual rate
to this project's 4×log units. Its rows cover fewer quarters (see column n),
and comparisons with it use only the quarters both have.

Run:  python scripts/evaluate_backtest.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.evaluation import (  # noqa: E402
    accuracy_table,
    comparison_tests,
    gdpnow_to_log_units,
)

GDPNOW_CSV = ROOT / "data" / "gdpnow_eoq.csv"

RUNS = {
    "2-factor": ROOT / "data" / "backtest_results_2factor.csv",
    "1-factor": ROOT / "data" / "backtest_results.csv",
}


def main() -> None:
    acc_all, tests_all = [], []
    for spec, path in RUNS.items():
        df = pd.read_csv(path, parse_dates=["target_quarter"])
        if GDPNOW_CSV.exists():
            gn = pd.read_csv(GDPNOW_CSV, parse_dates=["target_quarter"])
            gn["gdpnow_nowcast_ann"] = gdpnow_to_log_units(gn["gdpnow_ann"])
            df = df.merge(gn[["target_quarter", "gdpnow_nowcast_ann"]],
                          on="target_quarter", how="left")
        acc = accuracy_table(df)
        tests = comparison_tests(df)
        # Benchmarks don't depend on the DFM spec — keep them once (2-factor run).
        acc["dfm_spec"] = spec
        if spec != "2-factor":
            acc = acc[acc["model"] == "DFM"]
        tests["dfm_spec"] = spec
        acc_all.append(acc)
        tests_all.append(tests)

    acc = pd.concat(acc_all, ignore_index=True)
    acc["model"] = acc.apply(
        lambda r: f"DFM {r['dfm_spec']}" if r["model"] == "DFM" else r["model"], axis=1)
    acc = acc.drop(columns="dfm_spec")
    tests = pd.concat(tests_all, ignore_index=True)

    acc.round(4).to_csv(ROOT / "data" / "evaluation_accuracy.csv", index=False)
    tests.round(4).to_csv(ROOT / "data" / "evaluation_tests.csv", index=False)

    pd.set_option("display.width", 200)
    for target in ["current vintage", "first release"]:
        sub = acc[acc["target"] == target]
        print(f"\n=== RMSE / bias vs {target.upper()} GDP (annualized pp) ===")
        piv = sub.pivot_table(index="model", columns="sample", values=["rmse", "bias"])
        piv = piv.reindex(columns=["ex-2020", "full", "2020 only"], level=1)
        print(piv.round(2).to_string())

    print("\n=== DFM vs benchmarks: RMSE gap (neg = DFM better), 95% CI, DM p-value ===")
    show = tests[["dfm_spec", "target", "sample", "benchmark", "n",
                  "rmse_diff", "ci_lo", "ci_hi", "dm_pvalue"]]
    print(show.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
