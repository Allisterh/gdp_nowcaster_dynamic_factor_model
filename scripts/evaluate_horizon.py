"""
evaluate_horizon.py — scorecard for the through-the-quarter backtest.

Reads data/horizon_backtest.csv (scripts/run_horizon_backtest.py) and writes:

  data/horizon_accuracy.csv  RMSE / MAE / bias per horizon bucket, model,
                             target (first release, current vintage), sample
  data/horizon_tests.csv     DFM 2-factor vs each benchmark per bucket:
                             Diebold–Mariano (HLN) p-value. Losses are first
                             averaged within each quarter, because nowcasts of
                             the same quarter are not independent.
  data/calibration.csv       68% / 90% interval coverage, CRPS, mean band width
  docs/figures/horizon_rmse.png       RMSE vs days until GDP release
  docs/figures/fan_chart_example.png  nowcast path + bands, most recent quarter
  docs/figures/pit_histogram.png      PIT histogram (calibration check)

Run:  python scripts/evaluate_horizon.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.evaluation import (  # noqa: E402
    HORIZON_BUCKETS,
    TARGET_COLUMNS,
    dm_test_losses,
    gaussian_crps,
    horizon_accuracy,
    horizon_bucket,
    interval_coverage,
    pit_values,
    rmse,
)

DATA, FIG = ROOT / "data", ROOT / "docs" / "figures"
MODELS = {
    "DFM 2-factor":   "dfm_nowcast_ann",
    "DFM 1-factor":   "dfm1_nowcast_ann",
    "Robust AR(1)":   "ar1_robust_nowcast_ann",
    "Expanding mean": "mean_nowcast_ann",
    "AR(1)":          "ar1_nowcast_ann",
    "GDPNow":         "gdpnow_nowcast_ann",
}
DFM_SE = {"DFM 2-factor": "dfm_nowcast_se_ann", "DFM 1-factor": "dfm1_nowcast_se_ann"}
COLORS = {"DFM 2-factor": "#14365D", "Robust AR(1)": "#8A97A6",
          "Expanding mean": "#B0413E", "GDPNow": "#E8A33D"}


def samples(df):
    return {"ex-2020": ~df["is_2020"], "full": pd.Series(True, index=df.index)}


def tests_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    df = df.assign(horizon=horizon_bucket(df["days_to_release"]))
    for tname, tcol in TARGET_COLUMNS.items():
        for sname, mask in samples(df).items():
            for label, *_ in HORIZON_BUCKETS + [("all horizons", 0, 0)]:
                sub = df[mask] if label == "all horizons" else df[mask & (df["horizon"] == label)]
                for bname in ["Robust AR(1)", "Expanding mean", "AR(1)", "GDPNow", "DFM 1-factor"]:
                    pair = sub[["target_quarter", MODELS["DFM 2-factor"], MODELS[bname], tcol]].dropna()
                    if pair.empty:
                        continue
                    d = ((pair[MODELS["DFM 2-factor"]] - pair[tcol]) ** 2
                         - (pair[MODELS[bname]] - pair[tcol]) ** 2)
                    per_q = d.groupby(pair["target_quarter"]).mean()
                    dm = dm_test_losses(per_q)
                    rows.append({
                        "target": tname, "sample": sname, "horizon": label, "benchmark": bname,
                        "n_quarters": dm["n"], "n_nowcasts": len(pair),
                        "rmse_dfm": rmse(pair[MODELS["DFM 2-factor"]] - pair[tcol]),
                        "rmse_bench": rmse(pair[MODELS[bname]] - pair[tcol]),
                        "dm_pvalue": dm["pvalue"],
                    })
    out = pd.DataFrame(rows)
    out["rmse_diff"] = out["rmse_dfm"] - out["rmse_bench"]
    return out


def calibration_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    df = df.assign(horizon=horizon_bucket(df["days_to_release"]))
    for model, se_col in DFM_SE.items():
        m_col = MODELS[model]
        for tname, tcol in TARGET_COLUMNS.items():
            for sname, mask in samples(df).items():
                for label, *_ in HORIZON_BUCKETS + [("all horizons", 0, 0)]:
                    sub = df[mask] if label == "all horizons" else df[mask & (df["horizon"] == label)]
                    sub = sub[[m_col, se_col, tcol]].dropna()
                    if sub.empty:
                        continue
                    m, s, y = sub[m_col], sub[se_col], sub[tcol]
                    rows.append({
                        "model": model, "target": tname, "sample": sname, "horizon": label,
                        "n": len(sub),
                        "coverage_68": interval_coverage(m, s, y, 0.68),
                        "coverage_90": interval_coverage(m, s, y, 0.90),
                        "mean_se": float(s.mean()),
                        "rmse": rmse(m - y),
                        "crps": gaussian_crps(m, s, y),
                        "mae": float((m - y).abs().mean()),
                    })
    return pd.DataFrame(rows)


def plot_horizon_rmse(df: pd.DataFrame) -> None:
    sub = df[~df["is_2020"]].copy()
    edges = np.arange(0, 136, 15)
    sub["bin"] = pd.cut(sub["days_to_release"], edges, right=True)
    centers = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(9, 5))
    for model, color in COLORS.items():
        col = MODELS[model]
        r = sub.groupby("bin", observed=False).apply(
            lambda g: rmse(g[col] - g["first_release_ann"]) if g[col].notna().sum() >= 10 else np.nan)
        ax.plot(centers, r.values, marker="o", lw=2.2 if model == "DFM 2-factor" else 1.6,
                color=color, label=model)
    ax.invert_xaxis()
    ax.set_xlabel("Days until the quarter's first GDP release")
    ax.set_ylabel("RMSE vs first-release GDP (annualized pp)")
    ax.set_title("Nowcast accuracy as data arrives, 2015–2025 excl. 2020")
    ax.axvline(30, color="#bbb", lw=0.8, ls=":")
    ax.text(31, ax.get_ylim()[1] * 0.97, "≈ quarter ends", ha="right", va="top", fontsize=8, color="#777")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "horizon_rmse.png", dpi=150)
    plt.close(fig)


def plot_fan_chart(df: pd.DataFrame) -> str:
    q = df["target_quarter"].max()                  # rule: most recent quarter, not cherry-picked
    sub = df[df["target_quarter"] == q].sort_values("evaluation_date")
    x, m, s = sub["evaluation_date"], sub["dfm_nowcast_ann"], sub["dfm_nowcast_se_ann"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(x, m - 1.645 * s, m + 1.645 * s, color="#14365D", alpha=0.12, label="90% band")
    ax.fill_between(x, m - 0.994 * s, m + 0.994 * s, color="#14365D", alpha=0.22, label="68% band")
    ax.plot(x, m, color="#14365D", marker="o", lw=2, label="DFM nowcast")
    ax.plot(x, sub["gdpnow_nowcast_ann"], color="#E8A33D", marker="s", lw=1.5, label="GDPNow")
    ax.axhline(sub["first_release_ann"].iloc[0], color="k", ls="--", lw=1.2,
               label=f"First release ({sub['release_date'].iloc[0]:%Y-%m-%d})")
    label = sub["quarter_label"].iloc[0]
    ax.set_title(f"How the {label} nowcast evolved (most recent quarter in the sample)")
    ax.set_ylabel("Annualized growth (%)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "fan_chart_example.png", dpi=150)
    plt.close(fig)
    return label


def plot_pit(df: pd.DataFrame) -> None:
    sub = df[~df["is_2020"]]
    pit = pit_values(sub["dfm_nowcast_ann"], sub["dfm_nowcast_se_ann"], sub["first_release_ann"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pit, bins=10, range=(0, 1), color="#14365D", alpha=0.8, edgecolor="white")
    ax.axhline(len(pit) / 10, color="k", ls="--", lw=1, label="perfect calibration")
    ax.set_xlabel("PIT = P(model says GDP ≤ outcome)")
    ax.set_ylabel("Nowcasts")
    ax.set_title("Calibration of DFM intervals (excl. 2020, vs first release)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "pit_histogram.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA / "horizon_backtest.csv",
                     parse_dates=["target_quarter", "evaluation_date", "release_date"])
    acc = horizon_accuracy(df, MODELS)
    tests = tests_table(df)
    cal = calibration_table(df)
    acc.round(4).to_csv(DATA / "horizon_accuracy.csv", index=False)
    tests.round(4).to_csv(DATA / "horizon_tests.csv", index=False)
    cal.round(4).to_csv(DATA / "calibration.csv", index=False)
    plot_horizon_rmse(df)
    fan_q = plot_fan_chart(df)
    plot_pit(df)

    pd.set_option("display.width", 220)
    print(f"{len(df)} nowcasts, {df['target_quarter'].nunique()} quarters; fan chart: {fan_q}")
    for target in TARGET_COLUMNS:
        sub = acc[(acc["target"] == target) & (acc["sample"] == "ex-2020")]
        print(f"\n=== RMSE ex-2020 vs {target} by horizon ===")
        print(sub.pivot(index="model", columns="horizon", values="rmse")
                 [[b for b, *_ in HORIZON_BUCKETS]].round(2).to_string())
    print("\n=== DFM 2-factor vs benchmarks, ex-2020, first release (DM on per-quarter losses) ===")
    t = tests[(tests["sample"] == "ex-2020") & (tests["target"] == "first release")]
    print(t[["horizon", "benchmark", "n_quarters", "rmse_dfm", "rmse_bench", "rmse_diff", "dm_pvalue"]]
          .round(2).to_string(index=False))
    print("\n=== Calibration (DFM 2-factor, ex-2020, first release) ===")
    c = cal[(cal["model"] == "DFM 2-factor") & (cal["sample"] == "ex-2020") & (cal["target"] == "first release")]
    print(c[["horizon", "n", "coverage_68", "coverage_90", "mean_se", "rmse", "crps", "mae"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
