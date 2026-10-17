"""
site_builder.py — build the public research-note page (static HTML).

The page (site_template/index.html) is a single self-contained file; this
module computes the day's numbers and injects them as JSON. Everything shown
is generated from data, including every sentence, so nothing goes stale:

  hero     the current quarter's nowcast path: every 14 days from the quarter
           start plus today, each from that day's ALFRED vintage (the same
           no-look-ahead machinery as the backtest), with GDPNow alongside.
           The called-out "latest estimate" is the path's last point.
  news     which releases moved the nowcast since the last GDP release
           (compute_news on today's data).
  accuracy RMSE by days before release, from data/horizon_backtest.csv.
  factor   the model's business-cycle index (global factor), quarterly.

Run via scripts/build_site.py (daily in GitHub Actions).
"""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.evaluation import gdpnow_to_log_units
from src.backtest.runner import horizon_evaluation_dates, run_single_quarter
from src.backtest.vintage import (
    ALFRED_CACHE_DIR,
    _alfred_cache_path,
    fetch_all_releases,
    fetch_gdpnow_vintage,
    materialize_vintages,
    vintage_from_releases,
)
from src.data.fred_loader import SERIES_CATALOG
from src.model.dfm import GDP_ADVANCE_LAG_DAYS, compute_news, make_block_structure, prepare_dfm_data

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "site_template" / "index.html"

AUTHOR = "Joseph Russell-Goodman"

# Plain-English names for the news chart.
NEWS_LABELS = {
    "INDPRO": "Industrial production", "PAYEMS": "Payrolls", "UNRATE": "Unemployment rate",
    "PCEC96": "Consumer spending", "RSAFS": "Retail sales", "HOUST": "Housing starts",
    "DGORDER": "Durable goods orders", "DSPIC96": "Disposable income", "CPIAUCSL": "Consumer prices",
    "GACDISA066MSFRBNY": "Empire State survey", "GACDFSA066MSFRBPHI": "Philly Fed survey",
    "GDPC1": "GDP release",
}

NBER_RECESSIONS = [("2001-03", "2001-11"), ("2007-12", "2009-06"), ("2020-02", "2020-04")]


def recession_quarters() -> list[list[str]]:
    """NBER peak/trough months → first-month labels of their quarters ('2020-01')."""
    lab = lambda m: (lambda q: f"{q.year}-{q.month:02d}")(pd.Period(m, "M").asfreq("Q").to_timestamp())
    return [[lab(s), lab(e)] for s, e in NBER_RECESSIONS]


# ── Text ──────────────────────────────────────────────────────────────────────

def quarter_label(q: pd.Timestamp) -> str:
    return f"Q{q.quarter} {q.year}"


def headline(nowcast_ann: float, target: pd.Timestamp, today: pd.Timestamp) -> str:
    """'US growth is tracking about 2.5% this quarter' (or 'in Q3 2026' after it ends)."""
    when = ("this quarter" if target.to_period("Q") == today.to_period("Q")
            else f"in {quarter_label(target)}")
    if nowcast_ann >= 0:
        return f"US growth is tracking about {nowcast_ann:.1f}% {when}"
    return f"US output is on track to shrink about {abs(nowcast_ann):.1f}% {when}"


def news_sentence(items: list[dict], since: pd.Timestamp) -> str:
    """One line naming the biggest upward and downward contributions."""
    if not items:
        return f"No new data releases since {since:%-d %B}."
    ups = [i["label"] for i in sorted(items, key=lambda i: -i["impact"]) if i["impact"] > 0.005][:2]
    downs = [i["label"] for i in sorted(items, key=lambda i: i["impact"]) if i["impact"] < -0.005][:1]
    def lower(s):
        return s[0].lower() + s[1:] if not s.isupper() else s
    parts = []
    if ups:
        parts.append(" and ".join(lower(u) if k else u for k, u in enumerate(ups)) + " pushed it up")
    if downs:
        parts.append(lower(downs[0]) + " pulled it down")
    if not parts:
        return f"Since {since:%-d %B}, new data has barely moved the estimate."
    return f"Since {since:%-d %B}, " + lower("; ".join(parts)) + "."


def accuracy_sentence(model: float, bench: float, gdpnow: float) -> str:
    """Honest, data-driven verdict (overall RMSE, ex-2020, first-release target)."""
    if abs(model - bench) < 0.2:
        rel = "About as accurate as a simple benchmark"
    elif model < bench:
        rel = "More accurate than a simple benchmark"
    else:
        rel = "Less accurate than a simple benchmark"
    if gdpnow < model - 0.1:
        tail = "and behind the Atlanta Fed's GDPNow, which keeps improving as the quarter's data arrives."
    elif gdpnow > model + 0.1:
        tail = "and ahead of the Atlanta Fed's GDPNow."
    else:
        tail = "and level with the Atlanta Fed's GDPNow."
    return f"{rel} — {tail}"


# ── Data pieces ───────────────────────────────────────────────────────────────

def refresh_vintages(dates: list[pd.Timestamp]) -> None:
    """
    Re-fetch each series' revision history (one ALFRED call per series) and
    rebuild the vintages for `dates`. Today's files are rewritten so a second
    build on the same day sees releases published in between.
    """
    series = list(SERIES_CATALOG) + ["GDPNOW"]
    for sid in series:
        fetch_all_releases(sid, use_cache=False)
        for d in dates[-1:]:
            _alfred_cache_path(sid, str(d.date())).unlink(missing_ok=True)
    materialize_vintages([(s, str(d.date())) for d in dates for s in series])


def current_target(today: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(target quarter, date the previous quarter's GDP was first published)."""
    releases = fetch_all_releases("GDPC1")
    known = vintage_from_releases(releases, today, "GDPC1").dropna()
    last_q = pd.Timestamp(known.index[-1]).to_period("Q").to_timestamp()
    first_pub = releases.loc[releases["date"] == last_q, "realtime_start"].min()
    return last_q + pd.offsets.QuarterBegin(1, startingMonth=1), pd.Timestamp(first_pub)


def nowcast_path(target: pd.Timestamp, today: pd.Timestamp) -> list[dict]:
    """Real-time nowcasts of `target` through the quarter (no look-ahead)."""
    not_yet = today + pd.Timedelta(days=1)          # target's GDP is not out as of today
    dates = [d for d in horizon_evaluation_dates(target, not_yet) if d < today] + [today]
    refresh_vintages(dates)
    path = []
    for d in dates:
        r = run_single_quarter(target, str(d.date()), n_factors=2, release_date=not_yet)
        if not np.isfinite(r["dfm_nowcast_qtr"]):
            continue
        g = fetch_gdpnow_vintage(target, str(d.date()))
        path.append({
            "date": str(d.date()), "label": f"{d:%b} {d.day}",
            "nowcast": round(r["dfm_nowcast_qtr"] * 4, 3),
            "se": round(r["dfm_nowcast_se_ann"], 3),
            "gdpnow": None if pd.isna(g) else round(float(gdpnow_to_log_units(pd.Series([g])).iloc[0]), 3),
        })
    if not path:
        raise RuntimeError(f"No nowcast could be computed for {quarter_label(target)}.")
    return path


def news_and_factor(since: pd.Timestamp) -> tuple[list[dict], list[dict]]:
    """Per-series news impacts since `since`, and the quarterly global factor."""
    m, q = prepare_dfm_data(start="2000-01-01", use_cache=False)
    fb = make_block_structure(m, q)
    try:
        news = compute_news(before_end=str(since.date()), factor_blocks=fb, start="2000-01-01",
                            use_cache=True)
    except ValueError as exc:                     # e.g., no releases since `since`
        logger.warning("News unavailable: %s", exc)
        from src.model.dfm import fit_dfm
        return [], factor_series(fit_dfm(m, q, factor_blocks=fb))
    tidy = (news["impacts"].reset_index().groupby("updated variable")["impact"].sum() * 4)
    items = [{"label": NEWS_LABELS.get(s, s), "impact": round(float(v), 4)}
             for s, v in tidy.sort_values(ascending=False).items()]
    return items, factor_series(news["news_results"].updated)


def factor_series(results, today: pd.Timestamp | None = None) -> list[dict]:
    """Quarterly average of the global factor, signed so recessions are negative.
    The current, incomplete quarter is dropped (a partial average is misleading)."""
    f = results.factors.smoothed["global"]
    f.index = f.index.to_timestamp()
    if f.loc["2009-01-01":"2009-03-01"].mean() > 0:
        f = -f
    qf = f.resample("QS").mean().dropna()
    current = (today or pd.Timestamp.today()).to_period("Q").to_timestamp()
    qf = qf[qf.index < current]
    return [{"date": f"{t.year}-{t.month:02d}", "label": quarter_label(t), "value": round(float(v), 3)}
            for t, v in qf.items()]


def accuracy_bins(horizon_csv: Path) -> tuple[list[dict], dict]:
    """
    RMSE vs first-release GDP (ex-2020) in 15-day bins of days-before-release,
    for the model, GDPNow and the robust AR(1); bins with < 10 nowcasts → None.
    Also returns the overall RMSEs used for the verdict sentence.
    """
    df = pd.read_csv(horizon_csv)
    df = df[~df["is_2020"]]
    edges = np.arange(0, 136, 15)
    df["bin"] = pd.cut(df["days_to_release"], edges)
    cols = {"model": "dfm_nowcast_ann", "gdpnow": "gdpnow_nowcast_ann", "benchmark": "ar1_robust_nowcast_ann"}

    def rmse(g, col):
        e = (g[col] - g["first_release_ann"]).dropna()
        return round(float(np.sqrt((e ** 2).mean())), 3) if len(e) >= 10 else None

    rows = []
    for b, g in df.groupby("bin", observed=False):
        rows.append({"days": int((b.left + b.right) / 2), **{k: rmse(g, c) for k, c in cols.items()}})
    rows = [r for r in rows if r["model"] is not None][::-1]          # far → near the release
    overall = {k: rmse(df, c) for k, c in cols.items()}
    return rows, overall


# ── Assemble + render ─────────────────────────────────────────────────────────

def build_data(today: pd.Timestamp | None = None) -> dict:
    today = (today or pd.Timestamp.today()).normalize()
    target, last_release = current_target(today)
    path = nowcast_path(target, today)
    latest = path[-1]["nowcast"]
    news, factor = news_and_factor(last_release)
    acc, overall = accuracy_bins(ROOT / "data" / "horizon_backtest.csv")
    vals = [v for r in acc for v in (r["model"], r["gdpnow"], r["benchmark"]) if v is not None]
    expected = target + pd.offsets.QuarterEnd(0) + pd.Timedelta(days=GDP_ADVANCE_LAG_DAYS)
    return {
        "headline": headline(latest, target, today),
        "byline": f"{AUTHOR} · Updated {today:%-d %B %Y} · {quarter_label(target)} estimate",
        "latest": latest,
        "path": path,
        "hero_caption": (f"Annualized growth for {quarter_label(target)}. Each point uses only data "
                         f"published that day. First official estimate expected around "
                         f"{expected:%B %Y}."),
        "news": news,
        "news_line": news_sentence(news, last_release),
        "news_caption": ("Contribution of each data release to the change in the estimate since "
                         f"{last_release:%-d %B}, in percentage points of annualized growth."),
        "accuracy": acc,
        "acc_line": accuracy_sentence(overall["model"], overall["benchmark"], overall["gdpnow"]),
        "acc_caption": ("Typical error (root-mean-square, percentage points) against first-published "
                        "GDP, 2015–2025 excluding 2020, from 448 real-time estimates. Simple "
                        "benchmark: a robust autoregression on past GDP growth."),
        "acc_ymin": float(np.floor((min(vals) - 0.2) * 5) / 5),
        "acc_ymax": float(np.ceil((max(vals) + 0.2) * 5) / 5),
        "factor": factor,
        "recessions": recession_quarters(),
        "built": f"Built {pd.Timestamp.now(tz='UTC'):%-d %B %Y, %H:%M} UTC from data available that day.",
    }


def render(data: dict, out_path: Path, template: Path = TEMPLATE) -> Path:
    """Inject `data` into the template and write the page."""
    page = template.read_text()
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = page.replace("__DATA__", payload).replace("__DESCRIPTION__", html.escape(data["headline"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page)
    return out_path
