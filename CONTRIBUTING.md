# Contributing / working on this project

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python 3.11+
pip install -r requirements.txt
cp .env.example .env    # add your free FRED API key; never commit .env or print the key
```

## Tests

```bash
pytest -q -m "not integration"   # unit tests: no API key, no network (what CI runs)
pytest -q                        # + integration tests that fit real models on FRED/ALFRED data
```

Integration tests skip themselves when `FRED_API_KEY` is not set. Run the full
suite before changing anything in `src/model/` or `src/backtest/`.

## Running the backtest

Order matters — each step reads the previous step's output:

| Step | Command | Writes |
|---|---|---|
| 1 | `python scripts/run_phase4_backtest.py` | `data/backtest_results.csv`, `docs/figures/phase4_backtest.png` |
| 2 | `python scripts/run_phase4_backtest_2factor.py` | `data/backtest_results_2factor.csv`, `docs/figures/phase4_backtest_2factor.png` |
| 3 | `python scripts/rerun_failed_quarters.py` | only if steps 1–2 report failed quarters |
| 4 | `python scripts/evaluate_backtest.py` | `data/evaluation_accuracy.csv`, `data/evaluation_tests.csv` |
| 5 | `python scripts/precompute_news_example.py` | `data/news_example.csv`, `data/news_example_meta.json` |
| 6 | `streamlit run app/streamlit_app.py --server.port 8599`, then `python scripts/screenshot_tabs.py after_redesign` | `docs/figures/after_redesign_tab{1-4}.png` |

Through-the-quarter evaluation (after step 2):

| Step | Command | Writes |
|---|---|---|
| 7 | `python scripts/run_horizon_backtest.py` | `data/horizon_backtest.csv` (448 nowcasts, ~15 min, resumable) |
| 8 | `python scripts/evaluate_horizon.py` | `data/horizon_*.csv`, `data/calibration*.csv`, `docs/figures/horizon_rmse.png`, `fan_chart_example.png`, `pit_histogram.png` |

Step 7 fetches each series' revision history once (`data/alfred/*_all_releases.parquet`)
and rebuilds every needed vintage locally. Delete those files to refresh them.

Optional diagnostic: `python scripts/diagnose_information_timing.py`.

## The public page

`python scripts/build_site.py` writes `site/index.html` (about a minute; needs
`FRED_API_KEY`). Preview it with `python -m http.server -d site 8765` and open
http://localhost:8765. The page itself is `site_template/index.html`; all numbers
and sentences come from `src/site_builder.py`. GitHub Actions (`site.yml`)
rebuilds and publishes it daily.

**Data and caching.** Each backtest quarter uses the ALFRED vintage from the
last day of that quarter (12 series × 44 quarters ≈ 530 files). The first run
downloads them with throttling and automatic back-off for FRED's rate limit
(a few minutes); they are cached in `data/alfred/` and never change, so later
runs take about a minute. `data/raw/` caches current FRED data. Both folders
are gitignored; only the small result files listed in `.gitignore` are
committed, because the deployed dashboard reads them.

**After a rerun** update the numbers in `README.md` (abstract, §3) and add a
dated entry to `docs/progress.md`. The dashboard reads the CSVs directly.

## Ground rules for results

- **No look-ahead.** Backtest inputs come from ALFRED vintages as of the
  evaluation date; never mix in current-vintage data.
- **Report, don't select.** Report ex-2020 *and* full-sample results, against
  both scoring targets. Don't change the model, sample or benchmarks because of
  how they score on this backtest — that is fitting to the test set. If a
  change is worth testing, decide on it first, then report it whatever it shows.
- **No silent fallbacks.** If something can't be computed, raise or log a
  warning; a plausible-looking wrong number is worse than an error.
