"""
Tests for src/site_builder.py — the generated text, the accuracy binning and
the HTML rendering. No API calls (the live build is exercised by the daily
GitHub Actions job).
"""
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from src.site_builder import (
    accuracy_bins,
    accuracy_sentence,
    headline,
    news_sentence,
    recession_quarters,
    render,
)

ROOT = Path(__file__).resolve().parents[1]


class TestHeadline:

    def test_current_quarter(self):
        assert headline(2.54, pd.Timestamp("2026-07-01"), pd.Timestamp("2026-08-15")) == \
            "US growth is tracking about 2.5% this quarter"

    def test_after_quarter_end_names_the_quarter(self):
        assert headline(2.54, pd.Timestamp("2026-07-01"), pd.Timestamp("2026-10-17")) == \
            "US growth is tracking about 2.5% in Q3 2026"

    def test_negative_growth(self):
        assert headline(-1.26, pd.Timestamp("2026-07-01"), pd.Timestamp("2026-08-15")) == \
            "US output is on track to shrink about 1.3% this quarter"


class TestSentences:

    def test_news_names_biggest_moves(self):
        items = [{"label": "Retail sales", "impact": 0.15}, {"label": "Consumer spending", "impact": 0.07},
                 {"label": "Payrolls", "impact": 0.01}, {"label": "Housing starts", "impact": -0.05}]
        s = news_sentence(items, pd.Timestamp("2026-07-30"))
        assert s == ("Since 30 July, retail sales and consumer spending pushed it up; "
                     "housing starts pulled it down.")

    def test_news_empty(self):
        assert news_sentence([], pd.Timestamp("2026-07-30")) == "No new data releases since 30 July."

    def test_news_negligible(self):
        s = news_sentence([{"label": "CPI", "impact": 0.001}], pd.Timestamp("2026-07-30"))
        assert "barely moved" in s

    @pytest.mark.parametrize("model,bench,gdpnow,start", [
        (1.84, 1.69, 1.22, "About as accurate"),
        (1.30, 1.69, 1.22, "More accurate"),
        (2.20, 1.69, 1.22, "Less accurate"),
    ])
    def test_accuracy_verdict_follows_the_numbers(self, model, bench, gdpnow, start):
        assert accuracy_sentence(model, bench, gdpnow).startswith(start)

    def test_accuracy_mentions_gdpnow_honestly(self):
        assert "behind the Atlanta Fed's GDPNow" in accuracy_sentence(1.84, 1.69, 1.22)
        assert "ahead of the Atlanta Fed's GDPNow" in accuracy_sentence(1.0, 1.69, 1.5)


class TestData:

    def test_accuracy_bins_from_committed_backtest(self):
        rows, overall = accuracy_bins(ROOT / "data" / "horizon_backtest.csv")
        assert rows[0]["days"] > rows[-1]["days"]                 # far → near the release
        assert all(r["model"] is not None for r in rows)
        assert 1.0 < overall["model"] < 3.0 and overall["gdpnow"] < overall["model"]

    def test_recession_quarters_cover_short_recessions(self):
        assert recession_quarters()[-1] == ["2020-01", "2020-04"]   # Feb–Apr 2020 spans two quarters


class TestRender:

    def test_injects_data_safely(self, tmp_path):
        data = {"headline": "Growth <b>&</b> more", "note": "</script><script>alert(1)"}
        out = render(data, tmp_path / "index.html")
        page = out.read_text()
        assert "__DATA__" not in page and "__DESCRIPTION__" not in page
        assert "</script><script>alert" not in page               # cannot break out of the script tag
        assert 'content="Growth &lt;b&gt;&amp;&lt;/b&gt; more"' in page
        payload = re.search(r"const DATA = (\{.*?\});\n", page, re.S).group(1)
        assert json.loads(payload.replace("<\\/", "</")) == data
