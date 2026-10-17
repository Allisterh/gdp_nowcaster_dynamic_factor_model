"""
build_site.py — build the public page into site/index.html.

Run:  python scripts/build_site.py            (needs FRED_API_KEY)
Runs daily in .github/workflows/site.yml, which publishes site/ to GitHub Pages.
"""
import logging
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.site_builder import build_data, render  # noqa: E402

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")

if __name__ == "__main__":
    data = build_data()
    out = render(data, ROOT / "site" / "index.html")
    print(f"{data['headline']}  ({len(data['path'])} points, {len(data['news'])} news items)")
    print(f"Wrote {out}")
