"""
PGA Tour stats scraper.

PGA Tour's stat pages (pgatour.com/stats/...) render their tables with
JavaScript after the page loads, so plain requests+BeautifulSoup gets you
an empty shell. This script uses Playwright to drive a real (headless)
browser, wait for the table to render, then extract it.

RUN THIS ON YOUR OWN COMPUTER, not in a cloud sandbox — PGA Tour's site
may rate-limit or block traffic from cloud/datacenter IPs.

Setup (one time):
    pip install playwright pandas
    playwright install chromium

Usage:
    python scraper.py --year 2025
    python scraper.py --year 2025 --out data/2025_stats.csv
"""

import argparse
import time
import pandas as pd
from playwright.sync_api import sync_playwright

# Stat IDs on pgatour.com. These map to specific stat pages.
# (Verify these still work before relying on them — PGA Tour renumbers
# stats occasionally. Search "pgatour.com/stats/detail/<id>" to confirm.)
STAT_IDS = {
    "gir_pct": "103",             # Greens in Regulation %
    "driving_accuracy": "102",    # Fairways Hit %
    "scoring_avg": "120",         # Scoring Average
    "birdie_avg": "156",          # Birdie Average
    "sg_total": "02675",          # Strokes Gained: Total
    "sg_off_tee": "02567",        # Strokes Gained: Off-the-Tee
    "sg_approach": "02568",       # Strokes Gained: Approach
    "sg_around_green": "02569",   # Strokes Gained: Around-the-Green
    "sg_putting": "02564",        # Strokes Gained: Putting
}


def scrape_stat_table(page, stat_id: str, year: int) -> pd.DataFrame:
    """Load one stat page and pull the player/value table."""
    url = f"https://www.pgatour.com/stats/detail/{stat_id}"
    page.goto(url, wait_until="networkidle", timeout=30000)

    # The table is rendered client-side; give it a moment and wait for rows.
    page.wait_for_selector("table tbody tr", timeout=15000)
    time.sleep(1.0)  # let pagination/lazy content settle

    rows = page.query_selector_all("table tbody tr")
    records = []
    for row in rows:
        cells = [c.inner_text().strip() for c in row.query_selector_all("td")]
        if len(cells) >= 3:
            # Typical layout: [Rank, Player, Value, ...]
            records.append({"player": cells[1], "value": cells[2]})

    return pd.DataFrame(records)


def scrape_all(year: int) -> pd.DataFrame:
    merged = None
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        )

        for stat_name, stat_id in STAT_IDS.items():
            print(f"Scraping {stat_name} ({stat_id})...")
            try:
                df = scrape_stat_table(page, stat_id, year)
                df = df.rename(columns={"value": stat_name})
                merged = df if merged is None else merged.merge(
                    df, on="player", how="outer"
                )
            except Exception as e:
                print(f"  Failed on {stat_name}: {e}")
            time.sleep(2)  # be polite between requests

        browser.close()

    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    df = scrape_all(args.year)
    out_path = args.out or f"data/{args.year}_stats.csv"
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} players to {out_path}")
