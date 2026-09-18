"""
Scrapes PGA Tour season stats from ESPN instead of pgatour.com.

Why ESPN: it has all four stats we need (scoring average, driving
accuracy, GIR%, birdies per round) on ONE page, instead of 9 separate
page loads like PGA Tour required — meaning 1 request instead of 9,
which is much less likely to trigger bot detection.

The table still needs JavaScript to render (confirmed: plain requests
returns no data), so this still uses Playwright. Column parsing reads
the actual header row and matches by label text (SCORE, DACC, GIR,
BIRDS, PLAYER) rather than assuming a fixed position — this way, if
ESPN's column order differs from what we expect, it won't silently put
the wrong numbers in the wrong columns.

Usage:
    python espn_scraper.py --year 2025
    python espn_scraper.py --year 2025 --debug   # prints header mapping found
"""

import argparse
import time
import pandas as pd
from playwright.sync_api import sync_playwright

# Maps ESPN's column headers to our model's field names.
HEADER_MAP = {
    "SCORE": "scoring_avg",
    "DACC": "driving_accuracy",
    "GIR": "gir_pct",
    "BIRDS": "birdie_avg",
}


def scrape_espn_stats(year: int, debug: bool = False) -> pd.DataFrame:
    url = f"https://www.espn.com/golf/stats/player/_/season/{year}/table/general"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        )
        page.goto(url, wait_until="domcontentloaded", timeout=45000)
        page.wait_for_selector("table tbody tr", timeout=30000)
        time.sleep(1.5)

        # ESPN typically loads only ~50 rows initially and reveals more via
        # a "Show More" button rather than a different URL. Click it
        # repeatedly until the row count stops growing (or we hit a
        # sanity-check cap of 15 clicks, which comfortably covers a full
        # ~200-player field at 50 rows/click).
        def current_row_count():
            tables = page.query_selector_all("table")
            if len(tables) < 2:
                return 0
            return len(tables[1].query_selector_all("tbody tr"))

        for click_num in range(15):
            before = current_row_count()
            show_more = page.locator("text=/show more/i").first
            if show_more.count() == 0:
                if debug:
                    print(f"No 'Show More' button found after {click_num} click(s).")
                break
            try:
                show_more.click(timeout=5000)
            except Exception:
                break
            page.wait_for_timeout(1200)
            after = current_row_count()
            if debug:
                print(f"Click {click_num + 1}: rows {before} -> {after}")
            if after <= before:
                break  # no new rows loaded, we're done

        # ESPN splits this into two side-by-side tables: a frozen
        # "name" table (RK, PLAYER, AGE) and a separate scrollable
        # "stats" table (EARNINGS...BIRDS). We read both and match rows
        # by position since they scroll in lockstep.
        tables = page.query_selector_all("table")
        if debug:
            print(f"Found {len(tables)} table(s) on the page.")

        if len(tables) < 2:
            raise RuntimeError(
                f"Expected 2 tables (names + stats), found {len(tables)}. "
                "ESPN's layout may have changed."
            )

        name_table, stats_table = tables[0], tables[1]

        # Player names: usually inside a link in the name table's rows.
        name_rows = name_table.query_selector_all("tbody tr")
        player_names = []
        for row in name_rows:
            link = row.query_selector("a")
            if link:
                player_names.append(link.inner_text().strip())
            else:
                player_names.append(row.inner_text().strip())
        if debug:
            print(f"First 5 names found: {player_names[:5]}")

        # Build column-name -> index map from the stats table's header row.
        header_cells = stats_table.query_selector_all("thead tr")[-1].query_selector_all("th")
        header_labels = [h.inner_text().strip() for h in header_cells]
        if debug:
            print(f"Stats header row found: {header_labels}")

        col_index = {label: i for i, label in enumerate(header_labels)}
        missing = [k for k in HEADER_MAP if k not in col_index]
        if missing:
            print(f"WARNING: expected columns not found: {missing}. "
                  f"ESPN may have changed their layout — check --debug output.")

        stat_rows = stats_table.query_selector_all("tbody tr")
        if len(stat_rows) != len(player_names):
            print(f"WARNING: name table has {len(player_names)} rows but "
                  f"stats table has {len(stat_rows)} — row matching may be off.")

        records = []
        for name, row in zip(player_names, stat_rows):
            cells = row.query_selector_all("td")
            if len(cells) < len(header_labels):
                continue

            record = {"player": name}
            for espn_label, our_name in HEADER_MAP.items():
                if espn_label in col_index:
                    record[our_name] = cells[col_index[espn_label]].inner_text().strip()

            if record.get("player"):
                records.append(record)

        browser.close()

    df = pd.DataFrame(records).drop_duplicates(subset="player")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    df = scrape_espn_stats(args.year, debug=args.debug)
    out_path = args.out or f"data/{args.year}_stats.csv"
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} players to {out_path}")
