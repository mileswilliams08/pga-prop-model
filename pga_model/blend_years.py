"""
Blends stats across multiple seasons into one weighted average per
player, so a player's rating isn't based purely on however many rounds
they've played so far this season (which is noisy early in the year, or
for anyone who's missed events with injury, etc.).

Default weighting favors the most recent season heavily but still lets
prior years smooth things out:
    most recent season : 60%
    one year back       : 25%
    two years back       : 15%

If a player is missing from an older season (rookie, or wasn't on tour
yet), their weights are renormalized across whichever years they DO
have data for — so a rookie with only 2025 data still gets a rating
from 2025 alone (effective weight 1.0), rather than being penalized for
lacking history.

This is deliberately season-level blending, not course-history
blending — it smooths a player's general skill estimate, not their fit
for a specific course. A course-history model (e.g. "how has this
player scored at Augusta specifically") would need per-tournament
historical results, which is a bigger scraping project — see the
README for that distinction.

Usage (standalone):
    python blend_years.py --years 2025 2024 2023

Usage (as a library, from update_data.py or main.py):
    from blend_years import scrape_and_blend_years
    blended = scrape_and_blend_years([2025, 2024, 2023])
"""

import argparse
import pandas as pd

from features import clean_player_stats

DEFAULT_WEIGHTS_BY_RECENCY = [0.60, 0.25, 0.15]  # most recent year listed first

# The stats blending actually averages. Blending happens on CLEANED,
# numeric columns (post clean_player_stats) so units are consistent —
# raw scraped data has gir_pct as a percentage string like "68.4" and
# birdie_avg (per round) rather than birdie_rate_per_hole, so blending
# raw data directly would silently average incompatible units.
STAT_COLS = ["gir_pct", "driving_accuracy", "scoring_avg", "birdie_rate_per_hole"]


def blend_multi_year_stats(cleaned_dfs_by_year: dict,
                            weights_by_recency: list = None) -> pd.DataFrame:
    """
    cleaned_dfs_by_year: {year: cleaned_df}, where each cleaned_df has
        already been through features.clean_player_stats (numeric
        columns, bad/zero-stat rows already filtered).
    weights_by_recency: weights ordered most-recent-year-first. Defaults
        to DEFAULT_WEIGHTS_BY_RECENCY, truncated/extended to match how
        many years were actually provided.

    Returns a DataFrame with columns [player, gir_pct, driving_accuracy,
    scoring_avg, birdie_rate_per_hole] — the same shape
    build_player_course_profile() expects as input.
    """
    years_sorted = sorted(cleaned_dfs_by_year.keys(), reverse=True)
    if not years_sorted:
        return pd.DataFrame(columns=["player"] + STAT_COLS)

    weights = list((weights_by_recency or DEFAULT_WEIGHTS_BY_RECENCY)[:len(years_sorted)])
    # Pad with a small fallback weight if more years were given than
    # the default list covers (e.g. 4+ years of history).
    while len(weights) < len(years_sorted):
        weights.append(0.05)

    all_players = set()
    for df in cleaned_dfs_by_year.values():
        all_players.update(df["player"])

    rows = []
    for player in all_players:
        weighted_sum = {c: 0.0 for c in STAT_COLS}
        weight_total = {c: 0.0 for c in STAT_COLS}

        for year, w in zip(years_sorted, weights):
            df = cleaned_dfs_by_year[year]
            match = df[df["player"] == player]
            if match.empty:
                continue
            row = match.iloc[0]
            for c in STAT_COLS:
                if c in row.index and pd.notna(row[c]):
                    weighted_sum[c] += row[c] * w
                    weight_total[c] += w

        # Only keep the player if every stat has at least some coverage
        # (i.e. they appear in at least one season with that stat).
        if all(weight_total[c] > 0 for c in STAT_COLS):
            record = {"player": player}
            for c in STAT_COLS:
                record[c] = weighted_sum[c] / weight_total[c]
            rows.append(record)

    return pd.DataFrame(rows)


def scrape_and_blend_years(years: list, weights_by_recency: list = None,
                            debug: bool = False) -> pd.DataFrame:
    """
    Convenience wrapper: scrapes each requested year from ESPN, cleans
    each one, blends them, and returns the final blended DataFrame ready
    for features.build_player_course_profile().

    A single year's scrape failing doesn't kill the whole run — it's
    just excluded from the blend (with a printed warning), same
    fail-soft philosophy as the rest of the pipeline.
    """
    from espn_scraper import scrape_espn_stats

    cleaned_by_year = {}
    for year in years:
        try:
            raw = scrape_espn_stats(year, debug=debug)
            cleaned = clean_player_stats(raw)
            cleaned_by_year[year] = cleaned
            print(f"  {year}: {len(cleaned)} players with usable stats")
        except Exception as e:
            print(f"  {year}: scrape failed ({e}) — excluded from blend")

    if not cleaned_by_year:
        raise RuntimeError("Every year's scrape failed — nothing to blend.")

    return blend_multi_year_stats(cleaned_by_year, weights_by_recency)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", default=[2025, 2024, 2023],
                         help="Years to scrape and blend, most recent first "
                              "(default: 2025 2024 2023)")
    parser.add_argument("--out", type=str, default="data/blended_stats.csv")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"Scraping and blending years: {args.years}")
    blended = scrape_and_blend_years(args.years, debug=args.debug)

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    blended.to_csv(args.out, index=False)
    print(f"Saved {len(blended)} blended players to {args.out}")
