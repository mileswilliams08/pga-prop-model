"""
Runs the whole pipeline (scrape -> adjust for course -> compute prop
probabilities) and writes the result as JSON for the website to display.

This is the script GitHub Actions runs on a schedule. It's also totally
fine to run by hand:

    python update_data.py

Edit config.json before each tournament: update the course profile
numbers and, optionally, the `field` list to restrict output to players
actually in that week's field (leave it empty to include everyone
scraped).

Multi-year blending: set "use_blended_years": true in config.json (with
a "blend_years" list, e.g. [2025, 2024, 2023]) to rate players on a
weighted blend of multiple seasons instead of just the current one —
see blend_years.py for why this helps. Off by default since it makes
each run slower (multiple scrapes instead of one).
"""

import json
import sys
import pandas as pd

from features import clean_player_stats, build_player_course_profile
from prop_models import (gir_prop, fairways_prop, birdies_or_better_prop,
                          total_strokes_prop)
from match_props import attach_platform_lines

CONFIG_PATH = "config.json"
OUTPUT_PATH = "docs/data/props.json"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def get_clean_stats(cfg) -> pd.DataFrame:
    """
    Returns cleaned player stats (player, gir_pct, driving_accuracy,
    scoring_avg, birdie_rate_per_hole — all numeric). Tries live
    scraping first; falls back to the last saved CSV if scraping fails,
    so the site doesn't go blank on a bad run.
    """
    if cfg.get("use_blended_years"):
        try:
            from blend_years import scrape_and_blend_years
            years = cfg.get("blend_years", [cfg["year"]])
            blended = scrape_and_blend_years(years)
            blended.to_csv("data/latest_blended_stats.csv", index=False)
            return blended
        except Exception as e:
            print(f"Blended scrape failed ({e}), falling back to last "
                  f"saved blend CSV.")
            return pd.read_csv("data/latest_blended_stats.csv")

    try:
        from espn_scraper import scrape_espn_stats
        df = scrape_espn_stats(cfg["year"])
        df.to_csv("data/latest_stats.csv", index=False)
        return clean_player_stats(df)
    except Exception as e:
        print(f"Scrape failed ({e}), falling back to last saved CSV.")
        return clean_player_stats(pd.read_csv("data/latest_stats.csv"))


def build_props(cfg, profile: pd.DataFrame) -> list:
    lines = cfg["prop_lines"]
    rows = []
    for _, r in profile.iterrows():
        if pd.isna(r.get("adj_gir_pct")):
            continue  # skip players missing data
        rows.append({
            "player": r["player"],
            "gir": {
                "line": lines["gir"],
                **gir_prop(r["adj_gir_pct"], lines["gir"]),
            },
            "fairways": {
                "line": lines["fairways"],
                **fairways_prop(r["adj_driving_accuracy"], lines["fairways"]),
            },
            "birdies": {
                "line": lines["birdies"],
                **birdies_or_better_prop(r["adj_birdie_rate_per_hole"],
                                          lines["birdies"]),
            },
            "strokes": {
                "line": lines["strokes"],
                **total_strokes_prop(r["adj_scoring_avg"], lines["strokes"]),
            },
        })
    return rows


def get_platform_lines() -> dict:
    """Pulls current lines from both platforms; failures don't stop the run."""
    lines = {}
    try:
        from prizepicks_scraper import get_golf_props
        lines["prizepicks"] = get_golf_props()
    except Exception as e:
        print(f"PrizePicks fetch failed: {e}")
        lines["prizepicks"] = None
    try:
        from underdog_scraper import fetch_over_under_lines
        lines["underdog"] = fetch_over_under_lines()
    except Exception as e:
        print(f"Underdog fetch failed: {e}")
        lines["underdog"] = None
    return lines


def main():
    cfg = load_config()
    clean = get_clean_stats(cfg)

    if cfg.get("field"):
        clean = clean[clean["player"].isin(cfg["field"])]

    profile = build_player_course_profile(clean, cfg["course"], cfg["tour_avg"])
    props = build_props(cfg, profile)
    props = attach_platform_lines(props, get_platform_lines())

    # Sort so the most lopsided (highest-confidence) props float to the top.
    def max_confidence(p):
        return max(abs(p["gir"]["over"] - 0.5), abs(p["fairways"]["over"] - 0.5),
                   abs(p["birdies"]["over"] - 0.5), abs(p["strokes"]["over"] - 0.5))

    props.sort(key=max_confidence, reverse=True)

    output = {
        "tournament_name": cfg["tournament_name"],
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "props": props,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(props)} players' props to {OUTPUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
