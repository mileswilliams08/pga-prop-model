"""
Runs the whole pipeline (scrape -> adjust for course -> compute prop
probabilities) and writes the result as JSON for the website to display.

This is the script GitHub Actions runs on a schedule. It's also totally
fine to run by hand:

    python update_data.py

Edit config.json before each tournament: set `tournament_name` to that
week's event and, optionally, the `field` list to restrict output to
players actually in that week's field (leave it empty to include
everyone scraped).

Course adjustment: if `course_history_years` is set (e.g. [2025, 2024]),
get_course_profile() looks up this week's `tournament_name` in REAL
per-round course difficulty from those years
(data/historical_rounds/{year}_rounds.csv, from historical_scraper.py)
and uses that instead of the hand-typed `course` dict — backtesting
found this matters far more than anything else tuned in this project
(see get_course_profile()'s docstring). The hand-typed `course` dict in
config.json is kept as a fallback for events with no matching scraped
history yet (a new event, or a name that didn't match across years).

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
from course_history import build_course_profiles, normalize_tournament_name

CONFIG_PATH = "config.json"
OUTPUT_PATH = "docs/data/props.json"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def get_course_profile(cfg) -> dict:
    """
    Looks up this week's tournament (cfg["tournament_name"]) in REAL
    course history built from past per-round data
    (data/historical_rounds/{year}_rounds.csv, from historical_scraper.py)
    instead of using config.json's hand-typed "course" numbers.

    Backtesting (backtest_rounds.py --course-adjust) found this is a much
    bigger lever than anything else tuned in this project: GIR +9%,
    Fairways +13%, Birdies +4%, Strokes +7% vs. the climatology baseline,
    compared to under 1% from phi/shrinkage/bias tuning alone — flat,
    hand-typed course numbers were leaving the single largest source of
    round-to-round variance (how hard the course actually plays) on the
    table.

    Falls back to cfg["course"] (the manual numbers) when there's no
    scraped history to match against — a new event, a name that doesn't
    match across years (see course_history.normalize_tournament_name),
    or cfg["course_history_years"] not set yet. This keeps the pipeline
    working the old way for anything not covered by real data yet,
    rather than failing or silently using an unadjusted tour average.
    """
    years = cfg.get("course_history_years", [])
    if not years:
        print("No course_history_years set in config.json — using the "
              "hand-typed 'course' numbers.")
        return cfg["course"]

    profiles = build_course_profiles(years)
    key = normalize_tournament_name(cfg["tournament_name"])

    if key in profiles.index:
        row = profiles.loc[key]
        print(f"Using real course history for '{cfg['tournament_name']}' "
              f"({int(row['n_rounds'])} real rounds from {years}).")
        return {
            "gir_pct": row["gir_pct"],
            "driving_accuracy": row["driving_accuracy"],
            "scoring_avg": row["scoring_avg"],
            "birdie_rate_per_hole": row["birdie_rate_per_hole"],
        }

    print(f"No real course history found for '{cfg['tournament_name']}' "
          f"(tried years {years}) — falling back to the hand-typed "
          f"'course' numbers in config.json.")
    return cfg["course"]


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

    course = get_course_profile(cfg)
    profile = build_player_course_profile(clean, course, cfg["tour_avg"])
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
