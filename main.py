"""
End-to-end example. This uses realistic sample stats (season 2025-ish
numbers) instead of scraped data, so you can see the full pipeline work
right away. Once scraper.py gives you a real CSV, swap out `sample_data()`
for `pd.read_csv("data/2025_stats.csv")` + features.clean_player_stats().

Multi-year blending: instead of using just one season's stats, you can
blend multiple years (e.g. 2025 + 2024 + 2023, weighted toward the most
recent year) via blend_years.py — this smooths out noise from a small
sample of rounds played so far this season. See USE_BLENDED_YEARS below.
"""

import pandas as pd
from features import (clean_player_stats, build_player_course_profile)
from prop_models import (gir_prop, fairways_prop, birdies_or_better_prop,
                          total_strokes_prop)
from ev_calculator import evaluate_bet

# Set this to True to blend multiple years of stats instead of using a
# single season's CSV. The first run will scrape all three years (takes
# a few minutes); after that, main.py will re-read the file it saved
# unless you delete it or change YEARS_TO_BLEND.
USE_BLENDED_YEARS = True
YEARS_TO_BLEND = [2025, 2024, 2023]
BLENDED_CACHE_PATH = "data/blended_stats.csv"


def sample_data() -> pd.DataFrame:
    # Realistic-ish PGA Tour season stats for a few players.
    return pd.DataFrame([
        {"player": "Scottie Scheffler", "gir_pct": "72.1", "driving_accuracy": "68.5",
         "scoring_avg": 68.9, "birdie_avg": 4.6},
        {"player": "Xander Schauffele", "gir_pct": "69.8", "driving_accuracy": "63.2",
         "scoring_avg": 69.4, "birdie_avg": 4.1},
        {"player": "Average Tour Pro", "gir_pct": "65.0", "driving_accuracy": "60.5",
         "scoring_avg": 70.9, "birdie_avg": 3.5},
    ])


def tour_averages() -> dict:
    # Season-long tour-wide averages (rough real-world figures).
    return {
        "gir_pct": 0.65,
        "driving_accuracy": 0.605,
        "scoring_avg": 70.9,
        "birdie_rate_per_hole": 3.5 / 18,
    }


def course_profile_example() -> dict:
    # A tougher-than-average course: fewer GIR, fewer birdies, higher scores.
    return {
        "gir_pct": 0.60,
        "driving_accuracy": 0.58,
        "scoring_avg": 71.8,
        "birdie_rate_per_hole": 2.9 / 18,
    }


def get_player_stats() -> pd.DataFrame:
    """
    Returns already-cleaned player stats (columns: player, gir_pct,
    driving_accuracy, scoring_avg, birdie_rate_per_hole — all numeric,
    gir_pct/driving_accuracy as 0-1 fractions).

    Three ways to get here, in order of what most people will do:
      1. USE_BLENDED_YEARS=True: blend multiple seasons (scrapes on
         first run, then reads its own cached CSV after that).
      2. USE_BLENDED_YEARS=False + a real scraped CSV on disk: reads it
         and cleans it. This is what you'll use for a single season of
         real data — see the comment on the `raw = ...` line below.
      3. Neither of the above present: falls back to a few sample
         players so the pipeline always runs out of the box.
    """
    if USE_BLENDED_YEARS:
        import os
        if os.path.exists(BLENDED_CACHE_PATH):
            print(f"Loading cached blended stats from {BLENDED_CACHE_PATH}")
            return pd.read_csv(BLENDED_CACHE_PATH)
        else:
            from blend_years import scrape_and_blend_years
            print(f"No cached blend found — scraping and blending years "
                  f"{YEARS_TO_BLEND} (this takes a few minutes)...")
            blended = scrape_and_blend_years(YEARS_TO_BLEND)
            os.makedirs(os.path.dirname(BLENDED_CACHE_PATH) or ".", exist_ok=True)
            blended.to_csv(BLENDED_CACHE_PATH, index=False)
            print(f"Saved blend to {BLENDED_CACHE_PATH} for next time.")
            return blended

    # Single-season path: swap this line to read your real scraped CSV,
    # e.g. raw = pd.read_csv("data/2025_stats.csv")
    raw = sample_data()
    return clean_player_stats(raw)


def run():
    clean = get_player_stats()
    profile = build_player_course_profile(clean, course_profile_example(),
                                           tour_averages())

    print(profile[["player", "gir_pct", "adj_gir_pct", "scoring_avg",
                    "adj_scoring_avg"]].to_string(index=False))
    print()

    for _, row in profile.iterrows():
        print(f"--- {row['player']} ---")

        gir = gir_prop(row["adj_gir_pct"], line=12.5)
        print(f"  GIR over/under 12.5: over={gir['over']:.3f} under={gir['under']:.3f}")

        fw = fairways_prop(row["adj_driving_accuracy"], line=8.5)
        print(f"  Fairways over/under 8.5: over={fw['over']:.3f} under={fw['under']:.3f}")

        bb = birdies_or_better_prop(row["adj_birdie_rate_per_hole"], line=3.5)
        print(f"  Birdies-or-better over/under 3.5: over={bb['over']:.3f} under={bb['under']:.3f}")

        strokes = total_strokes_prop(row["adj_scoring_avg"], line=70.5)
        print(f"  Total strokes over/under 70.5: over={strokes['over']:.3f} under={strokes['under']:.3f}")

        # Example: sportsbook offers -115 on the "Over 3.5 birdies" prop
        bet_check = evaluate_bet(bb["over"], american_odds=-115)
        print(f"  EV check on 'Over 3.5 birdies' at -115: {bet_check}")
        print()


if __name__ == "__main__":
    run()
