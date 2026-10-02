"""
Builds per-(player, tournament) history from real historical per-round
data (historical_scraper.py's output) — "how has THIS player actually
performed at THIS specific course," as distinct from both of the two
signals already in the model:

  - blend_years.py's player blend: this player's general skill level,
    averaged across every course they've played.
  - course_history.py's course profile: the FIELD's average difficulty
    at this course, applied the same way to every player.

Combined, those two are a main-effects-only model: "take this player's
general skill, shift it by however hard this course plays for anyone."
That assumes a hard course is equally hard for every player and ignores
"horses for courses" — some players genuinely over- or under-perform
their own general numbers at a specific venue, for reasons (shot shape
fit, course familiarity, whatever) the model doesn't need to know, only
needs to detect. That's a player*course INTERACTION term, and this
module measures it directly from real per-round results instead of
guessing it exists.

Method: for one or more past years of real per-round data, group by
(player, tournament) and compute that player's own average GIR%,
driving accuracy, scoring average, and birdie rate at that specific
event — plus a sample-size column, since this is the thing to be most
careful about here (see shrink_player_course_rate below).

Matching is name-based for both axes — normalize_tournament_name from
course_history.py for the event, normalize_name from match_props.py for
the player — so this reuses the exact same matching logic already
proven out for the course-level adjustment, rather than inventing a new
scheme.

CRITICAL caveat this module does NOT solve on its own: with 2-3 years of
real per-round data, most players have played a given tournament only
2-4 times (roughly one appearance per year, a few rounds each, fewer if
they missed the cut). That is a tiny sample to treat as a player's "true"
course-specific rate — nowhere near the ~50-150+ rounds backing a
course's own field-wide profile. Used naively, this would mean telling
the model "Player X always shoots 66 here" off of two good rounds that
could just as easily be noise. shrink_player_course_rate() exists
specifically to pull this back toward the player's own course-ADJUSTED
general rate (not the tour average — their own expected number at this
course, from the two signals that already exist) in proportion to how
thin the specific-course sample is, so only players with an unusually
large or unusually consistent track record at one course actually move
much.

Usage (as a library):
    from player_course_history import build_player_course_profiles, shrink_player_course_rate
    profiles = build_player_course_profiles([2024, 2023])
    profiles.loc[("jon rahm", "masters tournament")]   # -> a row of stats, if it exists
"""

import os
import pandas as pd

from course_history import normalize_tournament_name, ROUNDS_DIR
from match_props import normalize_name

PROFILE_COLS = ["gir_pct", "driving_accuracy", "scoring_avg", "birdie_rate_per_hole"]

# Suggested starting point for shrink_player_course_rate's k — see that
# function's docstring for the reasoning. Exported so backtest_rounds.py
# can use the same default rather than duplicating the number, and so
# --player-course-sweep has an obvious "current default" candidate to
# compare everything else against.
DEFAULT_PLAYER_COURSE_K = 12.0


def build_player_course_profiles(years: list) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by (player_key, tournament_key) with
    columns [gir_pct, driving_accuracy, scoring_avg,
    birdie_rate_per_hole, n_rounds] — this specific player's own average
    at this specific event, pooled across every real per-round row found
    for the given years.

    Same fail-soft behavior as course_history.build_course_profiles: a
    missing year's file is skipped with a warning rather than failing
    the whole thing, and an empty years list (or no files found at all)
    returns an empty table rather than erroring.
    """
    frames = []
    for year in years:
        path = f"{ROUNDS_DIR}/{year}_rounds.csv"
        if not os.path.exists(path):
            print(f"  {path} not found — skipping {year} for player-course "
                  f"history (run historical_scraper.py --year {year} to add it).")
            continue
        df = pd.read_csv(path)
        for col in ["round_strokes", "round_birdies_or_better", "round_gir", "round_fairways"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(df)

    if not frames:
        empty = pd.DataFrame(columns=["player_key", "tournament_key"] + PROFILE_COLS + ["n_rounds"])
        return empty.set_index(["player_key", "tournament_key"])

    all_rounds = pd.concat(frames, ignore_index=True)
    all_rounds["tournament_key"] = all_rounds["tournament_name"].apply(normalize_tournament_name)
    all_rounds["player_key"] = all_rounds["player_name"].apply(normalize_name)
    all_rounds = all_rounds[(all_rounds["tournament_key"] != "") & (all_rounds["player_key"] != "")]

    grouped = all_rounds.groupby(["player_key", "tournament_key"]).agg(
        gir_pct=("round_gir", lambda s: (s / 18).mean()),
        driving_accuracy=("round_fairways", lambda s: (s / 14).mean()),
        scoring_avg=("round_strokes", "mean"),
        birdie_rate_per_hole=("round_birdies_or_better", lambda s: (s / 18).mean()),
        n_rounds=("round_strokes", "size"),
    )
    return grouped


def shrink_player_course_rate(general_rate: float, player_course_rate: float,
                               n_rounds: int, k: float = DEFAULT_PLAYER_COURSE_K) -> float:
    """
    Blends a player's own course-specific real rate with their already-
    course-adjusted general rate, weighted by how many real rounds back
    the course-specific number.

        shrink_weight = n_rounds / (n_rounds + k)
        result = shrink_weight * player_course_rate + (1 - shrink_weight) * general_rate

    k is deliberately much larger than blend_years.py's shrink_k (0.3)
    — that one shrinks toward a population average using coverage
    measured in fractional recency-weighted seasons (typically close to
    1.0). This shrinks toward a single player's own expected number
    using a sample measured in raw ROUNDS, which is usually 2-8 for a
    player-course pair even across 3 years of data. A small k here would
    let 2-3 rounds swing a player's prediction a lot; k=12 means even a
    full 3-year, no-missed-cuts history (12 rounds: 4 rounds x 3 years)
    only gets 50% weight, and the much more typical 2-4 round history
    gets 15-25% weight — real signal when it exists, but not enough to
    let a hot week or two dominate. Tune with
    backtest_rounds.py --player-course-sweep rather than guessing harder.
    """
    if n_rounds <= 0:
        return general_rate
    shrink_weight = n_rounds / (n_rounds + k)
    return shrink_weight * player_course_rate + (1 - shrink_weight) * general_rate


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", required=True,
                         help="Prior years of real per-round data to pool "
                              "(e.g. --years 2024 2023).")
    parser.add_argument("--player", default=None,
                         help="Show only this player's rows (loose name match).")
    args = parser.parse_args()

    profiles = build_player_course_profiles(args.years)
    if profiles.empty:
        print("No player-course profiles built — none of the requested years "
              "have a data/historical_rounds/{year}_rounds.csv file.")
    else:
        print(f"Built {len(profiles)} player-course pairs from years {args.years}.")
        if args.player:
            key = normalize_name(args.player)
            matches = profiles[profiles.index.get_level_values("player_key").str.contains(key, na=False)]
            print(f"\nRows matching '{args.player}':\n")
            print(matches.sort_values("n_rounds", ascending=False).to_string())
        else:
            print("\nTop 20 by sample size (most rounds at one event):\n")
            print(profiles.sort_values("n_rounds", ascending=False).head(20).to_string())
