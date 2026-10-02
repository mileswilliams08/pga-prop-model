"""
Builds per-tournament "course difficulty" profiles from real historical
per-round data (historical_scraper.py's output), so predictions can be
adjusted for the SPECIFIC course a tournament is played at instead of
using one fixed line/rate across the whole season.

Why this exists: backtest_rounds.py found GIR, Birdies-or-better, and
Strokes all sitting at essentially the same accuracy as "guess the
season average" across two full seasons (2024, 2025) — the model had
no course-specific signal, so a flat line applied to every tournament
can't beat a population-wide average no matter how good the player
stats underneath it are. Fairways was the one stat with a real,
repeatable edge, and even that didn't vary by course. This module is
the fix: real course difficulty, measured from real outcomes, instead
of config.json's hand-typed per-tournament numbers.

Method: for one or more PAST years of real per-round data, group by
tournament and compute the field-wide average GIR%, driving accuracy,
scoring average, and birdie rate at that specific event — the same
shape config.json's "course" dict already expects
(features.build_player_course_profile), but measured instead of
guessed.

Matching tournaments across years is NAME-based, not ID-based: a
sponsor's name attached to an event changes some years (e.g. "...
presented by Workday"), and tournament_id hasn't been confirmed to
share a stable cross-year suffix in the data collected so far. An
unmatched tournament (a new event, or a name that drifted too far to
match) simply gets no course adjustment — fail-soft, same philosophy
as the rest of this project, rather than erroring out or guessing.

Usage (as a library):
    from course_history import build_course_profiles
    profiles = build_course_profiles([2024])      # one prior year
    profiles.loc["sentry"]                        # -> a row of stats
"""

import os
import re
import pandas as pd

ROUNDS_DIR = "data/historical_rounds"

PROFILE_COLS = ["gir_pct", "driving_accuracy", "scoring_avg", "birdie_rate_per_hole"]


def normalize_tournament_name(name) -> str:
    """
    'the Memorial Tournament presented by Workday' -> 'memorial tournament'
    'Arnold Palmer Invitational presented by Mastercard' -> 'arnold palmer invitational'

    Cuts everything from "presented by" onward (where sponsor churn
    lives), strips punctuation, lowercases, and drops a leading "the".
    Not bulletproof against every possible rename, but handles the
    common sponsor-suffix case seen across the 2024/2025 schedules.
    """
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = re.split(r"\bpresented by\b", name)[0]
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r"^the\s+", "", name)
    return name


def build_course_profiles(years: list) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by normalized tournament name with
    columns [gir_pct, driving_accuracy, scoring_avg,
    birdie_rate_per_hole, n_rounds] — the field-wide average at that
    specific event, pooled across every real per-round row found for
    the given years (data/historical_rounds/{year}_rounds.csv).

    Pools raw rows rather than averaging each year's per-tournament
    average, so an event with more recorded rounds counts more, and a
    year that's missing entirely (no historical_scraper.py run for it
    yet) is skipped with a warning instead of failing the whole thing —
    the caller gets whatever years ARE available, even if that's zero
    (an empty profile table, which callers should treat as "no course
    adjustment available" rather than an error).
    """
    frames = []
    for year in years:
        path = f"{ROUNDS_DIR}/{year}_rounds.csv"
        if not os.path.exists(path):
            print(f"  {path} not found — skipping {year} for course history "
                  f"(run historical_scraper.py --year {year} to add it).")
            continue
        df = pd.read_csv(path)
        for col in ["round_strokes", "round_birdies_or_better", "round_gir", "round_fairways"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=PROFILE_COLS + ["n_rounds"])

    all_rounds = pd.concat(frames, ignore_index=True)
    all_rounds["tournament_key"] = all_rounds["tournament_name"].apply(normalize_tournament_name)
    all_rounds = all_rounds[all_rounds["tournament_key"] != ""]

    grouped = all_rounds.groupby("tournament_key").agg(
        gir_pct=("round_gir", lambda s: (s / 18).mean()),
        driving_accuracy=("round_fairways", lambda s: (s / 14).mean()),
        scoring_avg=("round_strokes", "mean"),
        birdie_rate_per_hole=("round_birdies_or_better", lambda s: (s / 18).mean()),
        n_rounds=("round_strokes", "size"),
    )
    return grouped


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", required=True,
                         help="Prior years of real per-round data to pool "
                              "(e.g. --years 2024 2023).")
    args = parser.parse_args()

    profiles = build_course_profiles(args.years)
    if profiles.empty:
        print("No course profiles built — none of the requested years "
              "have a data/historical_rounds/{year}_rounds.csv file.")
    else:
        print(f"Built {len(profiles)} tournament course profiles from years {args.years}:\n")
        print(profiles.sort_values("n_rounds", ascending=False).to_string())
