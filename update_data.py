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

Live in-week recalibration: apply_live_conditions() (live_course_conditions.py)
further adjusts the course profile above using THIS WEEK'S real,
already-completed rounds (from data/espn_cumulative_snapshots.json,
written daily by update_results.py), blended in by how many rounds of
this week's data exist so far. This exists because a course can play
meaningfully easier or harder than its multi-year history in any given
week (weather, setup) — see live_course_conditions.py's docstring for
the real example that motivated it. Optional config.json key:
"live_conditions_k" to override the default shrinkage constant. Fails
soft to the historical-only profile if no rounds have finished yet.
"""

import json
import os
import sys
import pandas as pd

from features import (clean_player_stats, build_player_course_profile,
                       add_player_course_history_adjustment)
from prop_models import (gir_prop, fairways_prop, birdies_or_better_prop,
                          total_strokes_prop)
from match_props import attach_platform_lines, build_line_lookup, normalize_name
from course_history import build_course_profiles, normalize_tournament_name
from live_course_conditions import apply_live_conditions
from prop_models import (expected_pars_bogeys, pars_prop, bogeys_prop,
                         DEFAULT_PARS_LINE, DEFAULT_BOGEYS_LINE)

CONFIG_PATH = "config.json"
OUTPUT_PATH = "docs/data/props.json"
PROPS_HISTORY_DIR = "data/props_history"


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


PROP_SPECS = {
    "gir": ("adj_gir_pct", gir_prop),
    "fairways": ("adj_driving_accuracy", fairways_prop),
    "birdies": ("adj_birdie_rate_per_hole", birdies_or_better_prop),
    "strokes": ("adj_scoring_avg", total_strokes_prop),
}


# EXPERIMENTAL props derived from scoring + birdie rate and the course's par
# (see prop_models.py). Tracked on the results page but kept out of headline
# numbers until they've proven themselves against real PrizePicks lines.
EXPERIMENTAL_SPECS = {"pars": pars_prop, "bogeys": bogeys_prop}
EXPERIMENTAL_DEFAULT_LINES = {"pars": DEFAULT_PARS_LINE, "bogeys": DEFAULT_BOGEYS_LINE}


def get_course_par(cfg):
    """
    Course par for the week: config.json's "course_par" if set, else read
    from ESPN's posted hole pars (only available once a round has been
    played), else None -- in which case Pars / Bogeys-or-Worse are simply
    skipped this run rather than priced against a guessed par.
    """
    if cfg.get("course_par"):
        return int(cfg["course_par"])
    try:
        from datetime import datetime, timedelta, timezone
        from espn_round_results import find_event_id, get_field, infer_course_par
        now = datetime.now(timezone.utc)
        event_id = find_event_id((now - timedelta(days=4)).strftime("%Y%m%d"),
                                 (now + timedelta(days=4)).strftime("%Y%m%d"),
                                 tournament_name=cfg["tournament_name"])
        par = infer_course_par(event_id, cfg["year"], get_field(event_id))
        if par:
            print(f"Course par {par} (read from ESPN hole data).")
        return par
    except Exception as e:
        print(f"Could not infer course par from ESPN ({e}).")
        return None


def build_props(cfg, profile: pd.DataFrame, line_lookup: dict = None,
                course_par=None) -> list:
    """
    For each player/stat, computes the model's probability against the
    REAL line a platform actually posted for that specific player
    (line_lookup, from match_props.build_line_lookup — PrizePicks
    preferred, Underdog as fallback) when one exists; otherwise falls
    back to config.json's generic "prop_lines" default, same as before
    any real per-player lines existed.

    Without line_lookup, every player's probability used to be computed
    against the SAME fixed line regardless of player (e.g. birdies=3.5
    for everyone) — so a strong player at a birdie-friendly week, posted
    at a PrizePicks line of 5.5, would still show a probability for
    clearing 3.5, a basically unrelated question, right next to that 5.5
    platform line on the site. Each stat dict now also carries "source"
    (the platform the line came from, or None when it's the generic
    config default) so the UI can tell which is which.
    """
    default_lines = cfg["prop_lines"]
    line_lookup = line_lookup or {}
    rows = []
    for _, r in profile.iterrows():
        if pd.isna(r.get("adj_gir_pct")):
            continue  # skip players missing data
        player_lines = line_lookup.get(normalize_name(r["player"]), {})
        entry = {"player": r["player"]}
        for category, (col, prop_func) in PROP_SPECS.items():
            match = player_lines.get(category)
            line = match["line"] if match else default_lines[category]
            source = match["source"] if match else None
            entry[category] = {
                "line": line,
                "source": source,
                # Only standard (two-sided) lines are ever used as a real line
                # -- see match_props.build_line_lookup. Recorded so grading can
                # tell new snapshots (labeled) from older unlabeled ones.
                **({"odds_type": match.get("odds_type", "standard")} if match else {}),
                **prop_func(r[col], line),
            }
        if course_par is not None and not pd.isna(r.get("adj_scoring_avg")) \
                and not pd.isna(r.get("adj_birdie_rate_per_hole")):
            exp_pars, exp_bogeys = expected_pars_bogeys(
                r["adj_scoring_avg"], r["adj_birdie_rate_per_hole"], course_par)
            for category, prop_func in EXPERIMENTAL_SPECS.items():
                match = player_lines.get(category)
                line = match["line"] if match else EXPERIMENTAL_DEFAULT_LINES[category]
                entry[category] = {
                    "line": line,
                    "source": match["source"] if match else None,
                    **({"odds_type": match.get("odds_type", "standard")} if match else {}),
                    "experimental": True,
                    "expected": round(float(exp_pars if category == "pars" else exp_bogeys), 2),
                    **prop_func(exp_pars if category == "pars" else exp_bogeys, line),
                }
        rows.append(entry)
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


def archive_props_snapshot(output: dict, cfg: dict) -> None:
    """
    Saves a dated copy of today's props BEFORE it's overwritten tomorrow,
    so grade_results.py has something to compare real results against
    later. docs/data/props.json only ever holds the latest snapshot — a
    results tab tracking hit rate needs to know what the model actually
    said on each PAST day, not just today.

    Keyed by tournament + the UTC calendar date this snapshot was
    generated (not round number — we don't reliably know which round is
    "live" at generation time, just which day it is). One file per
    tournament-day is enough: update_data.py already only runs once a
    day via the scheduled Action, so this never needs to distinguish
    multiple runs on the same day.

    Lives in data/ (not docs/) since it's grading input, not something
    the site needs to serve directly.
    """
    os.makedirs(PROPS_HISTORY_DIR, exist_ok=True)
    date_str = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    tournament_slug = normalize_tournament_name(cfg["tournament_name"]).replace(" ", "_")
    path = f"{PROPS_HISTORY_DIR}/{tournament_slug}_{date_str}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Archived today's props snapshot to {path}")


def count_real_lines(props: list) -> int:
    """Number of (player, stat) entries whose line came from a real platform."""
    return sum(1 for p in props for cat in PROP_SPECS
               if (p.get(cat) or {}).get("source") is not None)


def load_previous_output():
    """The props.json currently on disk, or None if missing/unreadable."""
    try:
        with open(OUTPUT_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def main(allow_no_lines: bool = False):
    cfg = load_config()
    clean = get_clean_stats(cfg)

    if cfg.get("field"):
        clean = clean[clean["player"].isin(cfg["field"])]

    course = get_course_profile(cfg)

    # Recalibrates the course profile above using THIS WEEK'S real,
    # in-progress field results (if any rounds have finished yet) instead
    # of relying purely on the multi-year historical average -- see
    # live_course_conditions.py's docstring for why (a course can play
    # meaningfully easier/harder than its history in any given week, and
    # that's invisible to the historical-only baseline until now). Fails
    # soft to `course` unchanged if no live data is available yet.
    course = apply_live_conditions(course, cfg, debug=True)

    profile = build_player_course_profile(clean, course, cfg["tour_avg"])

    # Layers each individual player's own real history at THIS specific
    # tournament on top of the course-level adjustment above (e.g. a
    # player who's genuinely a strong fit for this course and has the
    # rounds to prove it) -- see add_player_course_history_adjustment's
    # docstring for why the course-level adjustment alone isn't enough.
    # Off by default (empty list) until config.json sets
    # "player_course_years"; fails soft to a no-op otherwise.
    player_course_years = cfg.get("player_course_years", [])
    if player_course_years:
        profile = add_player_course_history_adjustment(
            profile, cfg["tournament_name"], player_course_years,
            k=cfg.get("player_course_k"), debug=True)

    # Fetched once, used twice: build_props() needs it to evaluate each
    # player's probability against their REAL posted line (not a generic
    # fixed default — see build_props' docstring), and attach_platform_lines
    # still separately records every line found for display/reference.
    platform_dfs = get_platform_lines()
    pp = platform_dfs.get("prizepicks")
    if pp is not None and not pp.empty:
        if "odds_type" in pp.columns:
            counts = pp["odds_type"].value_counts().to_dict()
            print(f"PrizePicks lines by type: {counts} "
                  f"(goblin/demon are one-way and are excluded from the model + hit rates).")
            if not pp.attrs.get("odds_type_field_seen", True):
                print("WARNING: PrizePicks rows had no 'odds_type' field -- goblins/demons "
                      "can't be told apart from standard lines. Check the API field name.")
    course_par = get_course_par(cfg)
    if course_par is None:
        print("No course par known yet (set \"course_par\" in config.json, or it is read "
              "from ESPN once a round is played) -- skipping the experimental Pars / "
              "Bogeys-or-Worse props this run.")
    props = build_props(cfg, profile, build_line_lookup(platform_dfs), course_par=course_par)
    props = attach_platform_lines(props, platform_dfs)

    # Sort so the most lopsided (highest-confidence) props float to the top.
    def max_confidence(p):
        return max(abs(p["gir"]["over"] - 0.5), abs(p["fairways"]["over"] - 0.5),
                   abs(p["birdies"]["over"] - 0.5), abs(p["strokes"]["over"] - 0.5))

    props.sort(key=max_confidence, reverse=True)

    # Guard: if the platforms posted NO real lines for this tournament right
    # now (e.g. PrizePicks pulls the board between rounds, or hasn't posted
    # the next round yet), don't overwrite the site's last good props with
    # a version built on generic default lines, and don't archive a snapshot
    # the grader would just skip. Pass --allow-no-lines to override.
    if not allow_no_lines and count_real_lines(props) == 0:
        previous = load_previous_output()
        if previous and previous.get("tournament_name") == cfg["tournament_name"]:
            print("No real platform lines found for any player right now "
                  "(PrizePicks board empty or not yet posted). Keeping the "
                  f"existing {OUTPUT_PATH} and skipping today's archive — "
                  "re-run once lines are posted (or use --allow-no-lines).")
            return
        print("No real platform lines found, but the existing site data is for "
              "a different tournament (or missing) — writing the lineless props "
              "so the site doesn't show the wrong event.")

    output = {
        "tournament_name": cfg["tournament_name"],
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "course_par": course_par,
        "props": props,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(props)} players' props to {OUTPUT_PATH}")

    # Only archive snapshots with real lines — the grader skips picks with no
    # posted line anyway, so a lineless snapshot has nothing to grade.
    if count_real_lines(props) > 0:
        archive_props_snapshot(output, cfg)
    else:
        print("No real lines in this run — not archiving a snapshot.")


if __name__ == "__main__":
    sys.exit(main(allow_no_lines="--allow-no-lines" in sys.argv))
