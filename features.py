"""
Turns raw scraped stats into the numbers the prop models actually need:
clean per-player rates, plus course-adjusted versions of those rates.

Course adjustment logic:
    A course being "easy" or "hard" shifts everyone's numbers the same
    direction. Rather than using a player's season average blindly, we
    shift it toward what's realistic *at this course* using a simple
    logit-space adjustment for rate stats (GIR%, fairway%, birdie%) and
    an additive adjustment for scoring average.

    adjusted_rate = inverse_logit( logit(player_rate)
                                    + logit(course_rate)
                                    - logit(tour_avg_rate) )

    This keeps everything bounded in [0, 1] and treats the course's
    effect as a multiplicative shift on the odds scale, which behaves
    much better than raw addition for numbers near 0 or 1.
"""

import numpy as np
import pandas as pd

# Calibrated against real per-round results (backtest_rounds.py,
# --birdie-bias-sweep, 2025 season, 12.8k player-rounds): ESPN's
# BIRDS/birdie_avg column undercounts what real "birdies-or-better"
# outcomes actually clear a 3.5 line by (season base rate 0.586 vs.
# model's uncorrected 0.553) — plausibly because it doesn't count
# eagles the same way a real per-hole score<par check does. This flat
# shift closed that gap and was the only tested value that beat the
# climatology baseline (0.2417 vs 0.2426 Brier score). Re-tune with
# `python backtest_rounds.py --birdie-bias-sweep` if ESPN's page or the
# scraper's column mapping changes.
BIRDIE_RATE_BIAS = 0.01


def _to_frac(x):
    """Convert a percentage string/number like '65.4' or '65.4%' to 0.654."""
    if isinstance(x, str):
        x = x.replace("%", "").strip()
    val = float(x)
    return val / 100 if val > 1.5 else val  # handles both "65.4" and "0.654"


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def inv_logit(x):
    return 1 / (1 + np.exp(-x))


def clean_player_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns: player, gir_pct, driving_accuracy, scoring_avg,
    birdie_avg, sg_total, sg_off_tee, sg_approach, sg_around_green, sg_putting
    (this matches scraper.py's output). Missing columns are left as NaN.
    """
    df = raw.copy()
    for col in ["gir_pct", "driving_accuracy"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: _to_frac(v) if pd.notna(v) else v)
    for col in ["scoring_avg", "birdie_avg", "sg_total", "sg_off_tee",
                "sg_approach", "sg_around_green", "sg_putting"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # birdie_avg on PGA Tour's site is birdies-per-round; convert to a
    # per-hole rate assuming 18 holes.
    if "birdie_avg" in df.columns:
        df["birdie_rate_per_hole"] = df["birdie_avg"] / 18 + BIRDIE_RATE_BIAS

    # Drop players with no meaningful season data. Retired or
    # limited-status players who played ~0 rounds show up with a
    # literal 0.0 for scoring average and GIR% (ESPN renders missing
    # data as zero rather than blank on this page) — a real active
    # golfer can't actually have 0% GIR or a 0.0 scoring average, so
    # these are data gaps, not real stats, and would otherwise wreck
    # the model with impossible probabilities.
    if "scoring_avg" in df.columns:
        df = df[df["scoring_avg"] >= 50]
    if "gir_pct" in df.columns:
        df = df[df["gir_pct"] > 0]

    return df


def course_adjust_rate(player_rate: pd.Series, course_rate: float,
                        tour_avg_rate: float) -> pd.Series:
    """Adjust a player's rate stat (0-1) for a specific course's difficulty."""
    adj_logit = logit(player_rate) + logit(course_rate) - logit(tour_avg_rate)
    return inv_logit(adj_logit)


def course_adjust_scoring(player_scoring_avg: pd.Series,
                           course_scoring_avg: float,
                           tour_avg_scoring_avg: float) -> pd.Series:
    """Additive adjustment for scoring average (strokes are already linear)."""
    return player_scoring_avg + (course_scoring_avg - tour_avg_scoring_avg)


def build_player_course_profile(stats: pd.DataFrame, course: dict,
                                 tour_avg: dict) -> pd.DataFrame:
    """
    course and tour_avg are dicts like:
        {"gir_pct": 0.65, "driving_accuracy": 0.60, "scoring_avg": 71.2,
         "birdie_rate_per_hole": 0.18}

    Returns a DataFrame with course-adjusted columns added
    (prefixed 'adj_').
    """
    out = stats.copy()
    if "gir_pct" in out.columns:
        out["adj_gir_pct"] = course_adjust_rate(
            out["gir_pct"], course["gir_pct"], tour_avg["gir_pct"])
    if "driving_accuracy" in out.columns:
        out["adj_driving_accuracy"] = course_adjust_rate(
            out["driving_accuracy"], course["driving_accuracy"],
            tour_avg["driving_accuracy"])
    if "birdie_rate_per_hole" in out.columns:
        out["adj_birdie_rate_per_hole"] = course_adjust_rate(
            out["birdie_rate_per_hole"], course["birdie_rate_per_hole"],
            tour_avg["birdie_rate_per_hole"])
    if "scoring_avg" in out.columns:
        out["adj_scoring_avg"] = course_adjust_scoring(
            out["scoring_avg"], course["scoring_avg"], tour_avg["scoring_avg"])
    return out


# (adj_col read/written, raw per-player-course column it blends toward it,
# output column name) -- shared with backtest_rounds.py's identical table
# so the two stay in lockstep if a stat is ever added.
PLAYER_COURSE_STAT_COLS = [
    ("adj_gir_pct", "gir_pct", "pc_gir_pct"),
    ("adj_driving_accuracy", "driving_accuracy", "pc_driving_accuracy"),
    ("adj_birdie_rate_per_hole", "birdie_rate_per_hole", "pc_birdie_rate_per_hole"),
    ("adj_scoring_avg", "scoring_avg", "pc_scoring_avg"),
]


def add_player_course_history_adjustment(profile: pd.DataFrame, tournament_name: str,
                                          player_course_years: list, k: float = None,
                                          debug: bool = False) -> pd.DataFrame:
    """
    Layers THIS SPECIFIC WEEK'S players' own real per-round history at
    THIS SPECIFIC tournament (player_course_history.py) on top of the
    course-level adjustment already baked into `profile`'s adj_* columns
    (must run build_player_course_profile first).

    Why this matters, concretely: a player's season-wide blended average
    plus a generic field-wide course-difficulty shift has no way to see
    that one particular player is a strong (or weak) fit for one
    particular course -- e.g. a player who has shot several rounds well
    under their own normal scoring average every time they've played
    this specific event. That real track record is exactly the signal
    PrizePicks' own "last 5 rounds at this course" line is likely
    pricing in, so without it the model ends up comparing the platform's
    line against a number that's still close to the player's generic
    season average -- manufacturing false confidence in whichever
    direction the platform's line already moved away from that generic
    number. shrink_player_course_rate (used internally here) keeps this
    safe: a player with only 1-2 rounds at the event barely moves at all,
    and a player who has never played it (or whose name doesn't match)
    is left completely unchanged.

    Fails soft: no years configured, no historical_rounds CSVs on disk
    yet, or no real history found for this specific tournament all just
    leave `profile` exactly as it was (printing why, if debug=True),
    rather than erroring or blocking a normal update_data.py run.
    """
    from player_course_history import build_player_course_profiles, DEFAULT_PLAYER_COURSE_K
    from course_history import normalize_tournament_name
    from match_props import normalize_name

    if k is None:
        k = DEFAULT_PLAYER_COURSE_K

    out = profile.copy()
    if not player_course_years:
        return out

    pc_profiles = build_player_course_profiles(player_course_years)
    if pc_profiles.empty:
        if debug:
            print(f"No player-course history available (tried years "
                  f"{player_course_years}) -- leaving course-level "
                  f"adjustment as-is for everyone.")
        return out

    tournament_key = normalize_tournament_name(tournament_name)
    if tournament_key not in pc_profiles.index.get_level_values("tournament_key"):
        if debug:
            print(f"No player-course history found for '{tournament_name}' "
                  f"(tried years {player_course_years}) -- leaving "
                  f"course-level adjustment as-is for everyone.")
        return out

    this_event = pc_profiles.xs(tournament_key, level="tournament_key").rename(columns={
        "gir_pct": "raw_pc_gir_pct",
        "driving_accuracy": "raw_pc_driving_accuracy",
        "scoring_avg": "raw_pc_scoring_avg",
        "birdie_rate_per_hole": "raw_pc_birdie_rate_per_hole",
        "n_rounds": "pc_n_rounds",
    }).reset_index().rename(columns={"player_key": "name_key"})

    out["name_key"] = out["player"].apply(normalize_name)
    out = out.merge(this_event, on="name_key", how="left")
    out["pc_n_rounds"] = out["pc_n_rounds"].fillna(0)

    if debug:
        matched = int((out["pc_n_rounds"] > 0).sum())
        print(f"Player-course history: {matched}/{len(out)} players in the field "
              f"have real rounds at '{tournament_name}' ({player_course_years}); "
              f"the rest keep their course-level-adjusted numbers unchanged.")

    shrink_weight = out["pc_n_rounds"] / (out["pc_n_rounds"] + k)
    for adj_col, _, out_col in PLAYER_COURSE_STAT_COLS:
        raw_col = f"raw_{out_col}"
        if adj_col not in out.columns or raw_col not in out.columns:
            continue
        course_specific_rate = out[raw_col].fillna(0.0)
        out[adj_col] = shrink_weight * course_specific_rate + (1 - shrink_weight) * out[adj_col]

    drop_cols = ["name_key", "pc_n_rounds"] + [f"raw_{c}" for _, _, c in PLAYER_COURSE_STAT_COLS]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])
    return out
