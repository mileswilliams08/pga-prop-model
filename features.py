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
        df["birdie_rate_per_hole"] = df["birdie_avg"] / 18

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
