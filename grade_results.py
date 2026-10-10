"""
Compares each day's ARCHIVED props snapshot (data/props_history/, from
update_data.py's archive_props_snapshot) against REAL round results
(espn_round_results.py + snapshot_store.py) and logs every gradable pick
as a hit, miss, or push — "every graded prop", per how this was scoped
(not just picks that cleared a breakeven threshold).

Which round did a given day's snapshot predict? update_data.py runs
once a day, before that day's round is played, so a snapshot generated
on calendar day D is predicting whichever round is actually played on
day D. We don't get an explicit round number from ESPN for "what's
about to be played" without another API call, so this infers it from
the tournament's start date instead: round_number = days since the
tournament started + 1 (capped at the event's total round count). This
is a simplifying assumption — a rain delay or restructured schedule
could throw it off by a day — so grade_tournament_day() only grades a
round once snapshot_store confirms that SPECIFIC round's real result is
actually available; a mismatched guess just means "nothing to grade
yet" rather than silently grading against the wrong round.

Push handling: a line can be a whole number (not just X.5), so an exact
tie between the model's pick and the real result is possible and isn't
a win or a loss — it's excluded from the hit-rate denominator, same as
a sportsbook push, but still logged so the record is complete.

Output: appends one row per gradable (player, category, round) to
data/graded_results.csv (append-only, deduplicated against what's
already logged so reruns are safe), then build_results_summary()
aggregates that log into docs/data/results.json for the website's
results tab — see that function's docstring for the exact shape.
"""

import csv
import glob
import json
import os
from datetime import date, datetime

GRADED_LOG_PATH = "data/graded_results.csv"
RESULTS_OUTPUT_PATH = "docs/data/results.json"
EXCLUDED_PICKS_PATH = "data/excluded_picks.csv"  # hand-maintained, see exclude_pick.py
PROPS_HISTORY_DIR = "data/props_history"

GRADED_LOG_FIELDS = [
    "date", "tournament", "round", "player", "category", "line", "source",
    "model_prob", "picked_side", "actual_value", "result", "fairways_estimated",
]

# category -> which field in a snapshot_store round-entry holds the real value.
CATEGORY_TO_RESULT_FIELD = {
    "gir": "gir",
    "fairways": "fairways_hit_est",
    "birdies": "birdies_or_better",
    "strokes": "strokes",
    "pars": "pars",
    "bogeys": "bogeys_or_worse",
}

# Still being validated: tracked and shown on the results page, but kept out
# of the headline "overall" numbers and the calibration table so they can't
# move the record of the core props.
EXPERIMENTAL_CATEGORIES = ("pars", "bogeys")


def infer_round_number(generated_at_iso: str, tournament_start_date_iso: str,
                        total_rounds: int = 4) -> int:
    """
    generated_at_iso: the archived snapshot's "generated_at" (UTC
    timestamp string). tournament_start_date_iso: the event's listed
    start date (e.g. ESPN's event "date" field, "2026-10-01T04:00Z").
    See module docstring for the day-offset assumption and its limits.
    """
    generated_date = datetime.fromisoformat(generated_at_iso.replace("Z", "+00:00")).date()
    start_date = datetime.fromisoformat(tournament_start_date_iso.replace("Z", "+00:00")).date()
    offset_days = (generated_date - start_date).days
    return max(1, min(total_rounds, offset_days + 1))


def calendar_round_guess(generated_at_iso: str, tournament_start_date_iso: str,
                          total_rounds: int = 4):
    """
    Fallback for snapshots with no stamped predicted_round: the same
    days-since-start guess as infer_round_number, but returns None when
    the date falls before the tournament starts or after its last round,
    instead of clamping to round 1 / the final round. A snapshot from
    outside the event's dates has nothing real to be graded against.
    """
    generated_date = datetime.fromisoformat(generated_at_iso.replace("Z", "+00:00")).date()
    start_date = datetime.fromisoformat(tournament_start_date_iso.replace("Z", "+00:00")).date()
    round_number = (generated_date - start_date).days + 1
    if round_number < 1 or round_number > total_rounds:
        return None
    return round_number


def predicted_round_from_field(rounds_completed_by_player: list, total_rounds: int = 4,
                                threshold: float = 0.4):
    """
    Which round was a snapshot archived right now predicting? Answers
    from what the field has ACTUALLY finished, not from the calendar —
    infer_round_number's "days since start + 1" assumes each day's round
    is played after that day's snapshot is generated, which only holds
    for US-timezone events. For an event in Japan (e.g. the Baycurrent
    Classic), each round is already over before the 13:00 UTC cron runs,
    so the date-based guess lands one round too early.

    A round counts as finished once at least `threshold` of the field
    has completed it (0.4, not ~1.0, so players who withdrew or missed
    the cut don't hold a finished round open, while a handful of early
    finishers in a round still in progress don't count it as done).
    Predicted round = finished rounds + 1. Returns None when every round
    is finished (tournament over — nothing left to predict) or when
    there's no data at all, so callers skip grading rather than guess.
    """
    n = len(rounds_completed_by_player)
    if n == 0:
        return None
    finished = 0
    for r in range(1, total_rounds + 1):
        count = sum(1 for c in rounds_completed_by_player if (c or 0) >= r)
        if count >= threshold * n:
            finished = r
        else:
            break
    if finished >= total_rounds:
        return None
    return finished + 1


def grade_one_pick(line: float, over_prob: float, under_prob: float,
                    actual_value: float) -> dict:
    """
    Picks the model's own side (whichever it gave >=50% to) and checks
    it against the real result. Returns {"picked_side", "model_prob",
    "result"} where result is "hit" / "miss" / "push" (actual_value
    exactly equals the line — can only happen on a whole-number line,
    e.g. a GIR line of 13 rather than 12.5).
    """
    if over_prob >= under_prob:
        picked_side, model_prob = "over", over_prob
    else:
        picked_side, model_prob = "under", under_prob

    if actual_value == line:
        result = "push"
    elif (actual_value > line) == (picked_side == "over"):
        result = "hit"
    else:
        result = "miss"

    return {"picked_side": picked_side, "model_prob": model_prob, "result": result}


ONE_WAY_ODDS_TYPES = ("goblin", "demon")


def is_untakeable_line(player_entry: dict, category: str, use_heuristic: bool = False) -> bool:
    """
    True if this prop shouldn't be graded because its line can't be taken
    both ways (a PrizePicks goblin/demon, Over-only) or can't be shown to be
    a standard two-sided line.

    New snapshots label each real line with "odds_type" ("standard" only
    ever reaches here). OLDER snapshots have no label; by default those are
    left alone (grading them as before — remove known one-way picks by hand
    with exclude_pick.py). With use_heuristic=True (only prune_one_way.py
    does this) the best available check is the player's recorded
    platform_lines: with one distinct
    PrizePicks line for the stat it's safely the standard; with three the
    evaluated line must be the middle one (goblin is the low line, demon the
    high); anything else is ambiguous and is skipped rather than risk
    counting a one-way prop.
    """
    stat = player_entry.get(category) or {}
    odds_type = stat.get("odds_type")
    if odds_type is not None:
        return str(odds_type).strip().lower() in ONE_WAY_ODDS_TYPES

    if not use_heuristic or stat.get("source") != "prizepicks":
        return False
    lines = sorted({pl.get("line") for pl in (player_entry.get("platform_lines") or [])
                    if pl.get("source") == "prizepicks" and pl.get("category") == category
                    and pl.get("line") is not None})
    if len(lines) <= 1:
        return False
    if len(lines) == 3 and stat.get("line") == lines[1]:
        return False
    return True


def grade_tournament_day(props_snapshot: dict, graded_rounds_by_player: dict,
                          round_number: int) -> list:
    """
    props_snapshot: one archived data/props_history/*.json file's
        parsed content (tournament_name, generated_at, props[]).
    graded_rounds_by_player: normalized_player_key -> snapshot_store's
        per-player "rounds" dict (round_num_str -> {strokes,
        birdies_or_better, gir, fairways_hit_est, ...}).
    round_number: which round this day's snapshot was predicting (see
        infer_round_number).

    Returns a list of graded-row dicts (GRADED_LOG_FIELDS shape, minus
    "date"/"tournament" which the caller fills in) — one per player per
    prop category that BOTH had a real platform line (source is not
    None, i.e. this was an actual postable prop, not a generic-default
    fallback) AND has a real, gradable result for this exact round
    (skips silently otherwise — a still-ungradeable GIR/fairways round
    from a diffing gap, or a player with no ESPN match, just isn't
    included yet, not an error).
    """
    from match_props import normalize_name

    rows = []
    for player_entry in props_snapshot.get("props", []):
        player_key = normalize_name(player_entry["player"])
        real_rounds = graded_rounds_by_player.get(player_key)
        if not real_rounds:
            continue
        real_round = real_rounds.get(str(round_number))
        if not real_round:
            continue
        if real_round.get("holes_played") != 18:
            continue  # round not actually finished yet -- a partial or
                       # zeroed placeholder round (see espn_round_results.py's
                       # matching guard) isn't a real result to grade against.
                       # Belt-and-suspenders: that guard should already stop
                       # a placeholder from reaching here, but grading is
                       # the point where a bad round does the most damage,
                       # so it gets its own check too.

        for category, result_field in CATEGORY_TO_RESULT_FIELD.items():
            stat = player_entry.get(category)
            if not stat or stat.get("source") is None:
                continue  # no real platform line was posted for this prop — nothing to grade
            if is_untakeable_line(player_entry, category):
                continue  # goblin/demon (one-way) or ambiguous multi-line prop — not gradable
            actual_value = real_round.get(result_field)
            if actual_value is None:
                continue  # not gradable yet (e.g. GIR/fairways diffing gap)

            graded = grade_one_pick(stat["line"], stat["over"], stat["under"], actual_value)
            rows.append({
                "round": round_number,
                "player": player_entry["player"],
                "category": category,
                "line": stat["line"],
                "source": stat["source"],
                "actual_value": actual_value,
                "fairways_estimated": category == "fairways",
                **graded,
            })
    return rows


def _load_existing_keys(path: str) -> set:
    """Dedup key: (date, player, category, round) — reruns on the same
    day/round/player/category just skip, instead of double-logging."""
    if not os.path.exists(path):
        return set()
    keys = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            keys.add((row["date"], row["player"], row["category"], row["round"]))
    return keys


def append_graded_rows(rows: list, date_str: str, tournament_name: str,
                        path: str = GRADED_LOG_PATH) -> int:
    """Appends new rows (with "date"/"tournament" filled in) to the
    CSV log, skipping anything already logged. Returns how many NEW
    rows were actually written."""
    existing_keys = _load_existing_keys(path)
    new_rows = []
    for row in rows:
        full_row = {"date": date_str, "tournament": tournament_name, **row}
        key = (date_str, full_row["player"], full_row["category"], str(full_row["round"]))
        if key in existing_keys:
            continue
        new_rows.append(full_row)
        existing_keys.add(key)

    if not new_rows:
        return 0

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GRADED_LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in new_rows:
            writer.writerow({k: row.get(k, "") for k in GRADED_LOG_FIELDS})
    return len(new_rows)


CONFIDENCE_BUCKETS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]


def _bucket_label(prob: float) -> str:
    for lo, hi in CONFIDENCE_BUCKETS:
        if lo <= prob < hi:
            return f"{int(lo*100)}-{int(hi*100) if hi <= 1 else 100}%"
    return "unknown"


def load_exclusions(path: str = EXCLUDED_PICKS_PATH) -> set:
    """
    Picks the person running this has marked as not takeable (e.g. a
    PrizePicks goblin/demon, Over-only) and wants out of every hit rate,
    without deleting them from the graded log. One row per pick in
    data/excluded_picks.csv: tournament, round, player, category.
    Returns a set of (normalized tournament, str(round), normalized player,
    category) keys.
    """
    import pandas as pd
    from course_history import normalize_tournament_name
    from match_props import normalize_name
    if not os.path.exists(path):
        return set()
    ex = pd.read_csv(path, dtype=str).fillna("")
    keys = set()
    for _, r in ex.iterrows():
        keys.add((normalize_tournament_name(r["tournament"]), str(r["round"]).strip(),
                  normalize_name(r["player"]), r["category"].strip().lower()))
    return keys


def apply_exclusions(df, exclusions: set):
    """Drops excluded picks from a graded-log DataFrame (the CSV itself is untouched)."""
    if not exclusions or df.empty:
        return df
    from course_history import normalize_tournament_name
    from match_props import normalize_name
    keys = df.apply(lambda r: (normalize_tournament_name(str(r["tournament"])),
                               str(int(r["round"])) if str(r["round"]).replace(".0", "").isdigit()
                               else str(r["round"]),
                               normalize_name(r["player"]), str(r["category"]).lower()), axis=1)
    return df[~keys.isin(exclusions)].copy()


def build_results_summary(path: str = GRADED_LOG_PATH, recent_n: int = 50) -> dict:
    """
    Aggregates the graded-results log into the shape the website's
    results tab reads (docs/data/results.json):

        {
          "generated_at": iso timestamp,
          "overall": {"n": int, "hits": int, "hit_rate": float},
          "by_category": {"gir": {...}, "fairways": {...}, ...},
          "by_confidence_bucket": {"50-60%": {...}, "60-70%": {...}, ...},
          "recent": [ {date, tournament, player, category, line, result, model_prob}, ... ]
        }

    Pushes are excluded from every hit_rate denominator (counted
    separately as "pushes") — same convention as a sportsbook push,
    since neither side of the bet actually resolved.
    "By confidence bucket" is the calibration check: if the model is
    well-calibrated, its 70-80% bucket should hit close to 70-80% of
    the time, not more and not less.
    """
    import pandas as pd

    if not os.path.exists(path):
        return {
            "generated_at": pd.Timestamp.now("UTC").isoformat(),
            "overall": {"n": 0, "hits": 0, "pushes": 0, "hit_rate": None},
            "by_category": {}, "by_side": {}, "by_category_side": {}, "by_tournament": {},
            "by_round": {}, "picks": [], "by_confidence_bucket": {}, "recent": [],
        }

    df = pd.read_csv(path)
    df["model_prob"] = pd.to_numeric(df["model_prob"], errors="coerce")
    df = apply_exclusions(df, load_exclusions())

    def summarize(sub: pd.DataFrame) -> dict:
        pushes = int((sub["result"] == "push").sum())
        decided = sub[sub["result"] != "push"]
        hits = int((decided["result"] == "hit").sum())
        n = len(decided)
        return {
            "n": n, "hits": hits, "pushes": pushes,
            "hit_rate": (hits / n) if n > 0 else None,
        }

    # Headline numbers + calibration table cover the core props only.
    core = df[~df["category"].isin(EXPERIMENTAL_CATEGORIES)]
    overall = summarize(core)

    by_category = {
        cat: summarize(df[df["category"] == cat])
        for cat in sorted(df["category"].unique())
    }

    df["_bucket"] = df["model_prob"].apply(lambda p: _bucket_label(p) if pd.notna(p) else "unknown")
    core = df[~df["category"].isin(EXPERIMENTAL_CATEGORIES)]
    by_confidence_bucket = {
        bucket: summarize(core[core["_bucket"] == bucket])
        for bucket in sorted(core["_bucket"].unique()) if bucket != "unknown"
    }

    # Hit rate for the model's Over picks vs. its Under picks ("picked_side"
    # is the side the model chose, whichever had the higher probability).
    # Also split per prop type, since a lean can hide inside one category
    # (e.g. Total Strokes Overs) while the overall split looks balanced.
    df["picked_side"] = df["picked_side"].astype(str).str.lower()
    def side_breakdown(sub: pd.DataFrame):
        by_side_ = {side: summarize(sub[sub["picked_side"] == side])
                    for side in ("over", "under")}
        by_cat_side_ = {
            cat: {side: summarize(sub[(sub["category"] == cat) & (sub["picked_side"] == side)])
                  for side in ("over", "under")}
            for cat in sorted(sub["category"].unique())
        }
        return by_side_, by_cat_side_

    by_side, by_category_side = side_breakdown(df)

    df["round"] = pd.to_numeric(df["round"], errors="coerce")

    def scope_stats(sub: pd.DataFrame) -> dict:
        """Everything the results page shows for one slice of the graded log."""
        s_side, s_cat_side = side_breakdown(sub)
        return {
            "overall": summarize(sub),
            "by_category": {cat: summarize(sub[sub["category"] == cat])
                            for cat in sorted(sub["category"].unique())},
            "by_side": s_side,
            "by_category_side": s_cat_side,
        }

    def by_round_stats(sub: pd.DataFrame) -> dict:
        """The same stats split by tournament round (keys "1".."4")."""
        return {str(int(r)): scope_stats(sub[sub["round"] == r])
                for r in sorted(sub["round"].dropna().unique())}

    # Same numbers per tournament, so one week's results (e.g. a one-sided
    # week) don't hide inside the all-time totals above. The all-tournament
    # numbers above are unchanged. Each tournament (and the all-tournament
    # view) also carries a by_round split for the page's round filters.
    by_tournament = {}
    for tname in sorted(df["tournament"].dropna().unique()):
        sub = df[df["tournament"] == tname]
        t_side, t_cat_side = side_breakdown(sub)
        by_tournament[tname] = {
            "overall": summarize(sub),
            "by_side": t_side,
            "by_category_side": t_cat_side,
            "by_round": by_round_stats(sub),
        }
    by_round = by_round_stats(df)

    recent = (
        df.sort_values("date", ascending=False)
        .head(recent_n)[["date", "tournament", "player", "category", "line",
                          "result", "model_prob", "picked_side", "actual_value"]]
        .to_dict(orient="records")
    )

    # Compact per-pick rows (after exclusions) so the results page can filter
    # by anything -- tournament, round, prop, side, probability range -- and
    # compute hit rates in the browser instead of precomputing every
    # combination. t=tournament, r=round, c=category, s=side picked,
    # p=model probability of that side, x=h(it)/m(iss)/p(ush).
    result_code = {"hit": "h", "miss": "m", "push": "p"}
    picks = [
        {"t": r.tournament, "r": int(r.round) if pd.notna(r.round) else None,
         "c": r.category, "s": r.picked_side,
         "p": round(float(r.model_prob), 4) if pd.notna(r.model_prob) else None,
         "x": result_code.get(r.result, "p")}
        for r in df.itertuples()
    ]

    return {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "picks": picks,
        "overall": overall,
        "by_category": by_category,
        "by_side": by_side,
        "by_category_side": by_category_side,
        "by_tournament": by_tournament,
        "by_round": by_round,
        "by_confidence_bucket": by_confidence_bucket,
        "recent": recent,
    }


def write_results_json(summary: dict, path: str = RESULTS_OUTPUT_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def list_archived_snapshots(tournament_slug: str, directory: str = PROPS_HISTORY_DIR) -> list:
    """Every archived props snapshot for this tournament, oldest first."""
    paths = sorted(glob.glob(f"{directory}/{tournament_slug}_*.json"))
    return paths
