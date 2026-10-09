"""
Recalibrates this week's course profile using REAL, live field
performance from rounds already completed THIS tournament, instead of
relying purely on the multi-year historical average
(course_history.py/get_course_profile()), which is a good default but a
fixed number that can't see this week's actual conditions.

Why this exists: on 2026-10-04 (Bank of Utah Championship), the field
was averaging 68.037 strokes against a par of 71 -- 2.96 strokes under
par -- through round 3, while the model's course baseline (395 real
rounds from 2023-2025) sat much closer to even par. Every strokes/GIR/
fairways/birdies prop computed off the stale historical baseline was
systematically biased toward the field scoring worse than it actually
was, which is exactly what produced a one-sided run of missed "Over"
strokes picks that week. The fix: once a round or two of THIS week's
real results are in, use them.

Data source: data/espn_cumulative_snapshots.json (see snapshot_store.py),
written daily by update_results.py from real ESPN per-round results.
Strokes and birdies-or-better are exact every time; GIR and fairways-hit
are the diffed estimates snapshot_store.py already computes (see
espn_round_results.py's docstring for why fairways-hit is an estimate).

Blending: shrink_toward_mean style, like the rest of this project.
A single completed round of this week's data gets modest trust (the
course might play differently in different wind on a later day); several
completed rounds get dominant trust. The unit of trust is DISTINCT
TOURNAMENT ROUNDS completed so far this week (1, 2, 3...), not raw
player-round counts -- the field already has 100+ players per round, so
the real uncertainty is "will conditions change on a later day," not
"do we have enough players." See DEFAULT_LIVE_CONDITIONS_K.

Fails soft throughout: no snapshot store yet, nothing recorded for this
tournament, or zero fully-completed rounds all leave the historical
course profile untouched (printing why, if debug=True) -- this is a
refinement layered on top of get_course_profile(), never a hard
requirement for update_data.py to run.

NOTE on timing: the snapshot store is written by update_results.py,
which runs AFTER update_data.py in the daily GitHub Action (see
update.yml) -- so a given day's props reflect the store as of the END of
the PREVIOUS day's run, not anything from later today. In practice this
lines up fine (a round's results aren't final until evening anyway, well
after that day's props were generated), but it does mean a round's
live-conditions signal shows up in the NEXT day's props, not the same
day's.
"""

from course_history import normalize_tournament_name

# Rounds of historical weight a single live round must overcome. k=1.5
# means: after 1 completed round this week, weight = 1/(1+1.5) = 0.40 on
# live data; after 2 rounds, 0.57; after 3 rounds, 0.67. Deliberately NOT
# aggressive after just one round -- a single day's conditions (wind,
# one bad weather window) shouldn't fully override a 3-year baseline,
# but should meaningfully nudge it. Not yet validated against a full
# season of backtests (there's no way to backtest "live" data cleanly
# without rebuilding snapshot history day-by-day for past tournaments) --
# treat this as a reasonable starting point, not a tuned constant.
DEFAULT_LIVE_CONDITIONS_K = 1.5

# A round only counts toward the live adjustment once at least this share
# of the players recorded for the tournament have finished it. Without
# this, the first few players to finish a round (the earliest tee times,
# all in the same morning conditions) would count as a whole "round" at
# full weight. 0.35 (not higher) so rounds 3-4 still qualify after a cut,
# when only ~40-45% of the original field is still playing.
MIN_FIELD_SHARE_FOR_ROUND = 0.35


def load_live_field_profile(tournament_name: str, store: dict) -> dict:
    """
    store: the loaded data/espn_cumulative_snapshots.json dict (see
    snapshot_store.py), already keyed by tournament_key -> player_key ->
    {"rounds": {...}, "cumulative_checkpoints": {...}}.

    Averages every FULLY COMPLETED (holes_played == 18) player-round
    found for this tournament across the whole field, for each of the
    four stats, counting only rounds that at least
    MIN_FIELD_SHARE_FOR_ROUND of the recorded players have finished.
    Returns None if nothing is recorded yet for this tournament or no
    round has been finished by enough of the field.

    Returns:
        {"scoring_avg": float, "birdie_rate_per_hole": float or None,
         "gir_pct": float or None, "driving_accuracy": float or None,
         "n_rounds": int}  # distinct completed tournament rounds (1,2,3..)

    gir_pct/driving_accuracy can be None (if no player's GIR/fairways
    could be diffed yet for any completed round, e.g. day-one gap) even
    when scoring_avg/birdie_rate_per_hole are populated, since those two
    are exact and never depend on diffing.
    """
    tournament_key = normalize_tournament_name(tournament_name)
    tdata = store.get(tournament_key)
    if not tdata:
        return None

    strokes, birdie_rates, gir_rates, driving_rates = [], [], [], []
    distinct_rounds = set()

    # How many players have finished each round, versus everyone recorded
    # for this tournament. Rounds that too few have finished are skipped
    # entirely for now (see MIN_FIELD_SHARE_FOR_ROUND).
    n_players = len(tdata)
    finished_by_round = {}
    for pdata in tdata.values():
        for round_key, r in (pdata.get("rounds") or {}).items():
            if r.get("holes_played") == 18:
                finished_by_round[round_key] = finished_by_round.get(round_key, 0) + 1
    usable_rounds = {rk for rk, cnt in finished_by_round.items()
                     if n_players and cnt >= MIN_FIELD_SHARE_FOR_ROUND * n_players}

    for player_key, pdata in tdata.items():
        for round_key, r in (pdata.get("rounds") or {}).items():
            if r.get("holes_played") != 18:
                continue  # only count rounds the player actually finished --
                          # a partial round's totals aren't comparable.
            if round_key not in usable_rounds:
                continue  # too few of the field have finished this round yet
            distinct_rounds.add(round_key)
            strokes.append(r["strokes"])
            birdie_rates.append(r["birdies_or_better"] / 18)

            if r.get("gir") is not None:
                gir_rates.append(r["gir"] / 18)

            fairways_possible = r.get("fairways_possible")
            if r.get("fairways_hit_est") is not None and fairways_possible:
                driving_rates.append(r["fairways_hit_est"] / fairways_possible)

    if not strokes:
        return None

    def _avg(xs):
        return sum(xs) / len(xs) if xs else None

    return {
        "scoring_avg": _avg(strokes),
        "birdie_rate_per_hole": _avg(birdie_rates),
        "gir_pct": _avg(gir_rates),
        "driving_accuracy": _avg(driving_rates),
        "n_rounds": len(distinct_rounds),
        "n_gir_samples": len(gir_rates),
        "n_driving_samples": len(driving_rates),
    }


def blend_live_course_conditions(historical_course: dict, live_profile: dict,
                                  k: float = DEFAULT_LIVE_CONDITIONS_K,
                                  debug: bool = False) -> dict:
    """
    Blends live_profile (this week's real, observed field averages) into
    historical_course (the output of get_course_profile() -- either the
    real multi-year history or config.json's hand-typed fallback),
    weighted by live_profile["n_rounds"] via shrink_toward_mean style
    shrinkage (see module docstring for why rounds, not player-count, is
    the trust unit).

    Returns a NEW dict (never mutates historical_course). Any stat where
    live_profile's value is None (e.g. gir_pct with no diffable rounds
    yet) is left at its historical value.
    """
    if not live_profile or not live_profile.get("n_rounds"):
        return dict(historical_course)

    n_rounds = live_profile["n_rounds"]
    weight = n_rounds / (n_rounds + k)
    blended = dict(historical_course)

    for stat in ("scoring_avg", "gir_pct", "driving_accuracy", "birdie_rate_per_hole"):
        live_val = live_profile.get(stat)
        hist_val = historical_course.get(stat)
        if live_val is None or hist_val is None:
            continue
        blended[stat] = weight * live_val + (1 - weight) * hist_val
        if debug:
            print(f"  live conditions -- {stat}: historical={hist_val:.4f}  "
                  f"this week so far ({n_rounds} round(s))={live_val:.4f}  "
                  f"delta={live_val - hist_val:+.4f}  weight={weight:.2f}  "
                  f"-> blended={blended[stat]:.4f}")

    return blended


def apply_live_conditions(course: dict, cfg: dict, debug: bool = False) -> dict:
    """
    Entry point for update_data.py. Loads data/espn_cumulative_snapshots.json
    and blends this week's real, in-progress field performance into
    `course` (get_course_profile()'s output), returning a NEW dict.

    Fails soft: any error loading the store, nothing recorded yet for
    this tournament, or zero fully-completed rounds all return `course`
    unchanged (printing why, if debug=True) -- never blocks a normal run.

    config.json can optionally set "live_conditions_k" to override
    DEFAULT_LIVE_CONDITIONS_K.
    """
    try:
        from snapshot_store import load_store
        store = load_store()
    except Exception as e:
        if debug:
            print(f"Live conditions: couldn't load snapshot store ({e}) -- "
                  f"using the historical course profile as-is.")
        return course

    live_profile = load_live_field_profile(cfg["tournament_name"], store=store)
    if live_profile is None:
        if debug:
            print(f"Live conditions: no completed-round data yet for "
                  f"'{cfg['tournament_name']}' -- using the historical "
                  f"course profile as-is (normal before round 1 finishes, "
                  f"or if update_results.py hasn't run yet).")
        return course

    k = cfg.get("live_conditions_k", DEFAULT_LIVE_CONDITIONS_K)
    if debug:
        print(f"Live conditions: found {live_profile['n_rounds']} fully "
              f"completed round(s) of real data for this week "
              f"(k={k}) -- blending into the course profile:")
    return blend_live_course_conditions(course, live_profile, k=k, debug=debug)
