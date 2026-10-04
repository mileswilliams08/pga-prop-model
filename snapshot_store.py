"""
Persists each player's cumulative GIR/fairways numbers across daily
runs so grade_results.py can recover a SPECIFIC round's real GIR count
(exact) and fairways-hit count (a rounded estimate — see
espn_round_results.py's docstring for why) by diffing two consecutive
cumulative snapshots.

Why this needs to be a file on disk rather than computed fresh each
time: ESPN's endpoint only ever reports the cumulative total "as of
right now" — there's no way to ask it for "just round 2's GIR" directly.
The only way to recover a single round's number is to have ALREADY
recorded the cumulative total as of the round before it, then subtract.
That means this script has to run at least once after each round
completes and remember what it saw, or a round's GIR/fairways can never
be recovered later — if you skip two days in a row during a tournament,
whatever rounds completed in that gap get flagged "ungradeable" for
GIR/fairways (strokes and birdies-or-better are NOT affected — those
are exact every time regardless of gaps, since ESPN gives them directly
per round, not just cumulatively).

Storage shape (one file, data/espn_cumulative_snapshots.json):
    {
      "<tournament_key>": {
        "<player_key>": {
          "player_name": str,
          "rounds": {
            "1": {"strokes": .., "birdies_or_better": .., "fairways_possible": ..,
                  "holes_played": .., "gir": .. or null, "fairways_hit_est": .. or null,
                  "fairways_estimated": true},
            ...
          },
          "cumulative_checkpoints": {
            "1": {"gir_count": .., "fairways_hit_est": ..},   # as of AFTER round 1
            "2": {...},
            ...
          }
        }
      }
    }

cumulative_checkpoints exists purely as the running total used for
diffing; "rounds" is the actual per-round result grade_results.py reads.
Keeping both means a gap can self-heal later: if round 2 was skipped but
round 3's checkpoint is stored, and you backfill round 2's checkpoint
from an old ESPN query, update_and_diff can recompute round 3 cleanly on
the next run — the raw checkpoints are never discarded.
"""

import json
import os

STORE_PATH = "data/espn_cumulative_snapshots.json"


def load_store(path: str = STORE_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_store(store: dict, path: str = STORE_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(store, f, indent=2)


def _round_or_none(value):
    return None if value is None else round(value)


def update_and_diff(store: dict, tournament_key: str, player_key: str,
                     snapshot: dict) -> dict:
    """
    snapshot: one get_player_cumulative_snapshot() result (from
    espn_round_results.py) for this player, as of right now.

    Merges it into `store` IN PLACE (caller saves afterward — see
    update_results.py for the batch flow) and returns this player's full
    known "rounds" dict after the merge: {round_num_str: {...}}, with
    "gir"/"fairways_hit_est" present (real numbers) for any round that
    could be diffed, or null for a round whose previous checkpoint is
    still missing (not gradable for GIR/fairways yet — strokes and
    birdies_or_better are still returned, since those never depend on
    diffing).
    """
    tournament_bucket = store.setdefault(tournament_key, {})
    player_bucket = tournament_bucket.setdefault(player_key, {
        "player_name": snapshot.get("player_name", ""),
        "rounds": {},
        "cumulative_checkpoints": {},
    })
    player_bucket["player_name"] = snapshot.get("player_name") or player_bucket["player_name"]

    rounds = player_bucket["rounds"]
    checkpoints = player_bucket["cumulative_checkpoints"]

    gir_pct = snapshot.get("cumulative_gir_pct")
    gir_poss = snapshot.get("cumulative_gir_poss")
    drive_pct = snapshot.get("cumulative_drive_accuracy_pct")
    fairways_poss_cum = snapshot.get("cumulative_fairways_possible")

    cumulative_gir_count = (
        _round_or_none(gir_pct / 100 * gir_poss) if gir_pct is not None and gir_poss is not None
        else None
    )
    cumulative_fairways_hit = (
        _round_or_none(drive_pct / 100 * fairways_poss_cum)
        if drive_pct is not None and fairways_poss_cum is not None
        else None
    )

    rounds_completed = snapshot.get("rounds_completed", 0)
    if rounds_completed > 0:
        checkpoints[str(rounds_completed)] = {
            "gir_count": cumulative_gir_count,
            "fairways_hit_est": cumulative_fairways_hit,
        }

    # Store/refresh every round's EXACT numbers (strokes, birdies-or-
    # better, fairways_possible) regardless of whether GIR/fairways can
    # be diffed yet — these never depend on diffing and shouldn't be
    # held hostage by a missing checkpoint.
    for round_num, exact in (snapshot.get("per_round_exact") or {}).items():
        round_key = str(round_num)
        entry = rounds.setdefault(round_key, {})
        entry.update({
            "strokes": exact["strokes"],
            "birdies_or_better": exact["birdies_or_better"],
            "fairways_possible": exact["fairways_possible"],
            "holes_played": exact["holes_played"],
        })

    # Now fill in gir/fairways_hit_est for every round where BOTH this
    # round's checkpoint and the previous round's checkpoint exist.
    for round_key in list(rounds.keys()):
        n = int(round_key)
        this_cp = checkpoints.get(str(n))
        prev_cp = checkpoints.get(str(n - 1)) if n > 1 else {"gir_count": 0, "fairways_hit_est": 0}
        entry = rounds[round_key]
        if this_cp is None or prev_cp is None:
            entry.setdefault("gir", None)
            entry.setdefault("fairways_hit_est", None)
            continue
        this_gir, prev_gir = this_cp.get("gir_count"), prev_cp.get("gir_count")
        this_fw, prev_fw = this_cp.get("fairways_hit_est"), prev_cp.get("fairways_hit_est")
        entry["gir"] = (this_gir - prev_gir) if this_gir is not None and prev_gir is not None else None
        entry["fairways_hit_est"] = (
            (this_fw - prev_fw) if this_fw is not None and prev_fw is not None else None
        )
        entry["fairways_estimated"] = True  # always true when present — see module docstring

    return rounds
