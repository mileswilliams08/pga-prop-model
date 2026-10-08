"""
Orchestrates the results-tracking pipeline for THIS week's tournament:
fetches real per-round results from ESPN, updates the cumulative
snapshot store (so GIR/fairways can be diffed — see
espn_round_results.py and snapshot_store.py for why that's necessary),
grades every archived props snapshot (data/props_history/, from
update_data.py) against those real results, and refreshes
docs/data/results.json for the website's results tab.

Run this once a day, same cadence as update_data.py but AFTER it (so a
day's freshly-archived snapshot can potentially be graded the moment a
real result becomes available) — see .github/workflows/update.yml.

Usage:
    python update_results.py
    python update_results.py --debug
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone

from course_history import normalize_tournament_name
from espn_round_results import find_event_id, get_event_metadata, get_tournament_snapshots
from snapshot_store import load_store, save_store, update_and_diff
from grade_results import (list_archived_snapshots, calendar_round_guess,
                            predicted_round_from_field,
                            grade_tournament_day, append_graded_rows,
                            build_results_summary, write_results_json)
from match_props import normalize_name

import json

CONFIG_PATH = "config.json"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def main(debug: bool = False):
    cfg = load_config()
    tournament_name = cfg["tournament_name"]
    tournament_key = normalize_tournament_name(tournament_name)
    tournament_slug = tournament_key.replace(" ", "_")

    # Searches a wide-enough window around "today" to reliably catch
    # this week's event without needing tournament dates hand-entered
    # in config.json — PGA Tour events run within a single week, so +/-
    # 4 days from now always contains the current one, whichever day
    # mid-tournament this happens to run on.
    now = datetime.now(timezone.utc)
    window_start = (now - timedelta(days=4)).strftime("%Y%m%d")
    window_end = (now + timedelta(days=4)).strftime("%Y%m%d")

    try:
        event_id = find_event_id(window_start, window_end,
                                  tournament_name=tournament_name, debug=debug)
    except Exception as e:
        print(f"Could not find this week's ESPN event ({e}) — skipping results update. "
              f"(Nothing in update_data.py's own output is affected by this.)")
        return

    try:
        meta = get_event_metadata(event_id, debug=debug)
    except Exception as e:
        print(f"Could not fetch event metadata for {event_id} ({e}) — skipping results update.")
        return

    try:
        snapshots = get_tournament_snapshots(event_id, cfg["year"], debug=debug)
    except Exception as e:
        print(f"Could not fetch ESPN round results ({e}) — skipping results update.")
        return

    print(f"Fetched real-time ESPN snapshots for {len(snapshots)} players "
          f"in '{tournament_name}' (event {event_id}).")

    store = load_store()
    graded_rounds_by_player = {}
    for snap in snapshots:
        player_key = normalize_name(snap["player_name"])
        rounds = update_and_diff(store, tournament_key, player_key, snap)
        graded_rounds_by_player[player_key] = rounds
    save_store(store)

    archived_paths = list_archived_snapshots(tournament_slug)
    if not archived_paths:
        print(f"No archived props snapshots found yet for '{tournament_name}' "
              f"(data/props_history/{tournament_slug}_*.json) — nothing to grade "
              f"today. This is normal the first day a tournament is configured; "
              f"update_data.py archives one each day it runs.")

    # Which round is the snapshot being archived TODAY predicting? Answered
    # from what the field has actually finished right now (see
    # predicted_round_from_field) rather than from the calendar, because
    # the calendar guess breaks for events in other timezones — e.g. an
    # event in Japan finishes each round before the 13:00 UTC cron runs.
    # Stamped into the archived file once, the first time this runs on the
    # day it was generated, so later reruns can't change the answer.
    today_utc = now.strftime("%Y-%m-%d")
    predicted_round_today = predicted_round_from_field(
        [s.get("rounds_completed", 0) for s in snapshots], meta["total_rounds"])

    total_new_rows = 0
    for path in archived_paths:
        with open(path) as f:
            snapshot = json.load(f)

        if "predicted_round" not in snapshot and snapshot["generated_at"][:10] == today_utc:
            snapshot["predicted_round"] = predicted_round_today
            with open(path, "w") as f:
                json.dump(snapshot, f, indent=2)
            print(f"  Stamped {path} with predicted_round={predicted_round_today} "
                  f"(None = tournament already complete).")

        if "predicted_round" in snapshot:
            round_number = snapshot["predicted_round"]
            if round_number is None:
                if debug:
                    print(f"  {path}: archived after the final round finished — skipping.")
                continue
        else:
            # Older snapshot with no stamp (archived before this existed, or
            # a day this script didn't run): fall back to the calendar guess,
            # but a date outside the tournament's own days means "nothing to
            # grade" — never clamp it to the first or last round.
            guess = calendar_round_guess(
                snapshot["generated_at"], meta["start_date"], meta["total_rounds"])
            if guess is None:
                if debug:
                    print(f"  {path}: unstamped and dated outside the tournament — skipping.")
                continue
            print(f"  {path}: no predicted_round stamp — using calendar guess (round {guess}). "
                  f"Fine for US-timezone events; verify for events in other timezones.")
            round_number = guess

        rows = grade_tournament_day(snapshot, graded_rounds_by_player, round_number)
        date_str = snapshot["generated_at"][:10]
        new_rows = append_graded_rows(rows, date_str, tournament_name)
        total_new_rows += new_rows
        if debug:
            print(f"  {path} (round {round_number}): {len(rows)} gradable, "
                  f"{new_rows} newly logged.")

    print(f"Logged {total_new_rows} newly-gradable picks across "
          f"{len(archived_paths)} archived day(s).")

    summary = build_results_summary()
    write_results_json(summary)
    print(f"Wrote results summary to docs/data/results.json "
          f"(overall: {summary['overall']['hits']}/{summary['overall']['n']} "
          f"= {summary['overall']['hit_rate']}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    sys.exit(main(debug=args.debug))
