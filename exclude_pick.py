"""
Manually take specific graded picks out of every hit rate (e.g. props that
turned out to be PrizePicks goblins/demons, which can only be taken Over).
The graded log (data/graded_results.csv) is never edited — exclusions live
in data/excluded_picks.csv and are applied when docs/data/results.json is
built, so they survive every daily run.

    python exclude_pick.py list --round 2                  # find the pick(s)
    python exclude_pick.py list --round 2 --category strokes
    python exclude_pick.py add "Max Homa" strokes 2        # exclude one pick
    python exclude_pick.py remove "Max Homa" strokes 2     # undo it
    python exclude_pick.py show                            # everything excluded

Defaults to the tournament in config.json; use --tournament to override.
Add --rebuild to add/remove to refresh docs/data/results.json right away.
"""

import argparse
import csv
import json
import os
import sys

import pandas as pd

from course_history import normalize_tournament_name
from grade_results import (GRADED_LOG_PATH, EXCLUDED_PICKS_PATH, build_results_summary,
                           write_results_json)
from match_props import normalize_name

FIELDS = ["tournament", "round", "player", "category"]


def _tournament(args) -> str:
    if args.tournament:
        return args.tournament
    with open("config.json") as f:
        return json.load(f)["tournament_name"]


def _read_exclusions() -> list:
    if not os.path.exists(EXCLUDED_PICKS_PATH):
        return []
    with open(EXCLUDED_PICKS_PATH, newline="") as f:
        return list(csv.DictReader(f))


def _write_exclusions(rows: list) -> None:
    os.makedirs(os.path.dirname(EXCLUDED_PICKS_PATH), exist_ok=True)
    with open(EXCLUDED_PICKS_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def _matches(df, tournament, rnd, player=None, category=None):
    m = df["tournament"].apply(lambda t: normalize_tournament_name(str(t))) == normalize_tournament_name(tournament)
    if rnd is not None:
        m &= pd.to_numeric(df["round"], errors="coerce") == int(rnd)
    if player:
        m &= df["player"].apply(normalize_name) == normalize_name(player)
    if category:
        m &= df["category"].str.lower() == category.lower()
    return df[m]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("list", "add", "remove", "show"):
        sp = sub.add_parser(name)
        sp.add_argument("--tournament")
        if name == "list":
            sp.add_argument("--round", type=int)
            sp.add_argument("--category")
        if name in ("add", "remove"):
            sp.add_argument("player")
            sp.add_argument("category")
            sp.add_argument("round", type=int)
            sp.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "show":
        rows = _read_exclusions()
        print(f"{len(rows)} excluded picks")
        for r in rows:
            print(f"  {r['tournament']} | round {r['round']} | {r['player']} | {r['category']}")
        return 0

    tournament = _tournament(args)
    df = pd.read_csv(GRADED_LOG_PATH) if os.path.exists(GRADED_LOG_PATH) else pd.DataFrame()

    if args.cmd == "list":
        sel = _matches(df, tournament, args.round, category=args.category)
        print(f"{len(sel)} graded picks for {tournament}"
              + (f", round {args.round}" if args.round else "")
              + (f", {args.category}" if args.category else ""))
        for _, r in sel.sort_values(["round", "player", "category"]).iterrows():
            print(f"  r{int(r['round'])} | {r['player']} | {r['category']} | line {r['line']} | "
                  f"{r['picked_side']} | actual {r['actual_value']} | {r['result']}")
        return 0

    # add / remove
    hit = _matches(df, tournament, args.round, player=args.player, category=args.category)
    existing = _read_exclusions()
    key = (normalize_tournament_name(tournament), str(args.round), normalize_name(args.player), args.category.lower())

    def _k(r):
        return (normalize_tournament_name(r["tournament"]), str(r["round"]), normalize_name(r["player"]),
                r["category"].lower())

    if args.cmd == "add":
        if hit.empty:
            print(f"No graded pick found for {args.player} / {args.category} / round {args.round} "
                  f"in {tournament}. Use `list` to find the exact name. Nothing changed.")
            return 1
        if any(_k(r) == key for r in existing):
            print("Already excluded.")
            return 0
        existing.append({"tournament": tournament, "round": str(args.round),
                         "player": hit.iloc[0]["player"], "category": args.category.lower()})
        _write_exclusions(existing)
        r = hit.iloc[0]
        print(f"Excluded: {r['player']} | {r['category']} | round {args.round} | line {r['line']} | "
              f"{r['picked_side']} | {r['result']}")
    else:
        kept = [r for r in existing if _k(r) != key]
        if len(kept) == len(existing):
            print("That pick wasn't in the exclusion list. Nothing changed.")
            return 1
        _write_exclusions(kept)
        print(f"Removed {args.player} / {args.category} / round {args.round} from exclusions.")

    if args.rebuild:
        summary = build_results_summary()
        write_results_json(summary)
        print(f"Rebuilt docs/data/results.json (overall: {summary['overall']}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
