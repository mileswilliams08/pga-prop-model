"""
One-time cleanup: removes already-graded rows from data/graded_results.csv
whose line can't be shown to be a normal two-sided PrizePicks line (a
goblin/demon, which can only be taken Over, or an ambiguous multi-line
prop). Uses the same check the grader now applies going forward
(grade_results.is_untakeable_line), against the archived snapshot each row
came from (data/props_history/), so a pruned row can't reappear on the next
update_results.py run.

Dry run by default (prints what WOULD be removed). Pass --apply to write.

    python prune_one_way.py            # preview
    python prune_one_way.py --apply    # rewrite data/graded_results.csv
    python prune_one_way.py --apply --rebuild   # also rebuild docs/data/results.json
"""

import argparse
import json
import os
import sys
from collections import Counter

import pandas as pd

from course_history import normalize_tournament_name
from grade_results import (GRADED_LOG_PATH, PROPS_HISTORY_DIR, is_untakeable_line,
                           build_results_summary, write_results_json)
from match_props import normalize_name


def main(apply: bool = False, rebuild: bool = False) -> int:
    if not os.path.exists(GRADED_LOG_PATH):
        print(f"{GRADED_LOG_PATH} not found — nothing to prune.")
        return 0

    df = pd.read_csv(GRADED_LOG_PATH)
    snapshots = {}

    def entry_for(row):
        slug = normalize_tournament_name(row["tournament"]).replace(" ", "_")
        path = f"{PROPS_HISTORY_DIR}/{slug}_{row['date']}.json"
        if path not in snapshots:
            snapshots[path] = None
            if os.path.exists(path):
                with open(path) as f:
                    snap = json.load(f)
                snapshots[path] = {normalize_name(p["player"]): p for p in snap.get("props", [])}
        by_player = snapshots[path]
        if by_player is None:
            return None
        return by_player.get(normalize_name(row["player"]))

    drop = []
    missing_snapshot = 0
    for idx, row in df.iterrows():
        if row.get("source") != "prizepicks":
            continue
        entry = entry_for(row)
        if entry is None:
            missing_snapshot += 1
            continue  # can't tell — keep the row
        if is_untakeable_line(entry, row["category"], use_heuristic=True):
            drop.append(idx)

    print(f"{len(df)} graded rows; {len(drop)} flagged as one-way/ambiguous; "
          f"{missing_snapshot} rows kept because their snapshot file wasn't found.")
    if drop:
        flagged = df.loc[drop]
        print("Flagged by tournament/date:",
              dict(Counter(zip(flagged["tournament"], flagged["date"]))))
        print("Flagged by category:", dict(Counter(flagged["category"])))

    if not apply:
        print("Dry run — nothing written. Re-run with --apply to remove them.")
        return 0

    df.drop(index=drop).to_csv(GRADED_LOG_PATH, index=False)
    print(f"Wrote {GRADED_LOG_PATH} ({len(df) - len(drop)} rows kept).")
    if rebuild:
        summary = build_results_summary()
        write_results_json(summary)
        print(f"Rebuilt docs/data/results.json (overall: {summary['overall']}).")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    sys.exit(main(apply=args.apply, rebuild=args.rebuild))
