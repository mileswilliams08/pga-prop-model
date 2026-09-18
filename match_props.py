"""
Matches PrizePicks/Underdog prop lines to our model's players by name,
so the website can show "here's the platform's line, here's our model's
probability" without you typing anything in.

Name matching is intentionally simple (lowercase, strip punctuation) —
DFS platforms are generally consistent with "First Last" naming, but if
a player's name doesn't match, it just won't show a platform line for
them (nothing breaks, it just falls back to no line pulled).
"""

import re
import pandas as pd


def normalize_name(name: str) -> str:
    name = (name or "").lower()
    name = re.sub(r"[^a-z\s]", "", name)
    return re.sub(r"\s+", " ", name).strip()


def attach_platform_lines(model_props: list, platform_dfs: dict) -> list:
    """
    model_props: the list built in update_data.py (one dict per player,
        each with gir/fairways/birdies/strokes sub-dicts).
    platform_dfs: dict like {"prizepicks": df, "underdog": df}, each with
        columns [player, stat_type, line, category].

    Adds a "platform_lines" list to each player's entry, e.g.:
        [{"source": "prizepicks", "category": "birdies", "line": 3.5}, ...]
    """
    lookup = {}  # normalized_name -> list of {source, category, line}
    for source, df in platform_dfs.items():
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            key = normalize_name(row["player"])
            lookup.setdefault(key, []).append({
                "source": source,
                "category": row["category"],
                "line": row["line"],
            })

    for player_entry in model_props:
        key = normalize_name(player_entry["player"])
        player_entry["platform_lines"] = lookup.get(key, [])

    return model_props
