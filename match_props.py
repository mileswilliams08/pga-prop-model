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


def normalize_name(name) -> str:
    # `name or ""` doesn't catch a missing value coming from pandas as
    # NaN (a float) — NaN is truthy in Python, so `.lower()` below would
    # crash on it instead of being treated as empty. Handle that
    # explicitly rather than assuming the input is always a real string.
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = re.sub(r"[^a-z\s]", "", name)
    return re.sub(r"\s+", " ", name).strip()


# Goblin/demon lines can only be taken one way (Over), so they are never a
# two-sided line the model can price. Platforms without an odds_type column
# (Underdog) are treated as all-standard.
ONE_WAY_ODDS_TYPES = ("goblin", "demon")


def is_one_way_row(row) -> bool:
    try:
        v = row["odds_type"]
    except (KeyError, IndexError):
        return False
    return str(v or "standard").strip().lower() in ONE_WAY_ODDS_TYPES


def build_line_lookup(platform_dfs: dict, preferred_order=("prizepicks", "underdog")) -> dict:
    """
    platform_dfs: dict like {"prizepicks": df, "underdog": df}, each with
    columns [player, stat_type, line, category].

    Returns normalized_name -> {category: {"source": str, "line": float}}
    — the SINGLE real line to actually evaluate the model's probability
    against for that player+stat, picking whichever source in
    preferred_order shows up first when more than one platform posts a
    line for the same player+category.

    This exists because build_props() used to compute every player's
    probability against the same fixed line from config.json's
    "prop_lines" (e.g. birdies=3.5 for everyone), regardless of what a
    platform actually posted for that specific player (often much higher
    for a good player at a birdie-friendly course). That meant the
    displayed probability answered a different question than the one
    actually being bet on. Feeding this lookup into build_props() fixes
    that: whenever a real line exists, the model's probability is for
    THAT line, not a generic placeholder. "other"-category markets (Pars,
    Bogeys or Worse, etc. — PrizePicks props with no equivalent model
    stat) are skipped since there is no model prop to evaluate them
    against.
    """
    lookup = {}
    for source in preferred_order:
        df = platform_dfs.get(source)
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            category = row["category"]
            if category not in ("gir", "fairways", "birdies", "strokes"):
                continue
            if is_one_way_row(row):
                continue  # goblin/demon: one-way line, not a real two-sided prop
            key = normalize_name(row["player"])
            player_lines = lookup.setdefault(key, {})
            if category not in player_lines:  # first source in preferred_order wins
                player_lines[category] = {"source": source, "line": row["line"],
                                          "odds_type": "standard"}
    return lookup


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
            entry = {
                "source": source,
                "category": row["category"],
                "line": row["line"],
            }
            if "odds_type" in df.columns:
                entry["odds_type"] = str(row["odds_type"] or "standard").strip().lower()
            lookup.setdefault(key, []).append(entry)

    for player_entry in model_props:
        key = normalize_name(player_entry["player"])
        player_entry["platform_lines"] = lookup.get(key, [])

    return model_props
