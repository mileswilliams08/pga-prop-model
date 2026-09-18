"""
Pulls current golf prop lines from PrizePicks' public API.

This is a genuinely public, unauthenticated JSON API (no login, no key) —
much more stable than scraping a rendered webpage. That said, it's not
officially documented, so field names could shift; if this starts
returning empty results, run with --debug to dump the raw response and
check the structure hasn't changed.

Usage:
    python prizepicks_scraper.py --debug   # inspect raw structure
    python prizepicks_scraper.py           # normal run, returns a DataFrame
"""

import argparse
import json
import time
import requests
import pandas as pd

BASE = "https://api.prizepicks.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def find_golf_league_id() -> str:
    resp = requests.get(f"{BASE}/leagues", headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()["data"]
    for league in data:
        name = league["attributes"].get("name", "").lower()
        if "golf" in name or "pga" in name:
            return league["id"]
    raise RuntimeError("Couldn't find a golf league in PrizePicks' league list.")


def fetch_projections(league_id: str) -> pd.DataFrame:
    resp = requests.get(
        f"{BASE}/projections",
        params={"league_id": league_id, "per_page": 250},
        headers=HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json()

    # JSON:API format: projections reference players via "included"
    players = {
        item["id"]: item["attributes"].get("name")
        for item in payload.get("included", [])
        if item.get("type") == "new_player"
    }

    rows = []
    for proj in payload.get("data", []):
        attrs = proj["attributes"]
        player_id = proj["relationships"]["new_player"]["data"]["id"]
        rows.append({
            "player": players.get(player_id, "Unknown"),
            "stat_type": attrs.get("stat_type"),
            "line": attrs.get("line_score"),
            "odds_type": attrs.get("odds_type", "standard"),
        })

    return pd.DataFrame(rows)


def normalize_stat_category(stat_type: str) -> str:
    """Maps PrizePicks' stat_type text to our model's four categories."""
    s = (stat_type or "").lower()
    if "birdie" in s:
        return "birdies"
    if "green" in s or "gir" in s:
        return "gir"
    if "fairway" in s:
        return "fairways"
    if "stroke" in s or "score" in s:
        return "strokes"
    return "other"


def get_golf_props(debug: bool = False) -> pd.DataFrame:
    league_id = find_golf_league_id()
    time.sleep(0.5)
    df = fetch_projections(league_id)
    if debug:
        print(json.dumps(df.head(20).to_dict(orient="records"), indent=2))
    df["category"] = df["stat_type"].apply(normalize_stat_category)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    df = get_golf_props(debug=args.debug)
    print(f"Found {len(df)} PrizePicks golf props "
          f"({df['category'].value_counts().to_dict()})")
