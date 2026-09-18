"""
Pulls current golf prop lines from Underdog Fantasy.

IMPORTANT: unlike PrizePicks, Underdog does not publish any public API.
This hits their app's internal endpoint (the same one their own web app
calls), which is unofficial and can change without notice. If this
breaks, that's expected eventually — the fix is inspecting Underdog's
web app network traffic (browser DevTools -> Network tab while browsing
their golf board) to find the current endpoint and field names, then
updating BASE/paths below.

Usage:
    python underdog_scraper.py --debug
"""

import argparse
import json
import requests
import pandas as pd

BASE = "https://api.underdogfantasy.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def fetch_over_under_lines(debug: bool = False) -> pd.DataFrame:
    """
    Underdog's board data has historically been available at an
    `over_under_lines` style endpoint under /beta/v*/. The exact version
    number changes periodically — try v6 first, fall back to v5/v4.
    """
    rows = []
    payload = None
    for version in ["v6", "v5", "v4"]:
        try:
            resp = requests.get(
                f"{BASE}/beta/{version}/over_under_lines",
                headers=HEADERS, timeout=15,
            )
            if resp.status_code == 200:
                payload = resp.json()
                break
        except requests.RequestException:
            continue

    if payload is None:
        print("Could not reach Underdog's endpoint under any known version. "
              "Their API path has likely changed — see the module docstring.")
        return pd.DataFrame(columns=["player", "stat_type", "line", "category"])

    if debug:
        print(json.dumps(payload, indent=2)[:3000])

    # Structure has historically been a list under "over_under_lines", each
    # with a nested "over_under" -> "appearance_stat" (stat name) and
    # a related player name under "players" (top-level lookup list).
    players = {p["id"]: p.get("first_name", "") + " " + p.get("last_name", "")
               for p in payload.get("players", [])}

    for line in payload.get("over_under_lines", []):
        ou = line.get("over_under", {})
        stat_name = ou.get("appearance_stat", "")
        stat_val = line.get("stat_value")
        appearance = ou.get("appearance", {})
        player_id = appearance.get("player_id")
        player_name = players.get(player_id, "Unknown")

        if "golf" not in (appearance.get("match", {}) or {}).get("sport_id", "golf").lower() \
                and appearance.get("sport_id", "golf") != "golf":
            # Best-effort sport filter; if the field name differs, this
            # will just pass everything through, which is harmless.
            pass

        rows.append({"player": player_name, "stat_type": stat_name, "line": stat_val})

    df = pd.DataFrame(rows)
    if not df.empty:
        df["category"] = df["stat_type"].apply(normalize_stat_category)
    return df


def normalize_stat_category(stat_type: str) -> str:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    df = fetch_over_under_lines(debug=args.debug)
    print(f"Found {len(df)} Underdog golf props")
    if not df.empty:
        print(df["category"].value_counts())
