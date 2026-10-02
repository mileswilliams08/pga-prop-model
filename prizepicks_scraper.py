"""
Pulls current golf (PGA) prop lines from PrizePicks' real public API.

PREVIOUS APPROACH (driving a headless Chromium browser at
api.prizepicks.com) is retired as of this version. The actual fix
turned out to be much simpler and far more durable: PrizePicks serves
its live board from a DIFFERENT, undocumented host —
partner-api.prizepicks.com — that is not behind the same bot-management
(DataDome) PrizePicks' main api.prizepicks.com and
api.prod01.universe.prizepicks.com both sit behind. A plain, honest
`requests` GET (real User-Agent, no stealth, no fingerprint spoofing, no
cookies) gets a normal 200 with real JSON from that host. This was
confirmed by a real capture of a working scraper hitting it — note the
"partner" in the hostname specifically, not "prod01.universe" (an
earlier version of this file guessed the latter from a related
project's hardcoded default and got a DataDome CAPTCHA redirect
(geo.captcha-delivery.com) instead — DIFFERENT hosts, DIFFERENT
protection, and only "partner-api" was ever confirmed working).

No browser, no Playwright, nothing to "work around" a challenge with —
which also means nothing here is likely to go stale the way Underdog's
client-version header does. If this host DOES start blocking requests
again, the fix is the same kind of fix as before: capture a fresh
working request from DevTools (Network tab, filter by "projections")
and compare the URL/params/headers against what's here.

Response shape: PrizePicks' API is JSON:API — a flat "data" list of
"projection" resources (one row per prop) whose player/team/game info
lives in a separate "included" list, cross-referenced by
(type, id) via each projection's "relationships". This module resolves
that itself rather than pulling in a full JSON:API client library,
since only a few relationship types (new_player, game) actually matter
here.

Usage:
    python prizepicks_scraper.py --debug   # inspect raw structure, league breakdown
    python prizepicks_scraper.py           # normal run, prints PGA props found
"""

import argparse
import json
import pandas as pd
import requests

BOARD_URL = "https://partner-api.prizepicks.com/projections"
GOLF_LEAGUE = "PGA"  # PrizePicks also runs a separate "EUROGOLF" league — not pulled here.

USER_AGENT = "pga-prop-model/1.0 (read-only board refresh; github.com user project)"

# PrizePicks' real per-round golf markets, as seen on a live board capture
# (2026-10-02). Anything not listed here (Pars, Bogeys or Worse, combo/
# matchup markets) maps to "other" rather than being dropped — same
# convention as underdog_scraper.normalize_stat_category — so new/renamed
# markets show up as "other" instead of silently vanishing.
#
# "Birdies or Better Matchup" is deliberately NOT mapped to "birdies"
# here (it falls through to the "other" default instead). It looks like
# a normal fixed-threshold birdies prop — it even shows a numeric
# "line" like 0.5 — but it's actually a head-to-head comparison between
# two specific players' birdie counts ("will Player A have more/less
# birdies than Player B"), not "will this player clear N birdies." Our
# model's birdies_or_better_prop() only knows how to evaluate the
# latter, so treating a matchup line as a real birdies line produced a
# meaningless probability against an unrelated 0.5 threshold. Mapping
# it to "other" means match_props.build_line_lookup() (which only
# accepts "gir"/"fairways"/"birdies"/"strokes") automatically excludes
# it from being used as a model line, while attach_platform_lines()
# still records it under platform_lines for reference.
_CATEGORY_BY_MARKET = {
    "greens in regulation": "gir",
    "fairways hit": "fairways",
    "driving accuracy": "fairways",
    "birdies or better": "birdies",
    "strokes": "strokes",
}


def normalize_stat_category(market: str) -> str:
    return _CATEGORY_BY_MARKET.get((market or "").strip().lower(), "other")


def _attributes_of(resource) -> dict:
    attrs = resource.get("attributes") if isinstance(resource, dict) else None
    return attrs if isinstance(attrs, dict) else {}


def _relationship_id(resource: dict, name: str):
    """(type, id) of a to-one relationship, or None if absent/malformed."""
    rel = (resource.get("relationships") or {}).get(name)
    data = rel.get("data") if isinstance(rel, dict) else None
    if not isinstance(data, dict):
        return None
    rid = data.get("id")
    rtype = data.get("type")
    if rid is None or rtype is None:
        return None
    return (str(rtype), str(rid))


def _build_included_index(included: list) -> dict:
    """(type, id) -> resource dict, for resolving relationships."""
    index = {}
    for res in included:
        if isinstance(res, dict) and res.get("type") is not None and res.get("id") is not None:
            index[(str(res["type"]), str(res["id"]))] = res
    return index


def fetch_board(board_url: str = BOARD_URL, timeout: float = 20.0, debug: bool = False) -> dict:
    """One plain GET of the full JSON:API board (all leagues — PrizePicks
    appears to ignore per_page, see fetch_golf_projections's docstring).
    Raises requests.RequestException / ValueError on failure; callers
    handle that (see get_golf_props)."""
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    params = {"per_page": 250, "single_stat": "true"}
    resp = requests.get(board_url, headers=headers, params=params, timeout=timeout)

    if debug:
        print(f"  -> GET {resp.url}")
        print(f"     HTTP status: {resp.status_code}")
        print(f"     Content-Type: {resp.headers.get('content-type')}")
        print(f"     Body length: {len(resp.text)} chars")

    if resp.status_code != 200:
        raise RuntimeError(
            f"HTTP {resp.status_code} from {board_url} — if this used to work and "
            f"just broke, PrizePicks may have changed hosts again or tightened "
            f"blocking on this one too. Capture a fresh working request from "
            f"DevTools (Network tab, filter 'projections') and compare against "
            f"this module's BOARD_URL/headers. First 300 chars of body: "
            f"{resp.text[:300]!r}"
        )
    payload = resp.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise RuntimeError(
            f"200 OK but the body doesn't look like a JSON:API board (missing/"
            f"malformed top-level 'data' list) — PrizePicks may have changed "
            f"their response shape. First 300 chars: {resp.text[:300]!r}"
        )
    return payload


def parse_golf_projections(payload: dict, league: str = GOLF_LEAGUE, debug: bool = False) -> pd.DataFrame:
    """Walks every 'projection' resource in payload['data'], resolves its
    player via payload['included'], and keeps only rows whose player's
    league matches (default "PGA"). Returns columns [player, stat_type,
    line, category] — the exact shape match_props.attach_platform_lines
    expects (same as underdog_scraper's output)."""
    data = payload.get("data") or []
    included = payload.get("included") or []
    index = _build_included_index(included)

    if debug:
        from collections import Counter
        league_counts = Counter()
        for res in included:
            if res.get("type") == "new_player":
                league_counts[_attributes_of(res).get("league")] += 1
        print(f"  Leagues seen among {len(included)} included resources "
              f"(by player count): {dict(league_counts.most_common(15))}")

    rows = []
    skipped_no_player = 0
    skipped_wrong_league = 0
    skipped_no_market_or_line = 0

    for p in data:
        if not isinstance(p, dict) or p.get("type") != "projection":
            continue
        attrs = _attributes_of(p)

        player_ref = _relationship_id(p, "new_player")
        player = index.get(player_ref) if player_ref else None
        if player is None:
            skipped_no_player += 1
            continue
        player_attrs = _attributes_of(player)
        player_league = (player_attrs.get("league") or "").strip()
        if player_league != league:
            skipped_wrong_league += 1
            continue

        display_name = (player_attrs.get("display_name") or player_attrs.get("name") or "").strip()
        market = (attrs.get("stat_display_name") or attrs.get("stat_type") or "").strip()
        line = attrs.get("line_score")
        if not display_name or not market or line is None:
            skipped_no_market_or_line += 1
            continue
        try:
            line = float(line)
        except (TypeError, ValueError):
            skipped_no_market_or_line += 1
            continue

        rows.append({"player": display_name, "stat_type": market, "line": line})

    if debug:
        print(f"  Parsed {len(rows)} {league} rows from {len(data)} total projections "
              f"({skipped_wrong_league} other-league, {skipped_no_player} unresolved player, "
              f"{skipped_no_market_or_line} missing market/line).")

    df = pd.DataFrame(rows, columns=["player", "stat_type", "line"])
    if not df.empty:
        df["category"] = df["stat_type"].apply(normalize_stat_category)
    else:
        df["category"] = pd.Series(dtype="object")
    return df


def get_golf_props(league: str = GOLF_LEAGUE, debug: bool = False) -> pd.DataFrame:
    """The function update_data.py calls. Never raises — a fetch/parse
    failure returns an empty DataFrame with the right columns, same
    fail-soft contract the old Playwright version had, so one platform
    going down doesn't take the whole site update with it."""
    try:
        payload = fetch_board(debug=debug)
    except Exception as e:
        print(f"PrizePicks fetch failed: {e}")
        return pd.DataFrame(columns=["player", "stat_type", "line", "category"])

    return parse_golf_projections(payload, league=league, debug=debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--league", default=GOLF_LEAGUE,
                         help=f"PrizePicks league name to filter to (default {GOLF_LEAGUE!r}; "
                              f"PrizePicks also separately runs 'EUROGOLF').")
    args = parser.parse_args()

    df = get_golf_props(league=args.league, debug=args.debug)
    if df.empty:
        print(f"Found 0 PrizePicks {args.league} props (empty/closed board right now, "
              f"or the request was blocked — rerun with --debug for details).")
    else:
        print(f"Found {len(df)} PrizePicks {args.league} props "
              f"({df['category'].value_counts().to_dict()})")
        if args.debug:
            print(json.dumps(df.head(20).to_dict(orient="records"), indent=2))
