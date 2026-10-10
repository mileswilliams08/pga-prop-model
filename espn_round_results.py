"""
Pulls REAL per-round results for a live/recent PGA Tour tournament from
ESPN's undocumented "playersummary" endpoint — the per-player,
round-by-round detail grade_results.py needs to check whether the
model's picks actually hit.

Why this isn't just espn_scraper.py again: that scrapes SEASON-TOTAL
stats (one row per player, averaged over the whole year) from a
completely different page. Grading needs TOURNAMENT-SPECIFIC,
ROUND-BY-ROUND numbers instead, which lives at a different ESPN
endpoint entirely:

    https://site.web.api.espn.com/apis/site/v2/sports/golf/pga/leaderboard/{event_id}/playersummary?season={year}&player={espn_player_id}

What this endpoint actually gives us, confirmed by hand against a live
tournament (2026-10-03, Bank of Utah Championship, event 401850915):

  - STROKES per round: exact (each round has its own `value` field in
    a `rounds` array, keyed by `period` = round number).
  - BIRDIES-OR-BETTER per round: exact. Each round entry has its OWN
    `statistics.categories[0].stats` block with real `birdies` and
    `eagles` counts for JUST that round — no estimation needed. We sum
    birdies + eagles for "birdies or better".
  - GIR per round: NOT given directly. Only a single CUMULATIVE
    gir%/girPoss pair is exposed (covering every round completed so
    far, as of whenever you ask). Getting one round's own GIR count
    means snapshotting this cumulative pair after each round and
    subtracting the previous snapshot — see snapshot_store.py. This is
    EXACT once two consecutive snapshots exist, since girPoss (holes
    played) is a real integer, not a percentage.
  - FAIRWAYS HIT per round: same cumulative-diffing idea, but ESPN only
    exposes a cumulative PERCENTAGE (driveAccuracyPct), never a raw
    count — so the fairways-hit NUMERATOR is always a rounded ESTIMATE,
    even after diffing. The DENOMINATOR (fairways possible that round)
    is exact, though: each hole's `par` is in that round's `linescores`,
    so real par-3s (no driver needed, no fairway to hit) are excluded
    precisely instead of assuming a flat "14 per round". Every row this
    module returns is tagged so callers never present this as more
    precise than it is.

Finding the right IDs:
  - event_id: find_event_id() below, via ESPN's scoreboard endpoint,
    searched by date range + a loose name match.
  - espn_player_id: get_field() below already returns every player in
    the tournament's field with their ESPN id from the leaderboard
    endpoint — no separate name-to-id scraper needed.

Fails soft throughout: a missing/unmatched player, a round that hasn't
finished yet, or a network hiccup all just mean that piece is skipped
(logged, not raised) — grading downstream treats "no real result yet"
as "can't grade this one today", not an error.
"""

import argparse
import json
import re
import requests

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard"
# NOTE: unlike SCOREBOARD_URL and PLAYER_SUMMARY_URL below, this endpoint
# does NOT have a "/pga/" path segment — confirmed against a live
# response; adding one 404s.
LEADERBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/golf/leaderboard"
PLAYER_SUMMARY_URL = ("https://site.web.api.espn.com/apis/site/v2/sports/golf/pga/"
                       "leaderboard/{event_id}/playersummary")

USER_AGENT = "pga-prop-model/1.0 (read-only results check; github.com user project)"
_REQUEST_TIMEOUT = 20.0


def _get_json(url: str, params: dict = None, debug: bool = False) -> dict:
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, params=params, timeout=_REQUEST_TIMEOUT)
    if debug:
        print(f"  -> GET {resp.url}  [{resp.status_code}]")
    resp.raise_for_status()
    return resp.json()


def _loose_match(a: str, b: str) -> bool:
    norm = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())
    a, b = norm(a), norm(b)
    return a != "" and (a in b or b in a)


def find_event_id(start_date: str, end_date: str, tournament_name: str = None,
                   debug: bool = False) -> str:
    """
    start_date/end_date: "YYYYMMDD". Searches ESPN's scoreboard for every
    event in that window and returns the id of the one whose name loosely
    matches tournament_name (substring match, case/punctuation-insensitive
    — handles "Bank of Utah Championship" vs a scraped "Bank of Utah
    Championship presented by..." variant). If tournament_name is None,
    or nothing matches, returns the first STROKE-PLAY event found (skips
    team events like the Presidents Cup/Ryder Cup, which don't have a
    per-player leaderboard in the shape this module expects).

    Raises RuntimeError if nothing usable is found — unlike the per-player
    functions below, there's no reasonable fallback for "couldn't even
    find this week's tournament."
    """
    payload = _get_json(SCOREBOARD_URL, params={"dates": f"{start_date}-{end_date}"}, debug=debug)
    events = payload.get("events") or []

    candidates = []
    for ev in events:
        name = ev.get("name", "")
        is_team_event = any(
            comp.get("type", {}).get("id") in ("2",)  # best-effort; see fallback below
            for comp in (ev.get("competitions") or [])
        )
        # More reliable team-event signal: competitors with type "pair"/"team"
        # rather than individual athletes.
        competitors = ((ev.get("competitions") or [{}])[0].get("competitors") or [])
        is_team_event = is_team_event or any(
            c.get("type") in ("pair", "team") for c in competitors
        )
        if is_team_event:
            continue
        candidates.append(ev)

    if tournament_name:
        for ev in candidates:
            if _loose_match(tournament_name, ev.get("name", "")):
                if debug:
                    print(f"Matched '{tournament_name}' -> event {ev['id']} ({ev['name']})")
                return ev["id"]

    if candidates:
        if debug:
            print(f"No name match for '{tournament_name}' — using first stroke-play "
                  f"event found: {candidates[0]['name']} ({candidates[0]['id']})")
        return candidates[0]["id"]

    raise RuntimeError(
        f"No stroke-play PGA Tour event found between {start_date} and {end_date} "
        f"(tournament_name={tournament_name!r}). Check the date window, or ESPN's "
        f"scoreboard endpoint/response shape may have changed."
    )


def get_event_metadata(event_id: str, debug: bool = False) -> dict:
    """
    Returns {"start_date": iso str, "total_rounds": int} for this event
    — grade_results.infer_round_number needs the start date (to work out
    which calendar day of the tournament a given archived snapshot was
    predicting), and total_rounds caps that inference at the real
    tournament length instead of assuming every event is always 4
    rounds (some are 3, playoffs can differ, etc).
    """
    payload = _get_json(LEADERBOARD_URL, params={"event": event_id}, debug=debug)
    events = payload.get("events") or []
    if not events:
        raise RuntimeError(f"No event data found for event_id={event_id}.")
    ev = events[0]
    return {
        "start_date": ev.get("date"),
        "total_rounds": (ev.get("tournament") or {}).get("numberOfRounds", 4),
    }


def get_field(event_id: str, debug: bool = False) -> list:
    """
    Returns [{"player_name": str, "espn_player_id": str}, ...] for every
    player ESPN lists in this event's leaderboard — this is also how we
    avoid needing a separate name->id lookup/scraper.
    """
    payload = _get_json(LEADERBOARD_URL, params={"event": event_id}, debug=debug)
    events = payload.get("events") or []
    if not events:
        return []
    competitions = events[0].get("competitions") or []
    if not competitions:
        return []
    competitors = competitions[0].get("competitors") or []

    field = []
    for c in competitors:
        athlete = c.get("athlete") or {}
        name = athlete.get("displayName")
        pid = athlete.get("id")
        if name and pid:
            field.append({"player_name": name, "espn_player_id": str(pid)})

    if debug:
        print(f"Found {len(field)} players in event {event_id}'s field.")
    return field


def _round_fairways_possible(round_entry: dict) -> int:
    """Counts non-par-3 holes actually played this round, from that
    round's own linescores — the EXACT denominator for fairways-hit,
    independent of ESPN's cumulative-percentage limitation."""
    linescores = round_entry.get("linescores") or []
    count = 0
    for hole in linescores:
        par = hole.get("par")
        if par is not None and par != 3:
            count += 1
    return count


def _round_hole_counts(round_entry: dict, expected_strokes, expected_birdies_or_better):
    """
    Exact pars / bogeys-or-worse / course par for one finished round, from
    that round's own hole-by-hole linescores (hole "value" = strokes,
    hole "par" = par). Returns None -- and the caller simply records
    nothing -- unless the holes are complete AND they reconcile with the
    numbers ESPN gives separately (strokes total and birdies-or-better
    count). That check is deliberate: these field names were not
    confirmed against a raw capture, so a wrong guess shows up as a
    mismatch and a logged warning instead of silently wrong grading.
    """
    holes = round_entry.get("linescores") or []
    scores, pars = [], []
    for h in holes:
        try:
            scores.append(float(h.get("value")))
            pars.append(float(h.get("par")))
        except (TypeError, ValueError):
            return None
    if len(scores) != 18:
        return None
    if abs(sum(scores) - float(expected_strokes)) > 0.5:
        return None
    under = sum(1 for s, p in zip(scores, pars) if s < p)
    if expected_birdies_or_better is not None and under != expected_birdies_or_better:
        return None
    return {
        "pars": sum(1 for s, p in zip(scores, pars) if s == p),
        "bogeys_or_worse": sum(1 for s, p in zip(scores, pars) if s > p),
        "par_total": int(sum(pars)),
    }


def infer_course_par(event_id: str, year: int, field: list, max_players: int = 8,
                      debug: bool = False):
    """
    Course par for the event, read from any field player's posted hole pars
    (a round with 18 holes of linescores). Returns None if no round data
    exists yet (e.g. before round 1) -- callers fall back to config.json's
    "course_par" or skip par-dependent props.
    """
    for player in (field or [])[:max_players]:
        try:
            payload = _get_json(PLAYER_SUMMARY_URL.format(event_id=event_id),
                                params={"season": year, "player": player["espn_player_id"]},
                                debug=debug)
        except Exception:
            continue
        for r in payload.get("rounds") or []:
            holes = r.get("linescores") or []
            if len(holes) == 18 and all(h.get("par") is not None for h in holes):
                return int(sum(float(h["par"]) for h in holes))
    return None


def get_player_cumulative_snapshot(event_id: str, year: int, espn_player_id: str,
                                    player_name: str = "", debug: bool = False) -> dict:
    """
    Returns a snapshot of one player's results in this tournament AS OF
    RIGHT NOW:

        {
          "espn_player_id": str,
          "player_name": str,
          "rounds_completed": int,
          "per_round_exact": {
              round_num: {"strokes": float, "birdies_or_better": int,
                          "fairways_possible": int, "holes_played": int},
              ...
          },
          "cumulative_gir_pct": float or None,
          "cumulative_gir_poss": int or None,
          "cumulative_drive_accuracy_pct": float or None,
          "cumulative_fairways_possible": int,  # sum of exact per-round values above
        }

    Only rounds with an actual posted score (a `value` in that round's
    `linescores`-bearing entry) count toward rounds_completed — an
    upcoming/in-progress round with no score yet is simply not included,
    so this never fabricates a result for a round that hasn't finished.

    Returns None (logged, not raised) if this player has no data at all
    for this event — e.g. missed the field, or a name/id mismatch —
    since one unmatched player shouldn't stop grading everyone else.
    """
    try:
        payload = _get_json(
            PLAYER_SUMMARY_URL.format(event_id=event_id),
            params={"season": year, "player": espn_player_id},
            debug=debug,
        )
    except Exception as e:
        print(f"  Could not fetch ESPN round summary for {player_name or espn_player_id}: {e}")
        return None

    rounds = payload.get("rounds") or payload.get("roundResults") or []
    # Different ESPN responses have nested this under slightly different
    # keys across sports; fall back to scanning the whole payload for a
    # list of dicts that look like round entries (have "period").
    if not rounds:
        for v in payload.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "period" in v[0]:
                rounds = v
                break

    per_round_exact = {}
    for r in rounds:
        period = r.get("period")
        if period is None:
            continue
        holes_played = len(r.get("linescores") or [])
        # ESPN can return a PLACEHOLDER entry for a round that hasn't
        # started yet or is still in progress -- "value": 0 (not None!)
        # with holes_played: 0, rather than omitting the round entirely.
        # `value is None` alone doesn't catch that (0 is not None), and
        # treating it as a real completed round silently corrupts
        # grading downstream (confirmed 2026-10-04: a round-4 placeholder
        # like this got graded as a real result with actual_value=0
        # before round 4 had actually finished). Requiring a full 18
        # holes is the real signal that a round is actually done.
        if r.get("value") is None or holes_played < 18:
            continue  # round not actually completed yet
        stats_block = (r.get("statistics") or {}).get("categories") or []
        birdies = eagles = 0
        for cat in stats_block:
            for stat in cat.get("stats", []):
                if stat.get("name") == "birdies":
                    birdies = int(float(stat.get("displayValue", 0) or 0))
                elif stat.get("name") == "eagles":
                    eagles = int(float(stat.get("displayValue", 0) or 0))
        per_round_exact[int(period)] = {
            "strokes": float(r["value"]),
            "birdies_or_better": birdies + eagles,
            "fairways_possible": _round_fairways_possible(r),
            "holes_played": holes_played,
        }
        counts = _round_hole_counts(r, r["value"], birdies + eagles)
        if counts is not None:
            per_round_exact[int(period)].update(counts)

    # Cumulative (tournament-total-so-far) stats live in a separate
    # flat list under "stats" at the payload's top level — confirmed
    # against a live response (2026-10-04, Eric Cole, event 401850915).
    # NOT the same key as each round's own per-round "statistics" block
    # (nested, under "categories") — easy to mix up, so named distinctly
    # in variable names here on purpose.
    cumulative_stats = payload.get("stats") or []
    stat_by_name = {s.get("name"): s for s in cumulative_stats if isinstance(s, dict)}

    def _num(name):
        v = stat_by_name.get(name, {}).get("displayValue")
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    snapshot = {
        "espn_player_id": espn_player_id,
        "player_name": player_name or payload.get("profile", {}).get("displayName", ""),
        "rounds_completed": len(per_round_exact),
        "per_round_exact": per_round_exact,
        "cumulative_gir_pct": _num("gir"),
        "cumulative_gir_poss": _num("girPoss"),
        "cumulative_drive_accuracy_pct": _num("driveAccuracyPct"),
        "cumulative_fairways_possible": sum(
            r["fairways_possible"] for r in per_round_exact.values()
        ),
    }
    return snapshot


def get_tournament_snapshots(event_id: str, year: int, field: list = None,
                              debug: bool = False) -> list:
    """
    field: optional pre-fetched get_field() result, to avoid refetching
    the leaderboard when the caller already has it. Returns a list of
    get_player_cumulative_snapshot() results, skipping (with a printed
    reason) any player that failed — never raises for one bad player.
    """
    if field is None:
        field = get_field(event_id, debug=debug)

    snapshots = []
    for player in field:
        snap = get_player_cumulative_snapshot(
            event_id, year, player["espn_player_id"], player["player_name"], debug=debug)
        if snap is not None:
            snapshots.append(snap)
    return snapshots


if __name__ == "__main__":
    # This module's field-name assumptions (payload["rounds"], the
    # top-level "statistics" list, each round's own nested statistics
    # block) were reverse-engineered from an undocumented ESPN endpoint
    # found via web research, not from a byte-for-byte raw capture — see
    # the module docstring. Before relying on this for real grading,
    # run it once against a live, in-progress tournament and sanity
    # check the output; `--raw` dumps the untouched JSON for one player
    # so you can confirm (or fix) the field-name assumptions above
    # against what ESPN is actually returning right now.
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--find-event", nargs=2, metavar=("START_YYYYMMDD", "END_YYYYMMDD"),
                         help="Find this week's event id, e.g. --find-event 20261001 20261004")
    parser.add_argument("--tournament-name", default=None)
    parser.add_argument("--event-id", default=None,
                         help="Skip --find-event and use a known event id directly.")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--player-id", default=None,
                         help="Fetch one specific player's snapshot (parsed).")
    parser.add_argument("--raw", action="store_true",
                         help="With --player-id, print the UNPARSED raw JSON instead "
                              "of the parsed snapshot, to verify field names.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    event_id = args.event_id
    if event_id is None and args.find_event:
        event_id = find_event_id(args.find_event[0], args.find_event[1],
                                  tournament_name=args.tournament_name, debug=args.debug)
        print(f"event_id = {event_id}")

    if event_id and not args.player_id:
        field = get_field(event_id, debug=args.debug)
        print(f"{len(field)} players in field. First 10:")
        for p in field[:10]:
            print(f"  {p['espn_player_id']}: {p['player_name']}")

    if event_id and args.player_id:
        if args.raw:
            payload = _get_json(
                PLAYER_SUMMARY_URL.format(event_id=event_id),
                params={"season": args.year, "player": args.player_id},
                debug=args.debug,
            )
            print(json.dumps(payload, indent=2))
        else:
            snap = get_player_cumulative_snapshot(
                event_id, args.year, args.player_id, debug=args.debug)
            print(json.dumps(snap, indent=2))
