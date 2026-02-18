#!/usr/bin/env python3
import argparse
from datetime import date as d_date, datetime, timedelta
from typing import Dict, Any, List, Optional

# Uses your existing ESPN client wrapper
from pickslab_elite.core.http.espn_client import get_espn_client


# ----------------------------
# Live ESPN NBA injury loader
# ----------------------------

def load_live_nba_injuries() -> Dict[str, Dict[str, Any]]:
    """
    ESPN endpoint returns containers per team (displayName: team name),
    each containing an "injuries" list of player injuries.
    We flatten to a pid -> {status, detail, date} map.
    """
    c = get_espn_client()
    j = c.get("sports/basketball/nba/injuries")

    injury_map: Dict[str, Dict[str, Any]] = {}

    for container in (j.get("injuries") or []):
        for inj in (container.get("injuries") or []):
            athlete = inj.get("athlete") or {}
            pid = str(athlete.get("id") or "")
            if not pid:
                continue

            injury_map[pid] = {
                "status": inj.get("status") or "UNKNOWN",
                "detail": inj.get("details") or inj.get("shortComment") or inj.get("longComment"),
                "date": inj.get("date"),
                "source_team": container.get("displayName"),
            }

    print(f"[injuries] loaded {len(injury_map)} player injuries")
    return injury_map


# ----------------------------
# ESPN slate helpers
# ----------------------------

def yyyymmdd_from_ymd(ymd: str) -> str:
    # input: YYYY-MM-DD
    dt = datetime.strptime(ymd, "%Y-%m-%d")
    return dt.strftime("%Y%m%d")

def find_next_active_slate(start_ymd: str, lookahead_days: int = 14) -> str:
    """
    If the requested date has 0 events (common in NBA All-Star break),
    scan forward until an active day is found.
    """
    c = get_espn_client()
    dt = datetime.strptime(start_ymd, "%Y-%m-%d")

    for _ in range(lookahead_days + 1):
        ds = dt.strftime("%Y%m%d")
        try:
            j = c.get(f"sports/basketball/nba/scoreboard?dates={ds}")
            if j.get("events"):
                if ds != yyyymmdd_from_ymd(start_ymd):
                    print(f"[schedule] no games on {start_ymd}; using next active slate {dt.strftime('%Y-%m-%d')} (ds={ds})")
                return dt.strftime("%Y-%m-%d")
        except Exception as e:
            # If ESPN hiccups, keep scanning forward a bit
            print(f"[warn] scoreboard fetch failed for {ds}: {e}")

        dt += timedelta(days=1)

    return start_ymd

def fetch_scoreboard_events(ymd: str) -> List[Dict[str, Any]]:
    c = get_espn_client()
    ds = yyyymmdd_from_ymd(ymd)
    j = c.get(f"sports/basketball/nba/scoreboard?dates={ds}")
    return j.get("events") or []

def extract_games(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert ESPN 'events' to a minimal list of games:
    [{game_id, start_ts, home_team_id, away_team_id, home, away}, ...]
    """
    games: List[Dict[str, Any]] = []

    for ev in events:
        comp = (ev.get("competitions") or [{}])[0]
        gid = str(comp.get("id") or ev.get("id") or "")
        if not gid:
            continue

        # date string like "2026-02-20T02:30Z"
        iso = comp.get("date") or ev.get("date")
        start_ts = None
        if iso:
            try:
                # ESPN uses Z; datetime.fromisoformat doesn't like Z in some py versions
                iso2 = iso.replace("Z", "+00:00")
                start_ts = int(datetime.fromisoformat(iso2).timestamp())
            except Exception:
                start_ts = None

        competitors = comp.get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)

        def team_id(side):
            return str(((side or {}).get("team") or {}).get("id") or "")

        games.append({
            "game_id": gid,
            "start_ts": start_ts,
            "home_team_id": team_id(home),
            "away_team_id": team_id(away),
            "home": ((home or {}).get("team") or {}).get("displayName"),
            "away": ((away or {}).get("team") or {}).get("displayName"),
        })

    return games


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate NBA slate + live injury map (ESPN).")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--lookahead", type=int, default=14, help="Days to scan forward for next slate if 0 games")
    ap.add_argument("--show", type=int, default=10, help="How many games to print")
    ap.add_argument("--show-inj", type=int, default=5, help="How many injury entries to print")
    args = ap.parse_args()

    # 1) injuries
    injury_map = load_live_nba_injuries()

    # 2) find active slate
    active_date = find_next_active_slate(args.date, lookahead_days=args.lookahead)

    # 3) fetch slate
    try:
        events = fetch_scoreboard_events(active_date)
    except Exception as e:
        print(f"[error] could not fetch scoreboard for {active_date}: {e}")
        return 2

    print(f"\n=== NBA SLATE {active_date} ===")
    print("events:", len(events))

    games = extract_games(events)
    print("games:", len(games))

    if not games:
        print("No games found. (Likely schedule gap / All-Star break / off-day.)")
        return 0

    # Print games
    for g in games[: args.show]:
        print(f"- {g['away']} @ {g['home']}   (gid={g['game_id']} home_id={g['home_team_id']} away_id={g['away_team_id']})")

    # Print injury samples
    if injury_map:
        print(f"\n=== Injury samples (pid -> status/detail/date) ===")
        n = 0
        for pid, info in injury_map.items():
            print(f"- {pid}: {info.get('status')} | {info.get('detail')} | {info.get('date')} | team={info.get('source_team')}")
            n += 1
            if n >= args.show_inj:
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
