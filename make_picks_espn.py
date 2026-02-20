from datetime import datetime, timezone
from espn_client import ESPNClient

LEAGUE = "nba"  # nfl/nba/mlb/nhl
DATES  = datetime.now(timezone.utc).strftime("%Y%m%d")

c = ESPNClient(LEAGUE)
sb = c.scoreboard(dates=DATES)
games = c.extract_matchups(sb)

print(f"LEAGUE={LEAGUE} DATES={DATES} GAMES={len(games)}")

for g in games:
    event_id = g["event_id"]
    home = g["home"]
    away = g["away"]
    print("\n---", away, "@", home, "event", event_id, "---")

    try:
        sm = c.summary(event_id)
        box = sm.get("boxscore", {})
        teams = box.get("teams", []) or []

        # defensive parsing: not all leagues expose same stat names
        for t in teams:
            abbr = (t.get("team") or {}).get("abbreviation")
            stats = {}
            for s in (t.get("statistics") or []):
                nm = s.get("name")
                if nm:
                    stats[nm] = s.get("value", s.get("displayValue"))
            if abbr:
                print("STATS", abbr, stats)
    except Exception as e:
        print("summary error:", e)
