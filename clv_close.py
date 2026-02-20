import json, sys
from datetime import datetime, timezone

def american_to_prob(a: int) -> float:
    a = int(a)
    if a > 0:
        return 100.0 / (a + 100.0)
    return (-a) / ((-a) + 100.0)

def norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def dt_parse(s: str) -> datetime:
    # handles "2026-02-20T04:30:06.570335+00:00" and "2026-02-20T03:10:00Z"
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)

def extract_picks(pobj):
    if isinstance(pobj, list):
        return pobj
    if isinstance(pobj, dict):
        for k in ("picks","results","entries"):
            if k in pobj and isinstance(pobj[k], list):
                return pobj[k]
    raise SystemExit("[err] couldn't find picks list in picks_output.json")

def pick_fields(p: dict):
    market = p.get("market") or p.get("market_key") or p.get("market_type")
    selection = p.get("selection") or p.get("outcome") or p.get("team") or p.get("name") or p.get("pick")
    home = p.get("home_team") or p.get("home")
    away = p.get("away_team") or p.get("away")
    game = p.get("game") or p.get("matchup") or p.get("event")
    commence = p.get("commence_time") or p.get("start_time") or p.get("start_utc")

    bet_odds = (
        p.get("best_odds")
        or p.get("odds")
        or p.get("price")
        or (p.get("best") or {}).get("odds")
        or (p.get("best") or {}).get("price")
    )
    point = p.get("point", None)
    line = p.get("line", None)
    if point is None and line is not None:
        point = line

    return market, selection, home, away, game, commence, bet_odds, point

def parse_game_string(game: str):
    g = (game or "")
    g = g.replace(" at ", " @ ").replace(" vs ", " @ ")
    if "@" in g:
        left, right = g.split("@", 1)
        away = norm(left)
        home = norm(right)
        return home, away
    return None, None

def build_index(snap: dict) -> dict:
    idx = {}
    for g in snap.get("games", []):
        home = norm(g.get("home_team"))
        away = norm(g.get("away_team"))
        for bm in g.get("bookmakers", []):
            for m in bm.get("markets", []):
                mk = norm(m.get("key"))
                for oc in m.get("outcomes", []):
                    name = norm(oc.get("name"))
                    price = oc.get("price")
                    point = oc.get("point", None)
                    if price is None:
                        continue
                    key = (home, away, mk, name, point if mk != "h2h" else None)
                    prev = idx.get(key)
                    if prev is None:
                        idx[key] = price
                    else:
                        if price > 0 and prev > 0:
                            if price > prev: idx[key] = price
                        elif price < 0 and prev < 0:
                            if price > prev: idx[key] = price
                        else:
                            if american_to_prob(price) < american_to_prob(prev):
                                idx[key] = price
    return idx

def load_snapshots(path: str):
    snaps = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            ts = dt_parse(s["ts_utc"])
            snaps.append((ts, s))
    snaps.sort(key=lambda x: x[0])
    return snaps

def latest_snapshot_before(snaps, t: datetime):
    best = None
    for ts, s in snaps:
        if ts <= t:
            best = (ts, s)
        else:
            break
    return best

def main():
    picks_path = sys.argv[1] if len(sys.argv) > 1 else "picks_output.json"
    snap_path  = sys.argv[2] if len(sys.argv) > 2 else "data/odds_snapshots.jsonl"

    picks_obj = json.load(open(picks_path, "r", encoding="utf-8"))
    picks = extract_picks(picks_obj)
    snaps = load_snapshots(snap_path)

    print("=== CLV CHECK (vs last snapshot BEFORE game start) ===")
    print(f"snapshots: {len(snaps)} lines")
    print()

    ok = 0
    miss = 0

    for i, p in enumerate(picks, 1):
        market, selection, home, away, game, commence, bet_odds, point = pick_fields(p)
        if not (market and selection and bet_odds and commence):
            miss += 1
            print(f"{i:2d}. [skip] missing market/selection/bet_odds/commence_time")
            continue

        home_n = norm(home)
        away_n = norm(away)
        if (not home_n or not away_n) and game:
            home_n, away_n = parse_game_string(game)
        if not (home_n and away_n):
            miss += 1
            print(f"{i:2d}. [skip] can't determine teams")
            continue

        try:
            start_t = dt_parse(commence)
            bet_odds_i = int(bet_odds)
        except Exception:
            miss += 1
            print(f"{i:2d}. [skip] bad commence_time or odds:", commence, bet_odds)
            continue

        snap_pair = latest_snapshot_before(snaps, start_t)
        if not snap_pair:
            miss += 1
            print(f"{i:2d}. [MISS] no snapshot before start:", start_t.isoformat())
            continue

        snap_ts, snap = snap_pair
        idx = build_index(snap)

        mk = norm(market)
        sel = norm(selection)
        key = (home_n, away_n, mk, sel, point if mk != "h2h" else None)
        snap_price = idx.get(key)
        if snap_price is None:
            key2 = (away_n, home_n, mk, sel, point if mk != "h2h" else None)
            snap_price = idx.get(key2)

        if snap_price is None:
            miss += 1
            pretty = f"{away_n} @ {home_n}"
            print(f"{i:2d}. [MISS] {pretty} | {mk} | {selection} | close_snap={snap_ts.isoformat()} | no match")
            continue

        p_bet = american_to_prob(bet_odds_i)
        p_close = american_to_prob(int(snap_price))
        clv = (p_close - p_bet)

        ok += 1
        pretty = f"{away_n} @ {home_n}"
        print(f"{i:2d}. {pretty} | {mk} | {selection} | bet {bet_odds_i:+d} vs close {int(snap_price):+d} | CLV={clv*100:+.2f} pp | snap={snap_ts.isoformat()}")

    print()
    print(f"[done] matched={ok} missed={miss} total={len(picks)}")
