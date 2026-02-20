import json, sys, math

def american_to_prob(a: int) -> float:
    a = int(a)
    if a > 0:
        return 100.0 / (a + 100.0)
    return (-a) / ((-a) + 100.0)

def norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def load_last_snapshot(path: str) -> dict:
    last = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if not last:
        raise SystemExit(f"[err] no snapshot lines in {path}")
    return json.loads(last)

def build_index(snap: dict) -> dict:
    # index[(home, away, market_key, outcome_name, point_or_none)] = best_price
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
                    key = (home, away, mk, name, point)
                    if price is None:
                        continue
                    # keep BEST price for bettor:
                    # for underdog (+) higher is better; for favorite (-) closer to 0 is better (i.e. larger number)
                    prev = idx.get(key)
                    if prev is None:
                        idx[key] = price
                    else:
                        # compare as "better for bettor"
                        if price > 0 and prev > 0:
                            if price > prev: idx[key] = price
                        elif price < 0 and prev < 0:
                            if price > prev: idx[key] = price  # -120 better than -150
                        else:
                            # mixed signs shouldn't happen same outcome, but just keep max by implied prob
                            if american_to_prob(price) < american_to_prob(prev):
                                idx[key] = price
    return idx

def extract_picks(pobj: dict):
    # supports either {"picks":[...]} or list
    if isinstance(pobj, list):
        return pobj
    if isinstance(pobj, dict):
        for k in ("picks","results","entries"):
            if k in pobj and isinstance(pobj[k], list):
                return pobj[k]
    raise SystemExit("[err] couldn't find picks list in picks_output.json")

def pick_fields(p: dict):
    # try common key names
    sport = p.get("sport") or p.get("sport_key") or p.get("league")
    market = p.get("market") or p.get("market_key") or p.get("market_type")
    selection = p.get("selection") or p.get("outcome") or p.get("team") or p.get("name") or p.get("pick")
    home = p.get("home_team") or p.get("home")
    away = p.get("away_team") or p.get("away")
    # sometimes "game" string contains teams; fallback handled later
    game = p.get("game") or p.get("matchup") or p.get("event")

    # odds fields
    bet_odds = (
        p.get("best_odds")
        or p.get("odds")
        or p.get("price")
        or (p.get("best") or {}).get("odds")
        or (p.get("best") or {}).get("price")
    )
    bet_book = (
        p.get("best_book")
        or p.get("book")
        or (p.get("best") or {}).get("book")
        or (p.get("best") or {}).get("sportsbook")
    )
    point = p.get("point", None)
    line = p.get("line", None)
    if point is None and line is not None:
        point = line

    return sport, market, selection, home, away, game, bet_odds, bet_book, point

def parse_game_string(game: str):
    g = (game or "")
    # expected: "Away @ Home" or "Away at Home"
    g = g.replace(" at ", " @ ").replace(" vs ", " @ ")
    if "@" in g:
        left, right = g.split("@", 1)
        away = norm(left)
        home = norm(right)
        return home, away
    return None, None

def main():
    picks_path = sys.argv[1] if len(sys.argv) > 1 else "picks_output.json"
    snap_path  = sys.argv[2] if len(sys.argv) > 2 else "data/odds_snapshots.jsonl"

    picks_obj = json.load(open(picks_path, "r", encoding="utf-8"))
    picks = extract_picks(picks_obj)
    snap = load_last_snapshot(snap_path)
    idx = build_index(snap)
    ts = snap.get("ts_utc")

    print(f"=== CLV CHECK (vs latest snapshot) ===")
    print(f"snapshot_ts_utc: {ts}")
    print()

    ok = 0
    miss = 0

    for i, p in enumerate(picks, 1):
        sport, market, selection, home, away, game, bet_odds, bet_book, point = pick_fields(p)
        if not (market and selection):
            miss += 1
            print(f"{i:2d}. [skip] missing market/selection keys")
            continue

        home_n = norm(home)
        away_n = norm(away)
        if (not home_n or not away_n) and game:
            home_n, away_n = parse_game_string(game)

        if not (home_n and away_n):
            miss += 1
            print(f"{i:2d}. [skip] can't determine teams (home/away)")
            continue

        mk = norm(market)
        sel = norm(selection)

        key = (home_n, away_n, mk, sel, point if mk != "h2h" else None)
        snap_price = idx.get(key)

        if snap_price is None:
            # Try flipped teams (some code stores as away/home)
            key2 = (away_n, home_n, mk, sel, point if mk != "h2h" else None)
            snap_price = idx.get(key2)

        if snap_price is None:
            miss += 1
            pretty = f"{away_n} @ {home_n}"
            print(f"{i:2d}. [MISS] {pretty} | {mk} | {selection} | point={point} | bet={bet_odds} | (no match in snapshot)")
            continue

        try:
            bet_odds_i = int(bet_odds)
        except Exception:
            miss += 1
            print(f"{i:2d}. [skip] bet odds not int: {bet_odds}")
            continue

        # CLV in implied prob (lower implied prob = better price for bettor)
        p_bet = american_to_prob(bet_odds_i)
        p_snap = american_to_prob(int(snap_price))
        clv = (p_snap - p_bet)  # positive means market moved toward your side (you beat the price)

        ok += 1
        pretty = f"{away_n} @ {home_n}"
        print(f"{i:2d}. {pretty} | {mk} | {selection} | bet {bet_odds_i:+d} vs snap {int(snap_price):+d} | CLV={clv*100:+.2f} pp")

    print()
    print(f"[done] matched={ok} missed={miss} total={len(picks)}")
    print("note: true CLV needs multiple snapshots and a defined 'close' time.")
