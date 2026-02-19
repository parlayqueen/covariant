from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).resolve().parent / ".env")

from env_loader import load_env
load_env()

#!/usr/bin/env python3
from __future__ import annotations
from sportsdata_client import get

import os
from dotenv import load_dotenv
load_dotenv(), json, glob, time, argparse, math, csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

RUNS_DIR_DEFAULT = Path("runs")

# ----------------------------
# Odds helpers
# ----------------------------
def american_to_decimal(odds: int) -> float:
    odds = int(odds)
    if odds == 0:
        return 1.0
    if odds > 0:
        return 1.0 + (odds / 100.0)
    return 1.0 + (100.0 / abs(odds))

def implied_prob_from_american(odds: int) -> float:
    odds = int(odds)
    if odds == 0:
        return 0.0
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)

def pick_profit(stake: float, odds: int, result: str) -> float:
    stake = float(stake or 0.0)
    if stake <= 0:
        return 0.0
    if result == "win":
        dec = american_to_decimal(int(odds))
        return stake * (dec - 1.0)
    if result == "loss":
        return -stake
    return 0.0

# ----------------------------
# Key masking / string utils
# ----------------------------
def mask_key(k: str) -> str:
    k = (k or "").strip()
    if len(k) <= 8:
        return "*" * len(k)
    return k[:4] + "…" + k[-4:]

def s(x: Any) -> str:
    return "" if x is None else str(x)

def norm_team(x: str) -> str:
    x = (x or "").strip().lower()
    x = x.replace(".", "").replace(",", "").replace("&", "and")
    x = " ".join(x.split())
    return x

def norm_iso_time(x: str) -> str:
    # normalize "2026-02-20T00:00:00Z" or "...+00:00" -> a stable comparable string
    x = s(x).strip()
    if not x:
        return ""
    return x.replace("+00:00", "Z")

# ----------------------------
# Ledger discovery / loading
# ----------------------------
def latest_picks_file(runs_dir: Path) -> str:
    cands = sorted(glob.glob(str(runs_dir / "picks_full_*.json")))
    if not cands:
        cands = sorted(glob.glob(str(runs_dir / "picks_*.json")))
    if not cands:
        raise SystemExit(f"No {runs_dir}/picks_full_*.json (or picks_*.json) found.")
    return cands[-1]

def load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())

def extract_picks(ledger: Dict[str, Any]) -> List[Dict[str, Any]]:
    picks = ledger.get("picks") or []
    if not isinstance(picks, list):
        raise SystemExit("Unexpected ledger: ledger['picks'] is not a list.")
    return picks

# ----------------------------
# HTTP with retry + 429 handling + quota visibility
# ----------------------------
def http_get_with_retry(url: str, params: Dict[str, Any], timeout: int = 30, tries: int = 6) -> requests.Response:
    last_exc: Optional[Exception] = None
    backoff = 1.0

    for attempt in range(1, tries + 1):
        try:
            r = get(url, params=params, timeout=timeout)

            # quota headers (if present)
            rem = r.headers.get("x-requests-remaining") or r.headers.get("X-Requests-Remaining")
            used = r.headers.get("x-requests-used") or r.headers.get("X-Requests-Used")
            if rem is not None or used is not None:
                print(f"[quota] remaining={rem} used={used}")

            if r.status_code == 401:
                raise SystemExit(
                    "401 Unauthorized from The Odds API.\n"
                    "Most common causes: wrong key, inactive key, or your plan doesn’t include this endpoint.\n"
                    f"Key used (masked): {mask_key(str(params.get('apiKey','')))}\n"
                    f"URL: {r.url}\n"
                )

            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", "2") or 2.0)
                wait = max(1.0, min(15.0, wait))
                print(f"[rate-limit] 429 -> sleeping {wait}s")
                time.sleep(wait)
                continue

            r.raise_for_status()
            return r

        except SystemExit:
            raise
        except Exception as e:
            last_exc = e
            time.sleep(min(12.0, backoff))
            backoff *= 1.7

    raise SystemExit(f"HTTP failed after retries. Last error: {last_exc}")

def chunked(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i:i+n] for i in range(0, len(xs), n)]

# ----------------------------
# Scores API fetch (multi-pass daysFrom 1..N)
# ----------------------------
def fetch_scores(api_key: str, sport: str, max_days_from: int, event_ids: List[str]) -> Dict[str, Any]:
    if not api_key:
        raise SystemExit("Missing ODDS_API_KEY. Fix: export ODDS_API_KEY='...'\n")

    # the endpoint typically supports 1..3; we clamp for safety
    max_days_from = max(1, min(3, int(max_days_from)))

    base = f"https://api.the-odds-api.com/v4/sports/{sport}/scores/"
    out: Dict[str, Any] = {}

    # multi-pass: 1..max_days_from (helps when games shift across day boundaries)
    for days_from in range(1, max_days_from + 1):
        print(f"[scores] fetching daysFrom={days_from} for {len(event_ids)} event_ids")
        for batch in chunked(event_ids, 50):
            params = {
                "apiKey": api_key,
                "daysFrom": days_from,
                "dateFormat": "iso",
                "eventIds": ",".join(batch),
            }
            r = http_get_with_retry(base, params=params, timeout=30, tries=6)
            arr = r.json()
            if not isinstance(arr, list):
                raise SystemExit(f"Scores API returned non-list payload: {type(arr)} -> {str(arr)[:400]}")

            for g in arr:
                gid = s(g.get("id")).strip()
                if gid:
                    # keep the "most complete" version if duplicates
                    prev = out.get(gid)
                    if prev is None:
                        out[gid] = g
                    else:
                        # prefer completed=True, or one with scores present
                        prev_comp = bool(prev.get("completed"))
                        new_comp = bool(g.get("completed"))
                        prev_scores = bool(prev.get("scores"))
                        new_scores = bool(g.get("scores"))
                        if (new_comp and not prev_comp) or (new_scores and not prev_scores):
                            out[gid] = g

    return out

def extract_scores(game: Dict[str, Any]) -> Optional[Dict[str, int]]:
    scores = game.get("scores") or []
    if not scores or len(scores) < 2:
        return None
    m: Dict[str, int] = {}
    for row in scores:
        name = s(row.get("name")).strip()
        sc = row.get("score")
        if name and sc is not None:
            try:
                m[name] = int(sc)
            except Exception:
                pass
    return m if len(m) >= 2 else None

def resolve_team_key(scoremap: Dict[str, int], team: str) -> Optional[str]:
    if team in scoremap:
        return team
    nt = norm_team(team)
    for k in scoremap.keys():
        if norm_team(k) == nt:
            return k
    return None

# ----------------------------
# Fallback matching (if event_id mismatches)
# ----------------------------
def build_game_fallback_index(scores_by_id: Dict[str, Any]) -> Dict[Tuple[str, str, str], str]:
    idx: Dict[Tuple[str, str, str], str] = {}
    for gid, g in scores_by_id.items():
        home = norm_team(s(g.get("home_team")))
        away = norm_team(s(g.get("away_team")))
        ct = norm_iso_time(s(g.get("commence_time")))
        if home and away and ct:
            idx[(home, away, ct)] = gid
    return idx

def find_game_for_pick(pick: Dict[str, Any], scores_by_id: Dict[str, Any], fallback_idx: Dict[Tuple[str,str,str], str]) -> Optional[Dict[str, Any]]:
    gid = s(pick.get("event_id")).strip()
    if gid and gid in scores_by_id:
        return scores_by_id[gid]

    # fallback: match by home/away + commence_time if present
    # (your picks_full ledger doesn’t store home/away directly, but game objects might be in some ledgers)
    # We attempt to infer from pick["matchup"] like "Away @ Home"
    matchup = s(pick.get("matchup"))
    commence = norm_iso_time(s(pick.get("commence_time")))  # if you add it later
    away = home = ""
    if "@" in matchup:
        parts = [p.strip() for p in matchup.split("@", 1)]
        if len(parts) == 2:
            away, home = parts[0], parts[1]
    key = (norm_team(home), norm_team(away), commence) if commence else None
    if key and key in fallback_idx:
        gid2 = fallback_idx[key]
        return scores_by_id.get(gid2)

    return None

# ----------------------------
# Graders
# ----------------------------
def grade_moneyline(selection: str, scoremap: Dict[str, int], home: str, away: str) -> str:
    hk = resolve_team_key(scoremap, home)
    ak = resolve_team_key(scoremap, away)
    if not hk or not ak:
        return "ungraded"
    hs = scoremap.get(hk)
    a_s = scoremap.get(ak)
    if hs is None or a_s is None:
        return "ungraded"
    if hs == a_s:
        return "push"
    winner = home if hs > a_s else away
    return "win" if norm_team(selection) == norm_team(winner) else "loss"

def grade_spread(selection: str, line: Any, scoremap: Dict[str, int], home: str, away: str) -> Tuple[str, Dict[str, Any]]:
    dbg: Dict[str, Any] = {}
    try:
        linef = float(line)
    except Exception:
        return "ungraded", {"reason": "bad_line"}

    hk = resolve_team_key(scoremap, home)
    ak = resolve_team_key(scoremap, away)
    if not hk or not ak:
        return "ungraded", {"reason": "team_mismatch"}

    hs = scoremap.get(hk)
    a_s = scoremap.get(ak)
    if hs is None or a_s is None:
        return "ungraded", {"reason": "missing_score"}

    if norm_team(selection) == norm_team(home):
        adj = hs + linef
        opp = a_s
        side = "home"
    elif norm_team(selection) == norm_team(away):
        adj = a_s + linef
        opp = hs
        side = "away"
    else:
        return "ungraded", {"reason": "selection_not_home_or_away"}

    dbg.update({"hs": hs, "as": a_s, "line": linef, "side": side, "adj": adj, "opp": opp})
    if math.isclose(adj, opp, abs_tol=1e-9):
        return "push", dbg
    return ("win" if adj > opp else "loss"), dbg

def grade_total(selection: str, line: Any, scoremap: Dict[str, int], home: str, away: str) -> Tuple[str, Dict[str, Any]]:
    dbg: Dict[str, Any] = {}
    try:
        linef = float(line)
    except Exception:
        return "ungraded", {"reason": "bad_line"}

    hk = resolve_team_key(scoremap, home)
    ak = resolve_team_key(scoremap, away)
    if not hk or not ak:
        return "ungraded", {"reason": "team_mismatch"}

    hs = scoremap.get(hk)
    a_s = scoremap.get(ak)
    if hs is None or a_s is None:
        return "ungraded", {"reason": "missing_score"}

    total = hs + a_s
    dbg.update({"hs": hs, "as": a_s, "line": linef, "total": total, "selection": selection})

    if math.isclose(total, linef, abs_tol=1e-9):
        return "push", dbg

    sel = norm_team(selection)
    if sel == "over":
        return ("win" if total > linef else "loss"), dbg
    if sel == "under":
        return ("win" if total < linef else "loss"), dbg
    return "ungraded", {"reason": "selection_not_over_under", **dbg}

def grade_pick(pick: Dict[str, Any], game: Dict[str, Any]) -> Tuple[str, Optional[int], Optional[int], Dict[str, Any]]:
    home = s(game.get("home_team"))
    away = s(game.get("away_team"))

    sm = extract_scores(game)
    if not sm:
        return "ungraded", None, None, {"reason": "no_scores_yet"}

    hk = resolve_team_key(sm, home) or home
    ak = resolve_team_key(sm, away) or away
    hs = sm.get(hk)
    a_s = sm.get(ak)

    market = s(pick.get("market")).strip().lower()
    selection = s(pick.get("selection")).strip()
    line = pick.get("line", None)

    if market == "moneyline":
        res = grade_moneyline(selection, sm, home, away)
        return res, hs, a_s, {"market": "moneyline", "home": home, "away": away, "selection": selection}

    if market == "spread":
        res, dbg = grade_spread(selection, line, sm, home, away)
        return res, hs, a_s, {"market": "spread", "home": home, "away": away, "selection": selection, **dbg}

    if market == "total":
        res, dbg = grade_total(selection, line, sm, home, away)
        return res, hs, a_s, {"market": "total", "home": home, "away": away, **dbg}

    return "ungraded", hs, a_s, {"reason": "unknown_market", "market": market}

# ----------------------------
# Summary + CSV
# ----------------------------
def pct(x: float) -> float:
    return round(100.0 * x, 2)

def summarize(picks: List[Dict[str, Any]]) -> Dict[str, Any]:
    win = loss = push = ungraded = 0
    pnl = 0.0
    staked = 0.0
    by_market: Dict[str, Dict[str, Any]] = {}

    for p in picks:
        res = s(p.get("result")) or "ungraded"
        market = s(p.get("market") or "unknown").lower()
        stake = float(p.get("stake") or 0.0)
        profit = float(p.get("profit") or 0.0)

        by_market.setdefault(market, {"win": 0, "loss": 0, "push": 0, "ungraded": 0, "staked": 0.0, "pnl": 0.0})

        if res not in ("win", "loss", "push", "ungraded"):
            res = "ungraded"

        by_market[market][res] += 1

        if res != "ungraded":
            by_market[market]["staked"] += stake
            staked += stake

        by_market[market]["pnl"] += profit
        pnl += profit

        if res == "win": win += 1
        elif res == "loss": loss += 1
        elif res == "push": push += 1
        else: ungraded += 1

    roi = (pnl / staked) if staked > 0 else 0.0
    winrate = (win / (win + loss)) if (win + loss) > 0 else 0.0

    for m, d in by_market.items():
        s2 = float(d.get("staked") or 0.0)
        d["roi"] = (float(d.get("pnl") or 0.0) / s2) if s2 > 0 else 0.0

    return {
        "win": win, "loss": loss, "push": push, "ungraded": ungraded,
        "staked": round(staked, 6),
        "net_profit": round(pnl, 6),
        "roi": round(roi, 8),
        "winrate": round(winrate, 8),
        "by_market": by_market,
        "total_picks": len(picks),
    }

def write_csv(path: Path, picks: List[Dict[str, Any]]) -> None:
    cols = [
        "event_id", "market", "selection", "line", "odds", "stake",
        "result", "profit", "implied_prob",
        "home_team", "away_team", "commence_time",
        "final_home", "final_away",
        "book"
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for p in picks:
            game = p.get("game") or {}
            fs = p.get("final_score") or {}
            ht = s(game.get("home_team"))
            at = s(game.get("away_team"))
            row = {
                "event_id": p.get("event_id"),
                "market": p.get("market"),
                "selection": p.get("selection"),
                "line": p.get("line"),
                "odds": p.get("odds"),
                "stake": p.get("stake"),
                "result": p.get("result"),
                "profit": p.get("profit"),
                "implied_prob": p.get("implied_prob"),
                "home_team": ht,
                "away_team": at,
                "commence_time": game.get("commence_time"),
                "final_home": fs.get(ht) if ht in fs else None,
                "final_away": fs.get(at) if at in fs else None,
                "book": p.get("book"),
            }
            w.writerow(row)

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Grade picks ledgers against The Odds API scores endpoint.")
    ap.add_argument("--ledger", default="", help="Ledger path (default: latest runs/picks_full_*.json).")
    ap.add_argument("--sport", default=os.environ.get("SPORT", "basketball_nba").strip(), help="Sport key (e.g., basketball_nba).")
    ap.add_argument("--days-from", type=int, default=int(os.environ.get("DAYS_FROM", "3")), help="Scores lookback (clamped 1..3).")
    ap.add_argument("--runs-dir", default=str(RUNS_DIR_DEFAULT), help="Runs dir (default: runs).")
    ap.add_argument("--out", default="", help="Output graded json path (default: runs/graded_<ledger>_<ts>.json).")
    ap.add_argument("--csv", default="", help="CSV output path. Use 'auto' to save runs/graded_<ledger>_<ts>.csv")
    ap.add_argument("--explain", action="store_true", help="Print a grading trace for each pick.")
    ap.add_argument("--max", type=int, default=0, help="Only grade first N picks (debug). 0 = all.")
    ap.add_argument("--dry-run", action="store_true", help="Don’t call API; validate ledger + show event_id count.")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("ODDS_API_KEY", "").strip()
    if not args.dry_run and not api_key:
        raise SystemExit("Missing ODDS_API_KEY. Fix: export ODDS_API_KEY='...'\n")

    ledger_path = args.ledger.strip() or latest_picks_file(runs_dir)
    ledger = load_json(ledger_path)
    picks = extract_picks(ledger)

    if args.max and args.max > 0:
        picks = picks[:args.max]

    event_ids = sorted({s(p.get("event_id")).strip() for p in picks if s(p.get("event_id")).strip()})
    if not event_ids:
        raise SystemExit("No event_id found in picks — cannot grade against scores endpoint.")

    if args.dry_run:
        print(json.dumps({
            "ok": True,
            "ledger": ledger_path,
            "sport": args.sport,
            "daysFrom": max(1, min(3, int(args.days_from))),
            "picks": len(picks),
            "unique_event_ids": len(event_ids),
            "api_key_present": bool(api_key),
            "api_key_masked": mask_key(api_key) if api_key else "",
        }, indent=2))
        return

    scores_by_id = fetch_scores(api_key=api_key, sport=args.sport, max_days_from=args.days_from, event_ids=event_ids)
    fallback_idx = build_game_fallback_index(scores_by_id)

    graded: List[Dict[str, Any]] = []
    misses = 0

    for i, p in enumerate(picks, start=1):
        game = find_game_for_pick(p, scores_by_id, fallback_idx)
        gid = s(p.get("event_id")).strip()

        if not game:
            misses += 1
            res, hs, a_s, dbg = "ungraded", None, None, {"reason": "no_game_returned_for_event_id", "event_id": gid}
        else:
            res, hs, a_s, dbg = grade_pick(p, game)

        stake = float(p.get("stake") or 0.0)
        odds = int(p.get("odds") or 0)
        prof = pick_profit(stake, odds, res)

        p2 = dict(p)
        p2["result"] = res
        p2["profit"] = round(prof, 6)
        p2["implied_prob"] = round(implied_prob_from_american(odds), 6) if odds else 0.0

        if game:
            p2["game"] = {
                "id": s(game.get("id") or gid),
                "commence_time": game.get("commence_time"),
                "completed": bool(game.get("completed")),
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
            }

        if hs is not None and a_s is not None and game:
            ht = s(game.get("home_team"))
            at = s(game.get("away_team"))
            p2["final_score"] = {ht: int(hs), at: int(a_s)}

        if args.explain:
            print(f"\n[{i}] event_id={gid} market={p.get('market')} sel={p.get('selection')} line={p.get('line')} odds={p.get('odds')} stake={p.get('stake')}")
            if game:
                print(f"    game: {game.get('away_team')} @ {game.get('home_team')} completed={game.get('completed')} commence={game.get('commence_time')}")
            print(f"    grade: {res} profit={round(prof,4)} debug={dbg}")

        graded.append(p2)

    summary = summarize(graded)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    ledger_stem = Path(ledger_path).stem
    out_path = Path(args.out) if args.out else (runs_dir / f"graded_{ledger_stem}_{stamp}.json")

    out_obj = {
        "graded_at": int(time.time()),
        "source_ledger": ledger_path,
        "sport": args.sport,
        "daysFrom": max(1, min(3, int(args.days_from))),
        "summary": summary,
        "api": {
            "key_masked": mask_key(api_key),
            "scores_returned": len(scores_by_id),
            "event_id_misses": misses,
        },
        "picks": graded,
    }
    out_path.write_text(json.dumps(out_obj, indent=2))

    csv_path: Optional[Path] = None
    if args.csv:
        if args.csv.strip().lower() == "auto":
            csv_path = runs_dir / f"graded_{ledger_stem}_{stamp}.csv"
        else:
            csv_path = Path(args.csv)
        write_csv(csv_path, graded)

    print(f"(saved) {out_path}")
    if csv_path:
        print(f"(saved) {csv_path}")

    print(json.dumps({
        "win": summary["win"],
        "loss": summary["loss"],
        "push": summary["push"],
        "ungraded": summary["ungraded"],
        "staked": summary["staked"],
        "net_profit": summary["net_profit"],
        "roi_pct": pct(summary["roi"]),
        "winrate_pct": pct(summary["winrate"]),
        "total_picks": summary["total_picks"],
        "scores_returned": len(scores_by_id),
        "event_id_misses": misses,
    }, indent=2))

    bym = summary.get("by_market") or {}
    if bym:
        print("\nPer-market:")
        for m in sorted(bym.keys()):
            d = bym[m]
            print(
                f"  - {m}: W{d.get('win',0)} L{d.get('loss',0)} P{d.get('push',0)} U{d.get('ungraded',0)} "
                f"staked={round(float(d.get('staked',0.0)),4)} pnl={round(float(d.get('pnl',0.0)),4)} roi%={pct(float(d.get('roi',0.0)))}"
            )

if __name__ == "__main__":
    main()
