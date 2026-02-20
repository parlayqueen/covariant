from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from espn_client import ESPNClient


# ----------------------------- small utilities ----------------------------- #
def _safe_get(d: Any, *path: str, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
    return cur if cur is not None else default


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _norm_key(s: str) -> str:
    return _as_str(s).strip().upper()


def _try_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(str(x).replace("%", "").strip())
    except Exception:
        return None


def _coalesce(*vals):
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return None


# ------------------------- ESPN stat schema parsers ------------------------ #
def _parse_stats_flat(team_blob: Dict[str, Any]) -> Dict[str, Any]:
    """
    Some ESPN payloads expose team stats as:
      team_blob["statistics"] = [ {"name": "...", "value": ..., "displayValue": ...}, ... ]
    """
    out: Dict[str, Any] = {}
    stats = team_blob.get("statistics") or []
    if not isinstance(stats, list):
        return out

    for s in stats:
        if not isinstance(s, dict):
            continue
        name = s.get("name")
        if not name:
            continue
        val = _coalesce(s.get("value"), s.get("displayValue"), s.get("summary"))
        if val is not None:
            out[_as_str(name)] = val
    return out


def _parse_stats_splits(team_blob: Dict[str, Any]) -> Dict[str, Any]:
    """
    NBA often exposes as nested:
      team_blob["statistics"] = [
        {"name": "...", "splits": {"categories":[{"name": "...", "value": ...}, ...]}}
      ]
    """
    out: Dict[str, Any] = {}
    groups = team_blob.get("statistics") or []
    if not isinstance(groups, list):
        return out

    for g in groups:
        if not isinstance(g, dict):
            continue
        splits = g.get("splits") or {}
        cats = splits.get("categories") or []
        if not isinstance(cats, list):
            continue

        for cat in cats:
            if not isinstance(cat, dict):
                continue
            name = cat.get("name")
            if not name:
                continue
            val = _coalesce(cat.get("value"), cat.get("displayValue"), cat.get("summary"))
            if val is not None:
                out[_as_str(name)] = val
    return out


def extract_team_stats_from_summary(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
      {
        "OKC": { ... raw_stats + normalized_metrics ... },
        "BKN": { ... }
      }

    This function tries multiple schema shapes and always returns abbr keys when possible.
    """
    box = summary.get("boxscore") or {}
    teams = box.get("teams") or []
    if not isinstance(teams, list):
        return {}

    out: Dict[str, Dict[str, Any]] = {}

    for t in teams:
        if not isinstance(t, dict):
            continue

        team = t.get("team") or {}
        abbr = _norm_key(team.get("abbreviation") or team.get("shortDisplayName") or team.get("name") or "")
        if not abbr:
            # As last resort, try header competitor mapping later; skip for now
            continue

        # Parse both possible stat layouts, merge them
        flat = _parse_stats_flat(t)
        split = _parse_stats_splits(t)

        merged: Dict[str, Any] = {}
        merged.update(flat)
        merged.update(split)

        # Add identity fields (useful for joins)
        merged["_team_id"] = _as_str(team.get("id") or "")
        merged["_team_name"] = _as_str(team.get("displayName") or team.get("name") or "")
        merged["_team_abbr"] = abbr

        # Some summaries include season aggregates elsewhere. Preserve if present.
        out[abbr] = merged

    return out


# -------------------------- Normalization layer --------------------------- #
def normalize_nba_team_metrics(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    ESPN provides different naming depending on view. We normalize what we can.
    This keeps your model stable even if ESPN shifts labels.
    """
    # Known labels we’ve already seen from your `make_picks_espn.py`
    # Example keys: avgPointsAgainst, avgPoints, fieldGoalPct, threePointFieldGoalPct,
    # avgRebounds, avgAssists, avgBlocks, avgSteals, avgTotalTurnovers, Last Ten Games, streak
    norm: Dict[str, Any] = {}

    # direct passthrough if present
    for k in [
        "avgPoints", "avgPointsAgainst",
        "fieldGoalPct", "threePointFieldGoalPct",
        "avgRebounds", "avgAssists",
        "avgBlocks", "avgSteals",
        "avgTotalTurnovers",
        "Last Ten Games", "streak",
    ]:
        if k in stats:
            norm[k] = stats[k]

    # Try alternate naming patterns (defensive)
    # Sometimes ESPN uses lower/upper or slightly different labels
    alt_map = {
        "avgPoints": ["pointsPerGame", "PPG", "avgPoints"],
        "avgPointsAgainst": ["oppPointsPerGame", "OPPG", "avgPointsAgainst"],
        "fieldGoalPct": ["fieldGoalPct", "fgPct", "FG%"],
        "threePointFieldGoalPct": ["threePointFieldGoalPct", "threePtPct", "3P%"],
        "avgRebounds": ["avgRebounds", "reboundsPerGame", "RPG"],
        "avgAssists": ["avgAssists", "assistsPerGame", "APG"],
        "avgTotalTurnovers": ["avgTotalTurnovers", "turnoversPerGame", "TOV"],
    }

    for k, candidates in alt_map.items():
        if k in norm:
            continue
        for c in candidates:
            if c in stats:
                norm[k] = stats[c]
                break

    # Numeric helpers (optional)
    # Store floats alongside raw strings if convertible
    for k in list(norm.keys()):
        fv = _try_float(norm[k])
        if fv is not None:
            norm[k + "__f"] = fv

    return norm


# ----------------------------- Main entrypoint ---------------------------- #
def get_espn_features(
    league: str = "nba",
    dates: Optional[str] = None,   # YYYYMMDD for ESPN
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Returns a list of games:
      [{
         event_id, date, home, away, home_score, away_score,
         team_stats: { "OKC": {...}, "BKN": {...} },
         team_norm:  { "OKC": {...normalized...}, "BKN": {...} }
      }, ...]
    """
    client = ESPNClient(league)
    sb = client.scoreboard(dates=dates) if dates else client.scoreboard()
    games = client.extract_matchups(sb)

    enriched: List[Dict[str, Any]] = []

    for g in games:
        event_id = g.get("event_id")
        home = _norm_key(g.get("home", ""))
        away = _norm_key(g.get("away", ""))

        if not event_id:
            if debug:
                print("Skipping: missing event_id", g)
            continue

        try:
            summary = client.summary(event_id)
            team_stats = extract_team_stats_from_summary(summary)

            # Some ESPN game summaries may omit boxscore teams until closer to tip-off.
            # Keep it, but warn in debug mode.
            if debug and (home not in team_stats or away not in team_stats):
                print(f"[warn] {away}@{home} event={event_id} missing team_stats keys:",
                      "have=", list(team_stats.keys()))

            # Normalize for NBA-style model features
            team_norm: Dict[str, Dict[str, Any]] = {}
            for abbr, st in team_stats.items():
                team_norm[abbr] = normalize_nba_team_metrics(st)

            g["team_stats"] = team_stats
            g["team_norm"] = team_norm

        except Exception as e:
            if debug:
                print("ESPN parse error:", event_id, e)
            g["team_stats"] = {}
            g["team_norm"] = {}

        enriched.append(g)

    return enriched


if __name__ == "__main__":
    # quick smoke test
    games = get_espn_features("nba", debug=True)
    print("games:", len(games))
    if games:
        g0 = games[0]
        print(g0.get("away"), "@", g0.get("home"), "event", g0.get("event_id"))
        print("keys(team_stats):", list((g0.get("team_stats") or {}).keys()))
        print("norm sample:", g0.get("team_norm", {}).get(_norm_key(g0.get("home","")), {}))
