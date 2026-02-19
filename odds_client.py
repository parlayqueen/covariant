import os
import time
import requests
from typing import Any, Dict, List, Optional, Tuple

ODDSAPI_BASE = os.getenv("ODDSAPI_BASE", "https://api.the-odds-api.com/v4").rstrip("/")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()

DEFAULT_REGIONS = os.getenv("ODDSAPI_REGIONS", "us")
DEFAULT_ODDS_FORMAT = os.getenv("ODDSAPI_ODDS_FORMAT", "american")
DEFAULT_DATE_FORMAT = os.getenv("ODDSAPI_DATE_FORMAT", "iso")


class OddsApiError(RuntimeError):
    pass


def _require_key() -> None:
    if not ODDS_API_KEY:
        raise OddsApiError("Missing ODDS_API_KEY in environment/.env")


def _get(url: str, params: Dict[str, Any], timeout: int = 25) -> Any:
    """
    Robust GET with light backoff for rate limits/transient failures.
    Odds API uses apiKey as query param.
    """
    _require_key()
    params = dict(params or {})
    params["apiKey"] = ODDS_API_KEY

    last_err = None
    for attempt in range(1, 5):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                # rate limited
                sleep_s = min(2 ** attempt, 12)
                time.sleep(sleep_s)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 8))
    raise OddsApiError(f"Odds API request failed after retries: {last_err}")


def list_sports() -> List[Dict[str, Any]]:
    url = f"{ODDSAPI_BASE}/sports"
    return _get(url, {"all": "true"})


def get_odds(
    sport_key: str,
    regions: str = DEFAULT_REGIONS,
    markets: str = "h2h,spreads,totals",
    bookmakers: Optional[str] = None,
    odds_format: str = DEFAULT_ODDS_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> List[Dict[str, Any]]:
    """
    Returns list of events with books/markets/outcomes.
    sport_key example (Odds API): 'baseball_mlb'
    markets: 'h2h,spreads,totals' (add 'player_props' variants if plan supports)
    """
    url = f"{ODDSAPI_BASE}/sports/{sport_key}/odds"
    params: Dict[str, Any] = {
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    return _get(url, params)


def normalize_oddsapi_event(evt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize Odds API event into a simplified schema Covariant can consume.
    Keeps original payload under 'raw' for traceability.
    """
    out = {
        "provider": "oddsapi",
        "id": evt.get("id"),
        "sport_key": evt.get("sport_key"),
        "sport_title": evt.get("sport_title"),
        "commence_time": evt.get("commence_time"),
        "home_team": evt.get("home_team"),
        "away_team": evt.get("away_team"),
        "bookmakers": [],
        "raw": evt,
    }

    for bk in evt.get("bookmakers", []) or []:
        bko = {"key": bk.get("key"), "title": bk.get("title"), "last_update": bk.get("last_update"), "markets": []}
        for m in bk.get("markets", []) or []:
            mo = {"key": m.get("key"), "last_update": m.get("last_update"), "outcomes": []}
            for oc in m.get("outcomes", []) or []:
                mo["outcomes"].append(
                    {
                        "name": oc.get("name"),
                        "price": oc.get("price"),
                        "point": oc.get("point"),
                        "description": oc.get("description"),
                    }
                )
            bko["markets"].append(mo)
        out["bookmakers"].append(bko)
    return out
