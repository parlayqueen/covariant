import os
from typing import Any, Dict, List, Optional

MARKET_PROVIDER = os.getenv("MARKET_PROVIDER", "oddsapi").lower().strip()

# Providers
import odds_client

# Optional: if you want to support SportsData odds too, we import lazily
def _sportsdata_get():
    from sportsdata_client import get
    return get


class MarketRouterError(RuntimeError):
    pass


def fetch_market_odds_mlb(
    *,
    oddsapi_sport_key: str = "baseball_mlb",
    regions: Optional[str] = None,
    markets: str = "h2h,spreads,totals",
    bookmakers: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Unified entry point: returns normalized events with books/markets.
    Default behavior: Odds API for live odds.
    """
    provider = MARKET_PROVIDER

    if provider == "oddsapi":
        data = odds_client.get_odds(
            oddsapi_sport_key,
            regions=regions or odds_client.DEFAULT_REGIONS,
            markets=markets,
            bookmakers=bookmakers,
        )
        return [odds_client.normalize_oddsapi_event(x) for x in data]

    if provider == "sportsdata":
        # Example SportsDataIO odds endpoint usage (adjust path to your existing code’s endpoints)
        # You likely already call GameOddsByDate; keep it but route through sportsdata_client.get
        get = _sportsdata_get()
        # You MUST pass a date string if using date-based endpoints; leave as caller responsibility.
        raise MarketRouterError("sportsdata provider selected but no date-based odds endpoint implemented here.")

    raise MarketRouterError(f"Unknown MARKET_PROVIDER={provider}. Use 'oddsapi' or 'sportsdata'.")
