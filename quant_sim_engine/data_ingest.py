import os
import requests
import json
from quant_sim_engine.sim.covariance_engine import PlayerCovarianceProfile
from quant_sim_engine.sim.joint_sampler import PlayerDistribution

def fetch_props():
    api_key = os.environ.get("ODDS_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("ODDS_API_KEY env var not set. Example: export ODDS_API_KEY='...'" )

    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "player_points",   # may vary by plan/market availability
        "oddsFormat": "american",
    }

    r = requests.get(url, params=params, timeout=30)
    # Print quick diagnostics
    print("HTTP:", r.status_code)
    print("Headers (requests remaining):", r.headers.get("x-requests-remaining"))
    print("Headers (requests used):", r.headers.get("x-requests-used"))

    try:
        data = r.json()
    except Exception:
        print("Non-JSON response (first 500 chars):")
        print(r.text[:500])
        raise

    # If API returns an error payload, it’s usually a dict with message/code
    if isinstance(data, dict):
        print("API returned a dict (likely error payload). Keys:", list(data.keys()))
        print("Payload (first 800 chars):")
        print(json.dumps(data, indent=2)[:800])
        raise SystemExit("Stopping: Odds API did not return an events list.")

    if not isinstance(data, list):
        raise SystemExit(f"Unexpected response type: {type(data)}")

    return data

def extract_distributions(events):
    dists = []
    # Use first bookmaker/market by default to avoid duplicates
    for game in events:
        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            continue
        book = bookmakers[0]
        markets = book.get("markets", [])
        if not markets:
            continue

        for market in markets:
            outcomes = market.get("outcomes", [])
            for outcome in outcomes:
                # Odds API prop outcomes typically: name=Over/Under, description=player, point=line
                player = outcome.get("description") or outcome.get("name")
                line = outcome.get("point")
                side = outcome.get("name", "")
                if player is None or line is None:
                    continue
                # Keep only Over so we don't duplicate the same line twice
                if str(side).lower() != "over":
                    continue

                # Placeholder std; replace with rolling game-log std later
                std = 6.0

                dists.append(PlayerDistribution(
                    player_id=str(player),
                    mean=float(line),
                    std=float(std),
                ))
        break

    return dists

def mock_profiles(distributions):
    profiles = []
    for d in distributions:
        profiles.append(PlayerCovarianceProfile(
            player_id=d.player_id,
            usage_rate=0.25,
            assist_rate=0.15,
            rebound_rate=0.12,
            position="G",
            minutes_avg=32.0,
        ))
    return profiles

if __name__ == "__main__":
    events = fetch_props()
    print("Events returned:", len(events))
    dists = extract_distributions(events)
    profiles = mock_profiles(dists)
    print("Players with distributions:", len(dists))
    if dists:
        print("Sample:", dists[0])
