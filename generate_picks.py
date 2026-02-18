import argparse
import requests
from datetime import datetime

ODDS_API_KEY = "75658550dfc5116233a171d37466cac5"

SPORT = "basketball_nba"


# ---------------------------
# Odds API Pull
# ---------------------------
def fetch_odds():
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american"
    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()


# ---------------------------
# Convert American odds → implied probability
# ---------------------------
def implied_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


# ---------------------------
# Choose best sportsbook line
# ---------------------------
def best_market(bookmakers, market_key):
    best = None

    for book in bookmakers:
        for market in book.get("markets", []):
            if market["key"] != market_key:
                continue

            for outcome in market["outcomes"]:
                if best is None or abs(outcome["price"]) < abs(best["price"]):
                    best = outcome

    return best


# ---------------------------
# Sophisticated Pick Logic
# ---------------------------
def generate_picks():
    games = fetch_odds()

    picks = []

    for game in games:
        home = game["home_team"]
        away = game["away_team"]
        books = game["bookmakers"]

        moneyline = best_market(books, "h2h")
        spread = best_market(books, "spreads")
        total = best_market(books, "totals")

        if not moneyline:
            continue

        prob = implied_prob(moneyline["price"])

        confidence = round(prob * 100, 2)

        pick = {
            "matchup": f"{away} @ {home}",
            "bet_type": "Moneyline",
            "selection": moneyline["name"],
            "odds": moneyline["price"],
            "confidence": confidence,
            "spread_hint": spread["point"] if spread else None,
            "total_hint": total["point"] if total else None,
        }

        picks.append(pick)

    picks.sort(key=lambda x: x["confidence"], reverse=True)

    return picks


# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    picks = generate_picks()

    print("\n=== ELITE PICKS ===\n")

    for p in picks[:args.top]:
        print(
            f"{p['matchup']} | "
            f"{p['bet_type']} → {p['selection']} "
            f"({p['odds']}) | "
            f"Confidence: {p['confidence']}%"
        )


if __name__ == "__main__":
    main()
