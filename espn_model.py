from espn_features import get_espn_features

def f(x):
    try:
        return float(str(x).replace('%',''))
    except:
        return 0.0

def last10(v):
    try:
        w,l = str(v).split('-')
        return (float(w)-float(l)) / 10.0   # normalize
    except:
        return 0.0

def streak(v):
    v = str(v or "")
    if v.startswith("W"):
        return float(v[1:]) / 10.0
    if v.startswith("L"):
        return -float(v[1:]) / 10.0
    return 0.0


# -------- calibrated rating --------
def rating(team, opp):

    offense_edge = (f(team.get("avgPoints")) -
                    f(opp.get("avgPointsAgainst"))) / 10.0

    shooting_edge = (f(team.get("fieldGoalPct")) -
                     f(opp.get("fieldGoalPct"))) / 5.0

    turnover_edge = (f(opp.get("avgTotalTurnovers")) -
                     f(team.get("avgTotalTurnovers"))) / 5.0

    form_edge = last10(team.get("Last Ten Games"))
    momentum_edge = streak(team.get("streak"))

    return (
        offense_edge * 0.35 +
        shooting_edge * 0.20 +
        turnover_edge * 0.20 +
        form_edge * 0.15 +
        momentum_edge * 0.10
    )


def generate_picks(league="nba"):

    games = get_espn_features(league)
    picks = []

    for g in games:

        home = g["home"].upper()
        away = g["away"].upper()

        norm = g.get("team_norm", {})

        home_stats = norm.get(home)
        away_stats = norm.get(away)

        if not home_stats or not away_stats:
            continue

        edge = round(
            rating(home_stats, away_stats)
            - rating(away_stats, home_stats),
            3
        )

        if abs(edge) < 0.15:
            conf = "LOW"
        elif abs(edge) < 0.35:
            conf = "MED"
        else:
            conf = "HIGH"

        winner = home if edge > 0 else away

        picks.append({
            "matchup": f"{away} @ {home}",
            "pick": winner,
            "edge": edge,
            "confidence": conf
        })

    picks.sort(key=lambda x: abs(x["edge"]), reverse=True)
    return picks


if __name__ == "__main__":

    picks = generate_picks("nba")

    print("\n===== ESPN MODEL PICKS (CALIBRATED) =====\n")

    for p in picks:
        print(
            f'{p["matchup"]:18} -> {p["pick"]:4} '
            f'EDGE {p["edge"]:>6}   {p["confidence"]}'
        )
