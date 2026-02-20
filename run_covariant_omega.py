
import os
import json
import datetime as dt

from omega.meta import derive_meta
from omega.allocator import adjust_probability, expected_value, kelly

# pretend this comes from your quant_sim_engine
covariant_output = [
    {
        "game": "LAL vs PHX",
        "market_prob": 0.52,
        "model_edge": 0.09,
        "decimal_odds": 1.95
    }
]

metrics = {
    "brier": 0.23,
    "drawdown": 2.1
}

meta = derive_meta(metrics)

print("REGIME:", meta.regime)

for bet in covariant_output:

    p_model = adjust_probability(
        bet["market_prob"],
        bet["model_edge"]
    )

    edge = p_model - bet["market_prob"]

    if edge < meta.edge_threshold:
        continue

    ev = expected_value(p_model, bet["decimal_odds"])

    stake = kelly(
        p_model,
        bet["decimal_odds"],
        meta.kelly_fraction
    )

    if meta.capital_preservation:
        stake *= 0.5

    print({
        "game": bet["game"],
        "model_prob": round(p_model,3),
        "EV": round(ev,3),
        "stake_units": round(stake,2)
    })

# -----------------------------
# AUDIT LOGGER
# -----------------------------
audit_path = "data/audit.json"

if not os.path.exists(audit_path):
    with open(audit_path, "w") as f:
        json.dump([], f)

with open(audit_path, "r") as f:
    history = json.load(f)

result["timestamp"] = datetime.utcnow().isoformat()

history.append(result)

with open(audit_path, "w") as f:
    json.dump(history, f, indent=2)
