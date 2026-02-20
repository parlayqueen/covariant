from dataclasses import dataclass

@dataclass
class MetaState:
    regime: str
    edge_threshold: float
    kelly_fraction: float
    capital_preservation: bool


def derive_meta(metrics):
    brier = metrics.get("brier", 0.25)
    drawdown = metrics.get("drawdown", 0)

    if drawdown > 6:
        regime = "CHAOTIC"
    elif brier > 0.30:
        regime = "OVERCONFIDENT"
    elif brier < 0.20:
        regime = "UNDERCONFIDENT"
    else:
        regime = "CALIBRATED"

    mult = {
        "CALIBRATED": 1.0,
        "OVERCONFIDENT": 1.4,
        "UNDERCONFIDENT": 0.85,
        "CHAOTIC": 1.6,
    }[regime]

    return MetaState(
        regime=regime,
        edge_threshold=0.02 * mult,
        kelly_fraction=0.25 * (0.6 if regime=="CHAOTIC" else 1.0),
        capital_preservation=(regime in ["CHAOTIC","OVERCONFIDENT"])
    )

