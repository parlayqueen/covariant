#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------
# Odds math
# ---------------------------
def american_to_decimal(american: float) -> float:
    a = float(american)
    if a == 0:
        return 1.0
    if a > 0:
        return 1.0 + (a / 100.0)
    return 1.0 + (100.0 / abs(a))

def american_to_implied_prob(american: float) -> float:
    a = float(american)
    if a > 0:
        return 100.0 / (a + 100.0)
    return abs(a) / (abs(a) + 100.0)

def decimal_to_net_odds(decimal_odds: float) -> float:
    # net profit per $1 staked
    return max(0.0, float(decimal_odds) - 1.0)

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

# ---------------------------
# Market aggregation
# ---------------------------
def remove_vig_two_way(p1: float, p2: float) -> Tuple[float, float]:
    s = p1 + p2
    if s <= 0:
        return 0.5, 0.5
    return p1 / s, p2 / s

def trimmed_mean(vals: List[float], trim_frac: float = 0.15) -> Optional[float]:
    if not vals:
        return None
    xs = sorted(vals)
    n = len(xs)
    k = int(n * trim_frac)
    if n - 2*k <= 0:
        return sum(xs) / n
    xs2 = xs[k:n-k]
    return sum(xs2) / len(xs2)

def consensus_prob_from_books(book_probs: List[float]) -> Optional[float]:
    # robust consensus: trim outliers, then mean
    return trimmed_mean([clamp(p, 1e-6, 1.0 - 1e-6) for p in book_probs], trim_frac=0.15)

# ---------------------------
# Model/market blending
# ---------------------------
def blend_model_with_market(p_model: float, p_market: float, confidence: float) -> float:
    """
    confidence in [0,1]: higher means trust model more.
    """
    p_model = clamp(p_model, 1e-6, 1.0 - 1e-6)
    p_market = clamp(p_market, 1e-6, 1.0 - 1e-6)
    w = clamp(confidence, 0.0, 1.0)
    return clamp(w * p_model + (1.0 - w) * p_market, 1e-6, 1.0 - 1e-6)

# ---------------------------
# EV + variance
# ---------------------------
def ev_per_dollar(p: float, dec_odds: float) -> float:
    """
    $EV for $1 stake. If win: profit = (dec-1). If lose: -1.
    """
    b = decimal_to_net_odds(dec_odds)
    p = clamp(p, 1e-9, 1.0 - 1e-9)
    return p * b - (1.0 - p)

def variance_per_dollar(p: float, dec_odds: float) -> float:
    """
    Variance of profit for $1 stake.
    Outcomes: +b with prob p, -1 with prob (1-p)
    """
    b = decimal_to_net_odds(dec_odds)
    p = clamp(p, 1e-9, 1.0 - 1e-9)
    mu = p * b - (1.0 - p)
    v = p * (b - mu) ** 2 + (1.0 - p) * (-1.0 - mu) ** 2
    return max(0.0, v)

def kelly_fraction(p: float, dec_odds: float) -> float:
    """
    Classic Kelly for binary bet with net odds b.
    f* = (b p - q)/b
    """
    b = decimal_to_net_odds(dec_odds)
    if b <= 0:
        return 0.0
    p = clamp(p, 1e-9, 1.0 - 1e-9)
    q = 1.0 - p
    f = (b * p - q) / b
    return max(0.0, f)

# ---------------------------
# Exposure control
# ---------------------------
@dataclass
class ExposureState:
    per_event_staked: Dict[str, float]
    per_type_staked: Dict[str, float]

    def __init__(self) -> None:
        self.per_event_staked = {}
        self.per_type_staked = {}

    def get_event(self, event_id: str) -> float:
        return float(self.per_event_staked.get(event_id, 0.0))

    def add_event(self, event_id: str, amt: float) -> None:
        self.per_event_staked[event_id] = self.get_event(event_id) + float(amt)

    def get_type(self, market_type: str) -> float:
        return float(self.per_type_staked.get(market_type, 0.0))

    def add_type(self, market_type: str, amt: float) -> None:
        self.per_type_staked[market_type] = self.get_type(market_type) + float(amt)

# ---------------------------
# Main sizing function
# ---------------------------
def stake_for_pick_v3(
    *,
    bankroll: float,
    p_model: float,
    p_market: float,
    dec_odds: float,
    event_id: str,
    market_type: str,
    exposure: ExposureState,
    # limits
    max_per_game_frac: float = 0.04,
    max_per_type_frac: float = 0.08,
    max_daily_frac: float = 0.10,
    total_staked_so_far: float = 0.0,
    # model trust / risk tuning
    model_confidence: float = 0.62,
    fractional_kelly: float = 0.35,
    var_penalty: float = 0.85,
    min_stake: float = 1.0,
    round_to: float = 1.0,
) -> Dict[str, float]:
    """
    Returns stake + diagnostics.
    """

    bankroll = max(0.0, float(bankroll))
    if bankroll <= 0:
        return {"stake": 0.0, "p_used": 0.0, "ev": 0.0, "edge": 0.0, "kelly": 0.0, "stdev": 0.0}

    # 1) Blend model with market to avoid overconfidence
    p_used = blend_model_with_market(p_model, p_market, model_confidence)

    # 2) Compute EV/edge/variance
    ev = ev_per_dollar(p_used, dec_odds)
    edge = p_used - p_market
    var_ = variance_per_dollar(p_used, dec_odds)
    stdev = math.sqrt(var_)

    # 3) Kelly sizing
    k = kelly_fraction(p_used, dec_odds)
    k_adj = fractional_kelly * k

    # 4) Variance penalty: smaller stake when outcome volatility is high
    # penalty in (0,1], stronger as stdev rises
    penalty = 1.0 / (1.0 + var_penalty * stdev)

    # 5) Raw stake
    raw = bankroll * k_adj * penalty

    # 6) Exposure caps
    max_per_game = bankroll * max_per_game_frac
    max_per_type = bankroll * max_per_type_frac
    max_daily = bankroll * max_daily_frac

    already_event = exposure.get_event(event_id)
    already_type = exposure.get_type(market_type)

    raw = min(raw, max(0.0, max_per_game - already_event))
    raw = min(raw, max(0.0, max_per_type - already_type))
    raw = min(raw, max(0.0, max_daily - float(total_staked_so_far)))

    # 7) Min + rounding
    if raw < min_stake:
        stake = 0.0
    else:
        stake = raw
        if round_to > 0:
            stake = round(stake / round_to) * round_to

    return {
        "stake": float(stake),
        "p_used": float(p_used),
        "ev": float(ev),
        "edge": float(edge),
        "kelly": float(k),
        "stdev": float(stdev),
        "penalty": float(penalty),
    }

# ---------------------------
# Helper: derive market prob from book odds
# ---------------------------
def market_prob_from_two_way_american(leg_american: float, opp_american: float) -> float:
    p1 = american_to_implied_prob(leg_american)
    p2 = american_to_implied_prob(opp_american)
    p1n, _ = remove_vig_two_way(p1, p2)
    return clamp(p1n, 1e-6, 1.0 - 1e-6)
