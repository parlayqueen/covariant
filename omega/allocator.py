from math import log, exp

def sigmoid(x):
    return 1/(1+exp(-x))

def logit(p):
    return log(p/(1-p))

def adjust_probability(market_prob, covariant_edge):
    shift = max(min(covariant_edge, 0.35), -0.35)
    return sigmoid(logit(market_prob) + shift)

def expected_value(p, decimal_odds):
    return p * decimal_odds - 1

def kelly(p, decimal_odds, frac):
    b = decimal_odds - 1
    q = 1 - p
    f = (b*p - q)/b
    return max(0, f * frac)

