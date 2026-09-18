"""
Compares your model's probability against a sportsbook's posted odds to
find betting edge, and sizes a bet using the Kelly Criterion.
"""


def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)


def american_to_implied_prob(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def expected_value(model_prob: float, american_odds: int,
                    stake: float = 100) -> float:
    """EV in dollars per `stake` risked."""
    decimal_odds = american_to_decimal(american_odds)
    win_amount = stake * (decimal_odds - 1)
    return model_prob * win_amount - (1 - model_prob) * stake


def edge(model_prob: float, american_odds: int) -> float:
    """Your probability minus the sportsbook's implied probability."""
    return model_prob - american_to_implied_prob(american_odds)


def kelly_fraction(model_prob: float, american_odds: int,
                    kelly_multiplier: float = 0.5) -> float:
    """
    Fraction of bankroll to bet. Full Kelly is mathematically optimal for
    long-run growth but very volatile for a model with any real
    uncertainty in its probability estimate — using a fractional Kelly
    (default: half-Kelly) is standard practice to reduce variance and
    protect against the model being overconfident.
    """
    b = american_to_decimal(american_odds) - 1  # net odds
    q = 1 - model_prob
    f = (model_prob * b - q) / b
    return max(0, f) * kelly_multiplier


def evaluate_bet(model_prob: float, american_odds: int,
                  bankroll: float = 1000, kelly_multiplier: float = 0.5) -> dict:
    return {
        "model_prob": round(model_prob, 4),
        "implied_prob": round(american_to_implied_prob(american_odds), 4),
        "edge_pct": round(edge(model_prob, american_odds) * 100, 2),
        "ev_per_100": round(expected_value(model_prob, american_odds, 100), 2),
        "kelly_stake": round(kelly_fraction(model_prob, american_odds,
                                             kelly_multiplier) * bankroll, 2),
        "recommendation": "BET" if edge(model_prob, american_odds) > 0.02 else "PASS",
    }
