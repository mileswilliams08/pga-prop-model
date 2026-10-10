"""
Turns a player's (course-adjusted) rate stats into actual prop bet
probabilities.
"""

import numpy as np
from scipy import stats


def overdispersed_binomial_pmf(n: int, p: float, k: np.ndarray,
                                phi: float = 1.15) -> np.ndarray:
    var_inflation = phi
    rho = max(1e-6, (var_inflation - 1) / (n - 1)) if n > 1 else 1e-6
    alpha = p * (1 - rho) / rho
    beta = (1 - p) * (1 - rho) / rho
    return stats.betabinom.pmf(k, n, alpha, beta)


def prob_over(n: int, p: float, line: float, phi: float = 1.15) -> float:
    threshold = int(np.floor(line)) + 1
    k = np.arange(threshold, n + 1)
    return overdispersed_binomial_pmf(n, p, k, phi).sum()


def prob_under(n: int, p: float, line: float, phi: float = 1.15) -> float:
    return 1 - prob_over(n, p, line, phi)


def gir_prop(adj_gir_pct: float, line: float, holes: int = 18,
             phi: float = 1.15) -> dict:
    return {
        "over": prob_over(holes, adj_gir_pct, line, phi),
        "under": prob_under(holes, adj_gir_pct, line, phi),
    }


def fairways_prop(adj_driving_accuracy: float, line: float,
                   driving_holes: int = 14, phi: float = 1.15) -> dict:
    return {
        "over": prob_over(driving_holes, adj_driving_accuracy, line, phi),
        "under": prob_under(driving_holes, adj_driving_accuracy, line, phi),
    }


def birdies_or_better_prop(adj_birdie_rate: float, line: float,
                            holes: int = 18, phi: float = 1.2) -> dict:
    return {
        "over": prob_over(holes, adj_birdie_rate, line, phi),
        "under": prob_under(holes, adj_birdie_rate, line, phi),
    }


def total_strokes_prop(adj_scoring_avg: float, line: float,
                        std_dev: float = 2.8) -> dict:
    dist = stats.norm(loc=adj_scoring_avg, scale=std_dev)
    p_under = dist.cdf(line)
    return {"under": p_under, "over": 1 - p_under}


def make_the_cut_prop(adj_scoring_avg: float, projected_cut_line: float,
                       rounds_to_cut: int = 2, std_dev: float = 2.8) -> float:
    two_round_mean = adj_scoring_avg * rounds_to_cut
    two_round_std = std_dev * np.sqrt(rounds_to_cut)
    dist = stats.norm(loc=two_round_mean, scale=two_round_std)
    return dist.cdf(projected_cut_line)


# ---------------------------------------------------------------------------
# EXPERIMENTAL: Pars and Bogeys-or-Worse (see backtest_pars_bogeys.py)
#
# Derived from the stats the model already has, no new player stat needed:
# every hole is a birdie-or-better, a par, or a bogey-or-worse, and strokes
# vs. par = (bogeys + extra on doubles) - (birdies + extra on eagles), so
#   raw_bogeys = (expected strokes - course par) + expected birdies
#   raw_pars   = 18 - expected birdies - raw_bogeys
# The raw numbers exaggerate player-to-player differences, so each is
# shrunk toward the average with a line (intercept + slope) fit on half of
# 2025's tournaments and tested on the other half (backtest: beat the
# base-rate baseline on 7/7 lines; bogeys has a real edge, pars a tiny one).
# Constants below are from that backtest (rounded to the precision it
# printed); refit with backtest_pars_bogeys.py if the model's inputs change.
# ---------------------------------------------------------------------------
PARS_BOGEYS_FIT = {
    "bogeys": {"intercept": 0.877, "slope": 0.60, "phi": 1.19},
    "pars": {"intercept": 7.546, "slope": 0.33, "phi": 1.01},
}
DEFAULT_PARS_LINE = 11.5
DEFAULT_BOGEYS_LINE = 2.5


def expected_pars_bogeys(adj_scoring_avg: float, adj_birdie_rate: float,
                          course_par: float, holes: int = 18, fit: dict = None):
    """Returns (expected_pars, expected_bogeys_or_worse) for one round."""
    fit = fit or PARS_BOGEYS_FIT
    b_hat = holes * min(max(adj_birdie_rate, 0.01), 0.6)
    g_raw = (adj_scoring_avg - course_par) + b_hat
    pars_raw = holes - b_hat - g_raw
    g = min(max(fit["bogeys"]["intercept"] + fit["bogeys"]["slope"] * g_raw, 0.1), 14)
    pars = min(max(fit["pars"]["intercept"] + fit["pars"]["slope"] * pars_raw, 2), 17)
    return pars, g


def pars_prop(expected_pars: float, line: float, holes: int = 18,
               fit: dict = None) -> dict:
    phi = (fit or PARS_BOGEYS_FIT)["pars"]["phi"]
    p = min(max(expected_pars / holes, 0.01), 0.99)
    return {"over": prob_over(holes, p, line, phi), "under": prob_under(holes, p, line, phi)}


def bogeys_prop(expected_bogeys: float, line: float, holes: int = 18,
                 fit: dict = None) -> dict:
    phi = (fit or PARS_BOGEYS_FIT)["bogeys"]["phi"]
    p = min(max(expected_bogeys / holes, 0.01), 0.99)
    return {"over": prob_over(holes, p, line, phi), "under": prob_under(holes, p, line, phi)}
