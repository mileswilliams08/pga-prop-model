"""
Turns a player's (course-adjusted) rate stats into actual prop bet
probabilities.

Model choices, and why:

- Greens in Regulation, Fairways Hit, Birdies-or-Better:
  These are all "did it happen on this hole, yes/no" events repeated
  across ~18 holes (14 for fairways, since par-3s don't have a fairway).
  A Binomial distribution is the natural fit: n independent-ish trials,
  each with the same success probability.

  Real golf rounds have *some* hole-to-hole correlation (a player who's
  swinging well tends to hit more greens across the whole round, not
  independently on each hole), so pure Binomial slightly understates the
  spread (variance) of outcomes. We correct for this with an
  "overdispersion" factor — see `overdispersed_binomial_pmf`.

- Total strokes (round score):
  Modeled as Normal around the course-adjusted scoring average, with a
  standard deviation estimated from real PGA Tour round-to-round
  variance (~2.6-3.0 strokes is typical; tune this from historical data
  when you have it).
"""

import numpy as np
from scipy import stats


def overdispersed_binomial_pmf(n: int, p: float, k: np.ndarray,
                                phi: float = 1.15) -> np.ndarray:
    """
    Binomial PMF with a variance inflation factor phi (phi=1 is plain
    Binomial). We approximate overdispersion using a Beta-Binomial,
    which is the standard trick: it keeps the same mean but fattens the
    tails, matching real-world "streaky" rounds better than iid Binomial.

    phi is the ratio (actual variance / binomial variance). A phi of
    1.1-1.3 is a reasonable starting assumption for golf; tighten this
    once you have your own round-level data to fit against.
    """
    var_inflation = phi
    # Convert phi to Beta-Binomial's concentration parameter.
    # Binomial variance = n*p*(1-p); Beta-Binomial variance =
    # n*p*(1-p) * (n + rho*(n-1)) / (n+1)-ish approximations vary, so we
    # solve for rho (intra-round correlation) directly from phi.
    rho = max(1e-6, (var_inflation - 1) / (n - 1)) if n > 1 else 1e-6
    alpha = p * (1 - rho) / rho
    beta = (1 - p) * (1 - rho) / rho
    return stats.betabinom.pmf(k, n, alpha, beta)


def prob_over(n: int, p: float, line: float, phi: float = 1.15) -> float:
    """P(count > line), e.g. line=3.5 birdies -> P(birdies >= 4)."""
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
    """
    P(strokes over/under a line), e.g. "Over/Under 69.5 for the round".
    Uses a Normal approximation to score distribution, which is standard
    in golf modeling and works well outside the extreme tails.
    """
    dist = stats.norm(loc=adj_scoring_avg, scale=std_dev)
    p_under = dist.cdf(line)
    return {"under": p_under, "over": 1 - p_under}


def make_the_cut_prop(adj_scoring_avg: float, projected_cut_line: float,
                       rounds_to_cut: int = 2, std_dev: float = 2.8) -> float:
    """
    Rough estimate of P(make the cut): treats the 2-round total as
    Normal(2*adj_scoring_avg, std_dev*sqrt(2)) and checks it against a
    projected cut line (you supply this — e.g. from historical cut data
    for the course, or a live estimate once Round 1 is underway).
    """
    two_round_mean = adj_scoring_avg * rounds_to_cut
    two_round_std = std_dev * np.sqrt(rounds_to_cut)
    dist = stats.norm(loc=two_round_mean, scale=two_round_std)
    return dist.cdf(projected_cut_line)
