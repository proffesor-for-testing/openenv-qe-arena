"""The null for leaderboard numbers: what k runs score if no run is better than the base model.

The baseline is ONE attempt per held-out task (5 per domain), so its per-domain rate is itself
uncertain. The model-based null draws each domain's true rate from Beta(s + .5, n - s + .5) (Jeffreys,
from the baseline's s of n) and lets the k runs share it: P(max <= x) = E_p[F(x; n, p)^k]. A plug-in
binomial at s/n is kept as a lower reference, and a sensitivity line fixes the rate of the domains the
baseline scored 0 in. Tasks within a domain are treated as exchangeable; real tasks are not, and a
domain of a few always-solved and a few never-solved tasks has a lower null than this."""

from __future__ import annotations

import math

import numpy as np

PRIOR = 0.5
# p = sin^2(theta) turns Beta(a, b) with a, b >= 1/2 into a smooth integrand on (0, pi/2).
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(400)
_THETA = (_NODES + 1.0) * np.pi / 4.0


def binom_pmf(n: int, p: float) -> np.ndarray:
    return np.array([math.comb(n, x) * p ** x * (1.0 - p) ** (n - x) for x in range(n + 1)])


def _beta_grid(a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    p = np.sin(_THETA) ** 2
    w = _WEIGHTS * np.sin(_THETA) ** (2 * a - 1) * np.cos(_THETA) ** (2 * b - 1)
    return p, w / w.sum()


def max_pmf(n: int, k: int, p: float | None = None, solved: int | None = None) -> np.ndarray:
    """pmf of the max of k runs on n tasks: binomial at a fixed rate ``p``, or the beta-binomial
    predictive from the baseline's ``solved`` of n (shared rate across the k runs)."""
    if p is not None:
        cdf = np.cumsum(binom_pmf(n, p)) ** k
    else:
        grid, weight = _beta_grid(solved + PRIOR, n - solved + PRIOR)
        cdfs = np.cumsum(np.array([binom_pmf(n, q) for q in grid]), axis=1) ** k  # nodes x (n+1)
        cdf = np.minimum(weight @ cdfs, 1.0)
    return np.diff(np.concatenate([[0.0], cdf]))


def composite_pmf(k: int, domains: dict[str, tuple[int, int]], mode: str = "model",
                  p_zero: float | None = None) -> np.ndarray:
    """Distribution of the best-per-domain composite of k null runs. ``domains`` maps a domain to
    (tasks, baseline solved). mode "model": beta-binomial predictive per domain; "plugin": binomial at
    solved/tasks. ``p_zero`` fixes the rate of domains the baseline scored 0 in."""
    dist = np.array([1.0])
    for n, solved in domains.values():
        if p_zero is not None and solved == 0:
            pmf = max_pmf(n, k, p=p_zero)
        elif mode == "plugin":
            pmf = max_pmf(n, k, p=solved / n)
        else:
            pmf = max_pmf(n, k, solved=solved)
        dist = np.convolve(dist, pmf)
    return dist


def mean(dist: np.ndarray) -> float:
    return float(np.dot(np.arange(len(dist)), dist))


def tail(dist: np.ndarray, x: int) -> float:
    return float(dist[x:].sum()) if x < len(dist) else 0.0
