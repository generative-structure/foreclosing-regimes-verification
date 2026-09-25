"""Data-generating process and closed forms, per PREREG sections 1.1-1.2.

Nothing here is tuned. Every constant is the one written in PREREG.md.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm, multivariate_normal as mvn
from scipy.optimize import brentq
from scipy.integrate import quad

# ---- frozen constants (PREREG section 3) ---------------------------------
N_POP   = 10**6
K       = 20
MU      = 0.5 + 0.1 * np.arange(K)         # fixed ladder, not drawn
T_ROUND = 20
ALPHA   = 0.05
ENTROPY = 20260823
PI_GRID   = (0.001, 0.01)
Q_GRID    = (0.001, 0.005)
RHO_GRID  = (0.3, 0.6, 0.9)
DELTA_FRAC= (0.10, 0.20)
BUDGET    = (100, 400)
I1        = 0                              # initial incumbent: method 1, the weakest


def tau_for(mu: float, pi: float, q: float) -> float:
    """Threshold giving method k the common flag rate q.  PREREG 1.2."""
    f = lambda t: pi * norm.sf(t - mu) + (1 - pi) * norm.sf(t) - q
    return brentq(f, -20.0, 20.0, xtol=1e-13)


def thresholds(pi: float, q: float) -> np.ndarray:
    return np.array([tau_for(m, pi, q) for m in MU])


def closed_forms(pi: float, q: float, rho: float) -> dict:
    """Sensitivity, true-positive mass, and pairwise cell structure.  PREREG 1.2."""
    tau = thresholds(pi, q)
    s = norm.sf(tau - MU)                       # P(F_k=1 | Y=1)
    a = pi * s                                  # true-positive mass
    c = rho ** 2                                # conditional correlation of scores
    rv = mvn(mean=[0.0, 0.0], cov=[[1.0, c], [c, 1.0]])

    def pair(j, k):
        out = {}
        for y in (1, 0):
            sj, sk = norm.sf(tau[j] - MU[j] * y), norm.sf(tau[k] - MU[k] * y)
            b = rv.cdf([-(tau[j] - MU[j] * y), -(tau[k] - MU[k] * y)])
            out[y] = (sj - b, sk - b)           # (j only, k only)
        q10 = pi * out[1][0] + (1 - pi) * out[0][0]
        q01 = pi * out[1][1] + (1 - pi) * out[0][1]
        return dict(q10=q10, q01=q01,
                    N10=N_POP * q10, N01=N_POP * q01,
                    M10=N_POP * pi * out[1][0], M01=N_POP * pi * out[1][1])
    return dict(tau=tau, s=s, a=a, pair=pair)


def union_coverage(pi: float, q: float, rho: float) -> float:
    """P(some method flags the unit), by the one-dimensional integral of PREREG 1.2."""
    tau = thresholds(pi, q)
    d = np.sqrt(1 - rho ** 2)
    def all_zero(y):
        g = lambda u: norm.pdf(u) * np.prod(norm.cdf((tau - MU * y - rho * u) / d))
        return quad(g, -9, 9, limit=200)[0]
    return 1.0 - (pi * all_zero(1) + (1 - pi) * all_zero(0))


def n_flag(q: float) -> int:
    """Units each method flags.  PREREG-AMENDMENT-1 section 2.1."""
    return int(round(q * N_POP))


def realize_population(pi: float, q: float, rho: float, rng) -> dict:
    """One realized population of N units.  Flags stored as a boolean matrix.

    PREREG-AMENDMENT-1 section 2.1: each method flags EXACTLY the top n_flag(q) units
    by its own score, not the units above a fixed threshold.  This makes matched flag
    volume a realized identity (N_10 = N_01 exactly for every ordered pair) rather than
    an equality in expectation, which is what the finite-population design condition of
    section 4 requires.  The cost is that the fixed-threshold closed forms become
    approximations; see PREREG-AMENDMENT-1 section 2.2.
    """
    nf = n_flag(q)
    Y = rng.random(N_POP) < pi
    U = rng.standard_normal(N_POP)
    base = rho * U
    scale = np.sqrt(1 - rho ** 2)
    F = np.zeros((K, N_POP), dtype=bool)
    for k in range(K):
        S = MU[k] * Y + base + scale * rng.standard_normal(N_POP)
        idx = np.argpartition(S, -nf)[-nf:]     # top nf by score, ties broken arbitrarily
        F[k, idx] = True
    return dict(Y=Y, F=F)


def pair_table(pop: dict) -> dict:
    """Exact cell counts for every ordered pair, from the realized population."""
    Y, F = pop["Y"], pop["F"]
    Fi = [np.flatnonzero(F[k]) for k in range(K)]
    Yb = Y
    N10 = np.zeros((K, K), dtype=np.int64); M10 = np.zeros((K, K), dtype=np.int64)
    for j in range(K):
        fj = F[j]
        for k in range(K):
            if j == k:
                continue
            only = fj & ~F[k]
            N10[j, k] = only.sum()
            M10[j, k] = (only & Yb).sum()
    return dict(N10=N10, M10=M10,
                a=np.array([(F[k] & Yb).sum() for k in range(K)]) / N_POP,
                s=np.array([(F[k] & Yb).sum() for k in range(K)]) / max(Yb.sum(), 1),
                flag_rate=np.array([F[k].sum() for k in range(K)]) / N_POP,
                union_cov=np.any(F, axis=0).mean(),
                prevalence=Yb.mean())
