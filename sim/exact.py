"""Exact finite-population test, Appendix B with Lemma B.1.

Matched flag volume gives N_C = N_I and the minimax allocation gives m_C = m_I, so
the sampling fractions are equal and the raw count difference D = T_C - T_I is a
valid statistic (Appendix B).  The composite null M_C - M_I <= d is maximised on the
clipped boundary M_C = min(N, M_I + d), which is Lemma B.1.

d may be ANY integer.  After round 1 the residual null carries the labelled imbalance
(Appendix B, residual-label test) and d is routinely negative, in which case M_I must
start at max(0, -d) rather than 0 -- otherwise M_C = min(N, M_I + d) goes negative and
indexes the pmf table from the wrong end.  Lemma B.1 is stated for every integer d.
"""
from __future__ import annotations
import numpy as np
from scipy.special import gammaln

_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

def _pmf_and_cdf(N: int, m: int):
    """rows M = 0..N, cols t = 0..m: pmf and cdf of Hypergeometric(N, M, m)."""
    key = (N, m)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    M = np.arange(N + 1)[:, None].astype(np.float64)
    t = np.arange(m + 1)[None, :].astype(np.float64)
    with np.errstate(invalid="ignore"):
        lg = (gammaln(M + 1) - gammaln(t + 1) - gammaln(M - t + 1)
              + gammaln(N - M + 1) - gammaln(m - t + 1) - gammaln(N - M - m + t + 1)
              - (gammaln(N + 1) - gammaln(m + 1) - gammaln(N - m + 1)))
        P = np.exp(lg)
    lo = np.maximum(0.0, m - (N - M))
    hi = np.minimum(float(m), M)
    P = np.where((t >= lo) & (t <= hi), P, 0.0)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    s = P.sum(axis=1, keepdims=True)
    P = np.divide(P, s, out=np.zeros_like(P), where=s > 0)
    C = np.cumsum(P, axis=1)
    if len(_CACHE) > 4:
        _CACHE.clear()
    _CACHE[key] = (P, C)
    return P, C

def sup_null_tail(N: int, m: int, d: int, c: int) -> float:
    """sup over the clipped null boundary of P(T_C - T_I >= c).  Lemma B.1, any integer d."""
    if d < -N:
        return 0.0                      # null region empty: no feasible (M_C, M_I)
    if c > m:
        return 0.0                      # the event is empty: T_C <= m, T_I >= 0
    if c <= -m:
        return 1.0
    P, C = _pmf_and_cdf(N, m)
    MI = np.arange(max(0, -d), N + 1)   # M_I below -d admits no feasible M_C
    if MI.size == 0:
        return 0.0
    pC = P[np.minimum(N, MI + d)]       # rows for M_C on the clipped boundary
    t = np.arange(m + 1)
    j = t - c                           # need T_I <= t - c
    keep = j >= 0
    if not keep.any():
        return 0.0
    jj = np.minimum(j[keep], m)
    return float((pC[:, keep] * C[MI][:, jj]).sum(axis=1).max())

def critical_value(N: int, m: int, d: int, alpha: float) -> int | None:
    """Smallest c with sup-null P(D >= c) <= alpha; None if the test cannot reject."""
    if sup_null_tail(N, m, d, m) > alpha:
        return None                     # even the most extreme outcome cannot reject
    lo, hi = -m, m
    while lo < hi:
        mid = (lo + hi) // 2
        if sup_null_tail(N, m, d, mid) <= alpha:
            hi = mid
        else:
            lo = mid + 1
    return int(lo)
