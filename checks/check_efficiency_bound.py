#!/usr/bin/env python3
"""The relative-efficiency ratio and its lower bound (the relative-efficiency bound).

Verification only. The parameter grid below is written out explicitly and is
deterministic -- no random draws, no generated population, no mechanism. Every
value is exact rational arithmetic on the stated expressions.
"""
from fractions import Fraction as F
from itertools import product
import sympy as sp

FAILURES = []
def check(label, cond):
    if not cond:
        FAILURES.append(label)
    return "PASS" if cond else "**FAIL**"

print("=" * 78)
print("A.7  Var(W) AND THE LAW OF TOTAL VARIANCE   W = Y (D_A - D_B)")
print("=" * 78)
q10, q01, r10, r01 = sp.symbols("q10 q01 r10 r01", nonnegative=True)
Delta = q10 * r10 - q01 * r01
EW2 = q10 * r10 + q01 * r01                       # W^2 indicates the two TP sets
VarW = sp.simplify(EW2 - Delta**2)
print(f"  [{check('E[W] = Delta', True)}]  E[W] = q10 r10 - q01 r01 = Delta  (by construction)")
print(f"  [{check('Var(W) expression', sp.simplify(VarW - (EW2 - Delta**2)) == 0)}]"
      "  Var(W) = q10 r10 + q01 r01 - Delta^2")
# law of total variance, conditioning on the cell
within = q10 * r10 * (1 - r10) + q01 * r01 * (1 - r01)
between = (q10 * r10**2 + q01 * r01**2) - Delta**2
print(f"  [{check('LTV decomposition', sp.simplify(VarW - (within + between)) == 0)}]"
      "  Var(W) = E[Var(W|C)] + Var(E[W|C])")

print()
print("=" * 78)
print("A.7  THE RATIO IS AT LEAST 1/q_d, HENCE AT LEAST 1   (exact rationals)")
print("=" * 78)
def ratio(a, b, r1, r0):
    D = a * r1 - b * r0
    num = a * r1 + b * r0 - D * D
    den = (sp.sqrt(r1 * (1 - r1)) * a + sp.sqrt(r0 * (1 - r0)) * b) ** 2
    return sp.nsimplify(num / den) if den != 0 else sp.oo

GRID = [F(1, 100), F(1, 20), F(1, 5), F(2, 5), F(1, 2), F(3, 5), F(9, 10), F(99, 100)]
SPLITS = [F(1, 10), F(1, 4), F(1, 2), F(3, 4), F(9, 10)]
QD = [F(1, 100), F(1, 10), F(1, 2), F(9, 10), F(1)]
worst, worst_at, n = None, None, 0
for qd, sp_, r1, r0 in product(QD, SPLITS, GRID, GRID):
    a, b = qd * sp_, qd * (1 - sp_)
    if a == 0 or b == 0:
        continue
    v = float(ratio(a, b, r1, r0))
    n += 1
    if v < float(1 / qd) - 1e-9:
        FAILURES.append(f"ratio < 1/q_d at ({a},{b},{r1},{r0})")
    if worst is None or v < worst:
        worst, worst_at = v, (a, b, r1, r0)
print(f"  grid points evaluated : {n}")
print(f"  smallest ratio seen   : {worst:.6f}  at (q10,q01,r10,r01) = "
      f"({worst_at[0]}, {worst_at[1]}, {worst_at[2]}, {worst_at[3]})")
print(f"  [{check('ratio >= 1/q_d on the whole grid', not any('ratio <' in f for f in FAILURES))}]"
      "  Cauchy-Schwarz bound holds everywhere on the grid")
print(f"  [{check('ratio >= 1 on the whole grid', worst >= 1 - 1e-9)}]  hence ratio >= 1")

print()
print("=" * 78)
print("A.7  THE UNITY BOUNDARY")
print("=" * 78)
r = F(1, 5)
even = float(ratio(F(1, 2), F(1, 2), r, r))
print(f"  q_d = 1, even split, r = 1/5 : ratio = {even:.6f}"
      f"   1/(1-r) = {float(1/(1-r)):.6f}")
print(f"  [{check('q_d->1 even split gives 1/(1-r), not 1', abs(even - float(1/(1-r))) < 1e-9)}]"
      "  q_d = 1 alone does NOT give unity")
tiny = float(ratio(F(1, 2), F(1, 2), F(1, 10**6), F(1, 10**6)))
print(f"  q_d = 1, even split, r -> 0  : ratio = {tiny:.6f}")
print(f"  [{check('q_d->1 with r->0 gives 1', abs(tiny - 1) < 1e-3)}]  unity needs r -> 0 as well")
deg = float(ratio(F(10**6 - 1, 10**6), F(1, 10**6), F(37, 100), F(1, 2)))
print(f"  degenerate partition q10 -> 1 : ratio = {deg:.9f}")
print(f"  [{check('degenerate partition gives exactly 1', abs(deg - 1) < 1e-5)}]"
      "  the designs coincide, and the ratio is 1")

print()
if FAILURES:
    print(f"  {len(FAILURES)} FAILED: " + "; ".join(FAILURES[:5]))
else:
    print("  all assertions passed")
raise SystemExit(1 if FAILURES else 0)
