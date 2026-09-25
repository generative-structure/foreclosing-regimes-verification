#!/usr/bin/env python3
"""Symbolic verification of the contrast-cancellation identity (the contrast identity).

Verification only. No free parameters, no generated population, no mechanism.
The identity is elementary; this script exists so a reader does not have to take
the three lines on trust.
"""
import sympy as sp

K, N = 4, 6
c = sp.symbols(f'c1:{K+1}')
Y = sp.symbols(f'Y1:{N+1}')
F = sp.Matrix(sp.symbols(f'F1:{N+1}_1:{K+1}')).reshape(N, K)

rows = []

# (A.5, line 1) interchange of two finite sums
lhs = sum(c[k] * sum(Y[i] * F[i, k] for i in range(N)) for k in range(K))
rhs = sum(Y[i] * sum(c[k] * F[i, k] for k in range(K)) for i in range(N))
rows.append(("line 1: sum_k c_k a_k == N^-1 sum_i Y_i (sum_k c_k F_ik)",
             sp.expand(lhs - rhs) == 0))

# (A.5, line 2) the inner sum vanishes on agreement
rows.append(("line 2a: unanimous zero -> inner sum 0",
             sum(c[k] * 0 for k in range(K)) == 0))
centered = {c[K-1]: -sum(c[:K-1])}
rows.append(("line 2b: unanimous one  -> inner sum = sum_k c_k = 0 (centered)",
             sp.simplify(sum(c).subs(centered)) == 0))

# (A.5, line 3) only disagreement units survive, on a concrete flag pattern
pattern = [[0,0,0,0], [1,1,1,1], [1,0,1,0], [0,1,1,0], [1,1,0,0], [1,1,1,1]]
disagree = [i for i, r in enumerate(pattern) if len(set(r)) > 1]
full = sum(c[k] * sum(Y[i] * pattern[i][k] for i in range(N)) for k in range(K))
only_d = sum(Y[i] * sum(c[k] * pattern[i][k] for k in range(K)) for i in disagree)
rows.append((f"line 3: agreement units drop out (disagreement rows {disagree})",
             sp.simplify(sp.expand((full - only_d).subs(centered))) == 0))

# two-method reduction
a_A = sum(Y[i] * pattern[i][0] for i in range(N))
a_B = sum(Y[i] * pattern[i][1] for i in range(N))
d10 = [i for i in range(N) if pattern[i][0] == 1 and pattern[i][1] == 0]
d01 = [i for i in range(N) if pattern[i][0] == 0 and pattern[i][1] == 1]
delta = sum(Y[i] for i in d10) - sum(Y[i] for i in d01)
rows.append(("two-method case c=(1,-1): a_A - a_B == M10 - M01",
             sp.simplify(sp.expand(a_A - a_B - delta)) == 0))

print("=" * 74)
print("A.5  CONTRAST-CANCELLATION IDENTITY  (symbolic, K=4, N=6)")
print("=" * 74)
for label, ok in rows:
    print(f"  [{'PASS' if ok else 'FAIL'}]  {label}")
print()
print("  All checks are exact symbolic identities, not numerical approximations.")
raise SystemExit(0 if all(ok for _, ok in rows) else 1)
