#!/usr/bin/env python3
"""The minimality converse, exhaustively at small K (Theorem: minimal common support).

Verification of supplied arithmetic. No mechanism is simulated, no population is
generated, and no output enters the argument. For every non-unanimous pattern the
script constructs the witness contrast e_j - e_l of the proof and confirms it is
centered and places nonzero weight on that pattern; it also confirms that the two
unanimous patterns are annihilated by every centered contrast, over a spanning set.
"""
from itertools import product

FAILURES = []
def check(label, cond):
    if not cond:
        FAILURES.append(label)
    return "PASS" if cond else "**FAIL**"

def witness(z):
    """e_j - e_l with z_j = 1, z_l = 0.  Returns None for unanimous z."""
    K = len(z)
    ones = [j for j in range(K) if z[j] == 1]
    zeros = [l for l in range(K) if z[l] == 0]
    if not ones or not zeros:
        return None
    j, l = ones[0], zeros[0]
    c = [0] * K
    c[j], c[l] = 1, -1
    return c

def dot(c, z):
    return sum(ci * zi for ci, zi in zip(c, z))

print("=" * 76)
print("MINIMALITY CONVERSE: every non-unanimous pattern admits a centered witness")
print("=" * 76)
print(f"  {'K':>3} {'patterns':>9} {'non-unanimous':>14} {'witness found':>14} {'centered':>9} {'c.z != 0':>9}")
for K in range(2, 9):
    pats = list(product((0, 1), repeat=K))
    nonu = [z for z in pats if 0 < sum(z) < K]
    found = centered = nonzero = 0
    for z in nonu:
        c = witness(z)
        if c is None:
            continue
        found += 1
        centered += (sum(c) == 0)
        nonzero += (dot(c, z) != 0)
    print(f"  {K:>3} {len(pats):>9} {len(nonu):>14} {found:>14} {centered:>9} {nonzero:>9}")
    check(f"K={K}: witness for every non-unanimous pattern", found == len(nonu))
    check(f"K={K}: every witness is centered", centered == len(nonu))
    check(f"K={K}: every witness weights its pattern", nonzero == len(nonu))

print()
print("=" * 76)
print("SUFFICIENCY: the two unanimous patterns are annihilated by all centered c")
print("=" * 76)
# a spanning set of the centered subspace: e_1 - e_k for k = 2..K
for K in range(2, 9):
    basis = []
    for k in range(1, K):
        c = [0] * K
        c[0], c[k] = 1, -1
        basis.append(c)
    zero, one = tuple([0] * K), tuple([1] * K)
    ok0 = all(dot(c, zero) == 0 for c in basis)
    ok1 = all(dot(c, one) == 0 for c in basis)
    print(f"  K={K:>2}  c.0 = 0 for all basis c: {str(ok0):<5}   c.1 = 0 for all basis c: {ok1}")
    check(f"K={K}: all-zero pattern annihilated", ok0)
    check(f"K={K}: all-one pattern annihilated", ok1)

print()
print("=" * 76)
print("A SPECIFIED CONTRAST CAN NEED STRICTLY LESS  (the (1,-1,0) example)")
print("=" * 76)
c = [1, -1, 0]
pats3 = [z for z in product((0, 1), repeat=3) if 0 < sum(z) < 3]
carried = [z for z in pats3 if dot(c, z) != 0]
dropped = [z for z in pats3 if dot(c, z) == 0]
print(f"  non-unanimous patterns at K=3 : {len(pats3)}")
print(f"  weighted by c = (1,-1,0)      : {len(carried)}  {carried}")
print(f"  annihilated, may be dropped   : {len(dropped)}  {dropped}")
print(f"  [{check('(0,0,1) is non-unanimous yet annihilated', (0, 0, 1) in dropped)}]"
      "  the specified-contrast support is a proper subset")

print()
if FAILURES:
    print(f"  {len(FAILURES)} FAILED: " + "; ".join(FAILURES[:5]))
else:
    print("  all assertions passed")
raise SystemExit(1 if FAILURES else 0)
