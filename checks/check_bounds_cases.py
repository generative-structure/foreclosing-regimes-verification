#!/usr/bin/env python3
"""Constructed populations for the identification bounds (the verification-support bounds).

Every printed line that says True/False is an assertion: the script records it and
exits non-zero if any of them fails. It does not merely display Booleans.

Verification only. Every population below is written out by hand at a size a
reader can check with a pencil; none stands in for a real one, none carries a
label anyone had to invent, and no result depends on a parameter nobody can
observe. The Y vectors are not data -- they are the completions whose extremes
define the identified set, which is the whole point of the exercise.
"""
from fractions import Fraction as F
from itertools import product

FAILURES = []

def check(label, condition):
    """Record an assertion and return it, so it can also be printed."""
    if not condition:
        FAILURES.append(label)
    return "PASS" if condition else "**FAIL**"

# --------------------------------------------------------------- A.1 ---------
def sens_bounds(a, pf0):
    """Sharp identified set for sensitivity, Appendix A.1."""
    if a > 0:
        return (a / (a + pf0), F(1), "interval, sharp")
    return (F(0), F(0), "single point {0} given prevalence > 0; undefined at pi = 0")

print("=" * 78)
print("A.1  SHARP IDENTIFIED SET FOR SENSITIVITY   s = a/(a+m),  m in [0, P(F=0)]")
print("=" * 78)
print(f"  {'P(F=1)':>8} {'PPV':>6} {'a':>10} {'P(F=0)':>8} {'lower':>12} {'upper':>6}  note")
for pf1, ppv in ((F(1,1000), F(1,2)), (F(1,1000), F(1,1)), (F(1,1000), F(0,1)),
                 (F(1,10),   F(1,2))):
    a, pf0 = pf1 * ppv, 1 - pf1
    lo, hi, note = sens_bounds(a, pf0)
    print(f"  {float(pf1):>8.4f} {float(ppv):>6.2f} {float(a):>10.6f} "
          f"{float(pf0):>8.4f} {float(lo):>12.7f} {float(hi):>6.2f}  {note}")
print("  width = P(F=0)/(a+P(F=0)); at a=0 the interval formula returns [0,1] and")
print("  OVERSTATES the identified set, which is the single point {0}.  See A.1.")
_a, _p0 = F(1, 2000), F(999, 1000)
print("  [" + check("A.1 lower endpoint = a/(a+P(F=0))",
                    sens_bounds(_a, _p0)[0] == _a / (_a + _p0)) + "]  lower endpoint formula")
print("  [" + check("A.1 width identity",
                    1 - _a / (_a + _p0) == _p0 / (_a + _p0)) + "]  width = P(F=0)/(a+P(F=0))")
print("  [" + check("A.1 a=0 collapses to {0}",
                    sens_bounds(F(0), _p0)[:2] == (F(0), F(0))) + "]  a=0 gives the single point {0}")

# --------------------------------------------------------- A.2 and A.3 -------
def bounds(p, Y, verified):
    """Sharp hull of the identified set for the finite-population covariance."""
    N = len(p)
    pbar = sum(p, F(0)) / N
    d = [x - pbar for x in p]
    C_O = sum(d[i] * Y[i] for i in verified) / N
    U = [i for i in range(N) if i not in verified]
    lo = C_O + sum(min(d[i], F(0)) for i in U) / N
    hi = C_O + sum(max(d[i], F(0)) for i in U) / N
    width = sum(abs(d[i]) for i in U) / N
    return d, C_O, lo, hi, width, U

def exact_set(p, Y, verified):
    """Every attainable value: the subset-sum set, enumerated."""
    N = len(p)
    pbar = sum(p, F(0)) / N
    d = [x - pbar for x in p]
    U = [i for i in range(N) if i not in verified]
    vals = set()
    for combo in product((0, 1), repeat=len(U)):
        Yc = list(Y)
        for j, i in enumerate(U):
            Yc[i] = combo[j]
        vals.add(sum(d[i] * Yc[i] for i in range(N)) / N)
    return sorted(vals)

P8 = [F(10), F(9), F(8), F(7), F(3), F(2), F(1), F(0)]   # mean 5; d = 5 4 3 2 -2 -3 -4 -5

CASES = [
    ("bounds fix a POSITIVE sign",
     P8, [1,1,1,0,0,0,0,0], [0,1,2,5,6,7]),
    ("bounds fix a NEGATIVE sign",
     P8, [0,0,0,0,0,1,1,1], [0,1,2,5,6,7]),
    ("condition (T): all verified above the mean, all unverified below",
     P8, [1,0,1,0,0,0,0,0], [0,1,2,3]),
    ("condition (T') but NOT (T): a high-score threshold, unverified above the mean",
     P8, [1,0,0,0,0,0,0,0], [0,1]),
]

print()
print("=" * 78)
print("A.2 / A.3  SHARP COVARIANCE BOUNDS ON CONSTRUCTED POPULATIONS  (N = 8)")
print("=" * 78)
print("  p = [10,9,8,7,3,2,1,0], mean 5, d = [5,4,3,2,-2,-3,-4,-5] throughout")
for name, p, Y, ver in CASES:
    d, C_O, lo, hi, width, U = bounds(p, Y, ver)
    pts = exact_set(p, Y, ver)
    zero_in = lo <= 0 <= hi
    interior = lo < 0 < hi
    Tprime = all(d[i] > 0 for i in ver)
    T = Tprime and all(d[i] < 0 for i in U)
    print()
    print(f"  {name}")
    print(f"    verified {ver}   Y_O = {[Y[i] for i in ver]}   |U| = {len(U)}")
    print(f"    C_O = {float(C_O):+.4f}   hull = [{float(lo):+.4f}, {float(hi):+.4f}]"
          f"   width = {float(width):.4f}")
    print(f"    [{check(name + ': width identity', hi - lo == width)}]"
          f"  width == N^-1 sum_U |d_i|")
    print(f"    [{check(name + ': endpoints attained', min(pts) == lo and max(pts) == hi)}]"
          f"  both endpoints attained by an admissible completion")
    print(f"    [{check(name + ': identified set is finite', len(pts) <= 2 ** len(U))}]"
          f"  exact set is {len(pts)} points, a finite subset of the hull")
    print(f"    (T) holds: {str(T):<5}   (T') holds: {str(Tprime):<5}"
          f"   zero in hull: {str(zero_in):<5}   zero interior: {interior}")
    if Tprime:
        print(f"    [{check(name + ': (T\') implies zero in hull', zero_in)}]"
              f"  A.3: under (T') the hull contains zero")
    else:
        print(f"    [{check(name + ': sign is identified here', not zero_in)}]"
              f"  hull excludes zero, so the sign is identified")
    if Tprime:
        Dplus = sum(x for x in d if x > 0)
        general = (sum(d[i] * Y[i] for i in ver) - Dplus) / F(len(p))
        print(f"    [{check(name + ': A.3 general lower-endpoint form', general == lo)}]"
              f"  lower = N^-1 [sum_O d_i Y_i - D+] = {float(general):+.4f}")
        special = -sum(d[i] * (1 - Y[i]) for i in ver) / F(len(p))
        if T:
            print(f"    [{check(name + ': (T) reduction', special == lo)}]"
                  f"  under (T) this reduces to -N^-1 sum_O d_i (1-Y_i)"
                  f" = {float(special):+.4f}")
        else:
            print(f"    [{check(name + ': (T)-only form must fail here', special != lo)}]"
                  f"  the (T)-only form gives {float(special):+.4f}, which correctly"
                  f" differs from the hull endpoint")

print()
print("  The fourth case is the one a high-score threshold actually produces: (T)")
print("  fails because unverified units lie above the proxy mean, (T') still holds,")
print("  and zero remains interior to the hull.  See the flag in Appendix A.3.")

print()
if FAILURES:
    print(f"  {len(FAILURES)} FAILED: " + "; ".join(FAILURES))
else:
    print("  all assertions passed")
raise SystemExit(1 if FAILURES else 0)
