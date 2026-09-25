#!/usr/bin/env python3
"""The verification-error reversal, and its cancellation under matched alert
volume (the reference-standard identity).

Verification only. Exact rational arithmetic on the worked instance stated in
the body. No free parameters: alpha, beta, q and r are the values the body
prints, and the point of the exercise is that the identity holds for them
exactly rather than approximately.
"""
from fractions import Fraction as F
import sympy as sp

# ---- the identity, symbolically -------------------------------------------
q10, q01, r10, r01, al, be = sp.symbols('q10 q01 r10 r01 alpha beta', positive=True)
J = al + be - 1
z10, z01 = (1 - be) + J * r10, (1 - be) + J * r01
Delta = q10 * r10 - q01 * r01
Dtilde = q10 * z10 - q01 * z01
identity_holds = sp.simplify(sp.expand(Dtilde - ((1 - be) * (q10 - q01) + J * Delta))) == 0

print("=" * 76)
print("REVERSAL IDENTITY   Dtilde = (1-beta)(q10-q01) + J*Delta,   J = alpha+beta-1")
print("=" * 76)
print(f"  [{'PASS' if identity_holds else 'FAIL'}]  identity holds symbolically "
      "(remainder simplifies to 0)")

# ---- the worked instance, in exact rationals ------------------------------
def instance(q10v, q01v, r10v, r01v, a, b):
    Jv = a + b - 1
    z1, z0 = (1 - b) + Jv * r10v, (1 - b) + Jv * r01v
    D = q10v * r10v - q01v * r01v
    Dt = q10v * z1 - q01v * z0
    offset = (1 - b) * (q10v - q01v)
    return z1, z0, D, Dt, offset, Jv

print()
print("  worked instance from the body, exact rational arithmetic")
print(f"  {'q10':>6} {'q01':>6} {'r10':>5} {'r01':>5} {'z10':>7} {'z01':>7}"
      f" {'Delta':>8} {'Dtilde':>9} {'offset':>8}  sign")
rows = [
    ("unequal cells", F(1,10), F(1,5), F(9,10), F(2,5)),
    ("matched alert volume, q10 = q01", F(3,20), F(3,20), F(9,10), F(2,5)),
]
for name, q1, q0, r1, r0 in rows:
    z1, z0, D, Dt, off, Jv = instance(q1, q0, r1, r0, F(4,5), F(4,5))
    flip = "REVERSED" if (D > 0) != (Dt > 0) else "preserved"
    print(f"  {str(q1):>6} {str(q0):>6} {str(r1):>5} {str(r0):>5} {str(z1):>7} {str(z0):>7}"
          f" {str(D):>8} {str(Dt):>9} {str(off):>8}  {flip}")
    print(f"      {name}")

# ---- explicit confirmation of the two body claims -------------------------
z1, z0, D, Dt, off, Jv = instance(F(1,10), F(1,5), F(9,10), F(2,5), F(4,5), F(4,5))
checks = [
    ("z10 = 0.74", z1 == F(37, 50)),
    ("z01 = 0.44", z0 == F(11, 25)),
    ("true Delta = +0.01", D == F(1, 100)),
    ("observed Dtilde = -0.014", Dt == F(-7, 500)),
    ("sign reverses under error identical in both arms", (D > 0) and (Dt < 0)),
]
q1 = q0 = F(3, 20)
_, _, D2, Dt2, off2, _ = instance(q1, q0, F(9,10), F(2,5), F(4,5), F(4,5))
checks += [
    ("offset (1-beta)(q10-q01) is identically 0 when q10 = q01", off2 == 0),
    ("Dtilde = J*Delta exactly under matched volume", Dt2 == Jv * D2),
    ("sign preserved under matched volume", (D2 > 0) == (Dt2 > 0)),
]
print()
for label, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}]  {label}")
print()
print("  Matched alert volume forces N10 = N01, hence q10 = q01, hence offset 0.")
print("  Nothing is assumed about alpha or beta beyond J > 0.")
raise SystemExit(0 if identity_holds and all(o for _, o in checks) else 1)
