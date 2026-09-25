"""Implementation gate, PREREG section 1.3 as amended by PREREG-AMENDMENT-1.

Two kinds of check, per amendment section 2.3.
  EXACT     flag counts.  Under top-qN flagging these have zero variance: every method
            flags exactly n_flag(q) units.  Checked as equality, not z-scored.
  Z-SCORED  everything else, against closed forms that are now APPROXIMATIONS rather
            than identities, at the tolerance of 4.0 declared in the original PREREG
            and held unchanged on measurement (amendment section 2.2).

What this gate does NOT test is listed in PREREG-AMENDMENT-1 section 2.7 and in
Appendix C.  It shows that the checked moments agree with the declared law.  It does
not show that the code implements the process.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sim.dgp import (N_POP, K, PI_GRID, Q_GRID, RHO_GRID, ENTROPY, n_flag,
                     closed_forms, union_coverage, realize_population, pair_table)

Z_TOL = 4.0                      # declared tolerance in Monte Carlo standard errors
PAIRS = [(0, 19), (9, 10)]       # extreme pair and adjacent pair, per PREREG 1.3

def z(obs, exp, se):
    return 0.0 if se <= 0 else (obs - exp) / se

def run():
    rows, worst = [], 0.0
    ss = np.random.SeedSequence(ENTROPY)
    combos = [(pi, q, rho) for pi in PI_GRID for q in Q_GRID for rho in RHO_GRID]
    for idx, (pi, q, rho) in enumerate(combos):
        rng = np.random.default_rng(ss.spawn(1000)[idx])
        cf = closed_forms(pi, q, rho)
        pop = realize_population(pi, q, rho, rng)
        pt = pair_table(pop)
        npos = max(int(pop["Y"].sum()), 1)
        checks = []

        nf = n_flag(q)
        exact = []                                           # zero-variance identities
        for k in range(K):                                   # flag count, EXACT
            obs = int(round(pt["flag_rate"][k] * N_POP))
            exact.append((f"flag_count[{k}]", obs == nf))
        for k in range(K):                                   # sensitivity
            e = cf["s"][k]; se = np.sqrt(max(e * (1 - e), 1e-12) / npos)
            checks.append((f"sensitivity[{k}]", z(pt["s"][k], e, se)))
        for k in range(K):                                   # true-positive mass
            e = cf["a"][k]; se = np.sqrt(max(e * (1 - e), 1e-12) / N_POP)
            checks.append((f"tp_mass[{k}]", z(pt["a"][k], e, se)))
        se = np.sqrt(pi * (1 - pi) / N_POP)                  # prevalence
        checks.append(("prevalence", z(pt["prevalence"], pi, se)))
        C = union_coverage(pi, q, rho)                       # union coverage
        se = np.sqrt(C * (1 - C) / N_POP)
        checks.append(("union_coverage", z(pt["union_cov"], C, se)))

        for (j, k) in PAIRS:                                 # cell structure
            e = cf["pair"](j, k)
            for lbl, obs, exp in (
                (f"N10[{j},{k}]", pt["N10"][j, k], e["N10"]),
                (f"N01[{j},{k}]", pt["N10"][k, j], e["N01"]),
                (f"M10[{j},{k}]", pt["M10"][j, k], e["M10"]),
                (f"M01[{j},{k}]", pt["M10"][k, j], e["M01"]),
            ):
                p = exp / N_POP
                se = np.sqrt(max(N_POP * p * (1 - p), 1e-12))
                checks.append((lbl, z(obs, exp, se)))

        mx = max(abs(v) for _, v in checks)
        arg = max(checks, key=lambda t: abs(t[1]))[0]
        worst = max(worst, mx)
        n_exact_fail = sum(1 for _, ok in exact if not ok)
        ar = pt["a"]
        inversions = int(sum(1 for k in range(K - 1) if ar[k] >= ar[k + 1]))
        rows.append(dict(pi=pi, q=q, rho=rho, n_checks=len(checks),
                         n_exact=len(exact), n_exact_fail=n_exact_fail,
                         flag_count=int(round(pt["flag_rate"][0] * N_POP)),
                         ladder_inversions=inversions,
                         ladder_strict=bool(inversions == 0),
                         argmax_method=int(np.argmax(ar)) + 1,
                         max_abs_z=round(mx, 3), at=arg,
                         union_cov_closed=round(C, 5),
                         union_cov_realized=round(float(pt["union_cov"]), 5),
                         gamma_realized=round(1 - float(pt["union_cov"]), 5)))
    return rows, worst

if __name__ == "__main__":
    rows, worst = run()
    nz = sum(r["n_checks"] for r in rows)
    ne = sum(r["n_exact"] for r in rows)
    nef = sum(r["n_exact_fail"] for r in rows)
    print("=" * 96)
    print("IMPLEMENTATION GATE   PREREG 1.3 as amended by PREREG-AMENDMENT-1")
    print("=" * 96)
    print(f"  {'pi':>6} {'q':>6} {'rho':>4} {'flags':>6} {'exact':>6} {'z-chk':>6} "
          f"{'max|z|':>7} {'at':>16} {'gamma':>7} {'ladder':>7} {'best':>5}")
    for r in rows:
        print(f"  {r['pi']:>6} {r['q']:>6} {r['rho']:>4} {r['flag_count']:>6} "
              f"{r['n_exact']-r['n_exact_fail']:>3}/{r['n_exact']:<2} {r['n_checks']:>6} "
              f"{r['max_abs_z']:>7.3f} {r['at']:>16} {r['gamma_realized']:>7.4f} "
              f"{'strict' if r['ladder_strict'] else str(r['ladder_inversions'])+' inv':>7} "
              f"{r['argmax_method']:>5}")
    print(f"\n  exact identities   : {ne-nef}/{ne} pass   (flag counts, zero variance)")
    print(f"  z-scored checks    : {nz} at tolerance |z| <= {Z_TOL}")
    print(f"  worst |z|          : {worst:.3f}")
    ok = (nef == 0) and (worst <= Z_TOL)
    print(f"\n  GATE: {'PASS' if ok else 'FAIL'}")
    print("\n  What this gate does not test: PREREG-AMENDMENT-1 section 2.7.")
    json.dump(rows, open(os.path.join(ROOT, "sim", "gate.json"), "w"), indent=1)
    raise SystemExit(0 if ok else 1)
