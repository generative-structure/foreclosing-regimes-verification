"""Sequential experiment, PREREG sections 3-7.

Implementation choices that PREREG leaves open are collected in CHOICES below and
reported with the results.  Nothing is tuned to any outcome.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sim.dgp import (N_POP, K, MU, T_ROUND, ALPHA, ENTROPY, PI_GRID, Q_GRID,
                     RHO_GRID, DELTA_FRAC, BUDGET, I1,
                     realize_population, pair_table, thresholds)
from sim.exact import critical_value

REPS = 2000

# Recovered stream indices of the reported run (see the comment at the point of use).
STREAM_INDEX = {
    ("oracle", "greedy"): 0,
    ("adv_differr", "random"): 7,
    ("adv_reuse", "random"): 8,
    ("adv_poorchal", "random"): 11,
    ("adv_differr", "greedy"): 13,
    ("adv_reuse", "greedy"): 14,
    ("adv_poorchal", "greedy"): 17,
    ("adv_drift", "random"): 23,
    ("adv_drift", "greedy"): 29,
    ("controlled", "random"): 55,
    ("oracle", "random"): 55,
    ("naive", "random"): 56,
    ("controlled", "greedy"): 61,
    ("naive", "greedy"): 62,
}

CHOICES = {
  "differential_error": "cell C: (alpha,beta)=(0.95,0.95), J=0.90;  cell I: (0.75,0.95), J=0.70",
  "drift": "4 populations per parameter combination with progressively shuffled mu ladders; round t uses population t mod 4",
  "reuse": "one verification pool of size = budget drawn at round 1 from the flagged union and reused unchanged at every round, treated as a fresh sample",
  "poor_challenger": "challengers drawn uniformly from the weakest half of the ladder",
  "allocation": "minimax m_h proportional to q_h; matched flag volume gives N_C = N_I so m_C = m_I = budget/2",
}

def build_populations(pi, q, rho, seed):
    """Base population plus three drifted ones (shuffled mu ladders)."""
    pops = []
    for d in range(4):
        rng = np.random.default_rng(seed.spawn(1)[0]) if d == 0 else np.random.default_rng(seed.spawn(4)[d])
        if d == 0:
            pop = realize_population(pi, q, rho, rng)
        else:
            perm = np.random.default_rng(ENTROPY + 7919 * d).permutation(K)
            import sim.dgp as D
            base = D.MU.copy(); D.MU = base[perm]
            pop = realize_population(pi, q, rho, rng); D.MU = base
        pt = pair_table(pop)
        pops.append(dict(pop=pop, pt=pt))
    return pops

def run_cell(pi, q, rho, dfrac, budget, seed, verbose=False):
    pops = build_populations(pi, q, rho, seed)
    base = pops[0]; pt = base["pt"]; a = pt["a"]; N10 = pt["N10"]; M10 = pt["M10"]
    m = budget // 2
    delta = dfrac * a[I1]; d = int(np.floor(N_POP * delta))
    alpha_t = {"naive": ALPHA, "controlled": ALPHA / T_ROUND}
    crit = {}
    def C(N, al):
        key = (N, m, d, al)
        if key not in crit:
            crit[key] = critical_value(N, min(m, N), d, al)
        return crit[key]

    F = base["pop"]["F"]; Y = base["pop"]["Y"]
    flagged_union = np.flatnonzero(np.any(F, axis=0))
    res = {}
    procs = ["naive", "controlled", "oracle", "adv_reuse", "adv_differr", "adv_drift", "adv_poorchal"]
    for policy in ("random", "greedy"):
        for proc in procs:
            # Stream assignment. The original line read
            #     seed.spawn(64)[hash((proc, policy)) % 64]
            # and selected each arm's stream through Python's string hashing, which is
            # randomized per process, so the assignment varied from run to run. The map
            # below is the assignment the reported run actually used, recovered from the
            # frozen sim/results.json by trial against cell 0 and confirmed by regenerating
            # all 48 cells byte-for-byte. It is explicit so that the reported run is exactly
            # reproducible. Note the collision: controlled|random and oracle|random drew from
            # the same stream (index 55).
            rng = np.random.default_rng(seed.spawn(64)[STREAM_INDEX[(proc, policy)]])
            false_up = np.zeros(REPS, int); n_up = np.zeros(REPS, int)
            null_true = np.zeros(REPS, int); null_rej = np.zeros(REPS, int)
            gain = np.zeros(REPS); final = np.zeros(REPS, int); used = np.zeros(REPS, int)
            for r in range(REPS):
                I = I1; unused = [k for k in range(K) if k != I1]
                if proc == "adv_reuse":
                    pool = rng.choice(flagged_union, size=min(budget, len(flagged_union)), replace=False)
                    poolF = F[:, pool]; poolY = Y[pool]
                for t in range(T_ROUND):
                    if not unused: break
                    if proc == "adv_poorchal":
                        cand = [k for k in unused if k < K // 2] or unused
                    else:
                        cand = unused
                    Ch = int(rng.choice(cand)) if policy == "random" else int(max(cand, key=lambda k: MU[k]))
                    tab = pops[t % 4]["pt"] if proc == "adv_drift" else pt
                    aa = tab["a"]
                    NC, NI = tab["N10"][Ch, I], tab["N10"][I, Ch]
                    MC, MI = tab["M10"][Ch, I], tab["M10"][I, Ch]
                    true_gain = aa[Ch] - aa[I]
                    if true_gain <= delta: null_true[r] += 1
                    rej = False
                    if proc == "oracle":
                        rej = true_gain > delta
                    elif NC > 0 and NI > 0:
                        mm = min(m, NC, NI)
                        if proc == "adv_reuse":
                            selC = poolF[Ch] & ~poolF[I]; selI = poolF[I] & ~poolF[Ch]
                            tC, tI = int((selC & poolY).sum()), int((selI & poolY).sum())
                            mm2 = min(int(selC.sum()), int(selI.sum()))
                            c = C(max(NC, 1), alpha_t["controlled"])
                            rej = (c is not None) and (tC - tI >= c) and mm2 > 0
                            used[r] += int(selC.sum() + selI.sum()) if t == 0 else 0
                        else:
                            tC = int(rng.hypergeometric(MC, NC - MC, mm))
                            tI = int(rng.hypergeometric(MI, NI - MI, mm))
                            if proc == "adv_differr":
                                aC, bC, aI, bI = 0.95, 0.95, 0.75, 0.95
                                tC = int(rng.binomial(tC, aC) + rng.binomial(mm - tC, 1 - bC))
                                tI = int(rng.binomial(tI, aI) + rng.binomial(mm - tI, 1 - bI))
                            al = alpha_t["naive"] if proc == "naive" else alpha_t["controlled"]
                            c = C(NC, al)
                            rej = (c is not None) and (tC - tI >= c)
                            used[r] += 2 * mm
                    if rej:
                        n_up[r] += 1
                        if true_gain <= delta: null_rej[r] += 1
                        if true_gain <= delta: false_up[r] += 1
                        gain[r] += true_gain
                        I = Ch
                    unused.remove(Ch)
                final[r] = I
            res[(proc, policy)] = dict(
                p_any_false=float((false_up > 0).mean()),
                e_false=float(false_up.mean()),
                p_zero_up=float((n_up == 0).mean()),
                mean_ups=float(n_up.mean()),
                mean_gain=float(gain.mean()),
                mean_rank=float(np.mean([int((a > a[f]).sum()) + 1 for f in final])),
                p_top_quartile=float(np.mean([int((a > a[f]).sum()) + 1 <= K // 4 for f in final])),
                mean_used=float(used.mean()),
                mean_null_true_rounds=float(null_true.mean()),
                null_rej_rate=float(null_rej.sum() / max(null_true.sum(), 1)),
                # PREREG-AMENDMENT-1 section 2.4, quantity (2): |V_T|/N is not tracked
                # exactly (a unit may fall in several rounds' cells), so this is an
                # UPPER bound on the adjudicated share -- total adjudications over N.
                frac_verified_upper=float(used.mean() / N_POP),
                # quantity (3): W(a) at the REALIZED TERMINAL incumbent.  An ORACLE
                # diagnostic, not an identified interval: under the observation regime
                # of amendment section 2.5 the system never identifies a_{I_t}.
                oracle_terminal_width=float(
                    np.mean([1 - a[f] / (a[f] + 1 - q) for f in final])),
            )
    gam = 1 - float(pt["union_cov"])
    inv = int(sum(1 for k in range(K - 1) if a[k] >= a[k + 1]))
    return res, dict(gamma=gam, delta=float(delta), d=d, m=m,
                     a_I1=float(a[I1]), a_best=float(a.max()),
                     # PREREG-AMENDMENT-1 section 2.6: realized-ladder reporting
                     flag_count=int(pt["flag_rate"][0] * N_POP),
                     flag_counts_equal=bool(len(set(
                         (pt["flag_rate"] * N_POP).round().astype(int).tolist())) == 1),
                     ladder_inversions=inv,
                     ladder_strict=bool(inv == 0),
                     argmax_method=int(np.argmax(a)) + 1,
                     # the floor of Corollary 12.  A FLOOR, not a measurement: the
                     # exact diameter of Proposition 11 needs V_T fully labelled and
                     # the run never labels a cell (amendment section 2.5).
                     prevalence_diam_floor=gam,
                     # width at the INITIAL incumbent, retained only so the superseded
                     # run's defective quantity can be reproduced and compared.
                     initial_width_SUPERSEDED=float(1 - a[I1] / (a[I1] + 1 - q)))

if __name__ == "__main__":
    t0 = time.time(); ss = np.random.SeedSequence(ENTROPY)
    cells, idx = [], 0
    for pi in PI_GRID:
        for q in Q_GRID:
            for rho in RHO_GRID:
                for dfrac in DELTA_FRAC:
                    for b in BUDGET:
                        cells.append((idx, pi, q, rho, dfrac, b)); idx += 1
    out = []
    kids = ss.spawn(len(cells))
    for (i, pi, q, rho, dfrac, b) in cells:
        res, meta = run_cell(pi, q, rho, dfrac, b, kids[i])
        out.append(dict(cell=i, pi=pi, q=q, rho=rho, dfrac=dfrac, budget=b,
                        meta=meta,
                        res={f"{p}|{pol}": v for (p, pol), v in res.items()}))
        print(f"  cell {i+1:>2}/48  pi={pi} q={q} rho={rho} d%={dfrac} B={b} "
              f"gamma={meta['gamma']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(dict(cells=out, choices=CHOICES, reps=REPS, entropy=ENTROPY),
              open(os.path.join(ROOT, 'sim', 'results.json'), 'w'), indent=1)
    print(f"\n  done in {time.time()-t0:.0f}s")
