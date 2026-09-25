"""Gate first, then the five falsification criteria in preregistered order, then headlines.
PREREG section 8 as amended by PREREG-AMENDMENT-1."""
import json, numpy as np
from collections import defaultdict
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R=json.load(open(os.path.join(ROOT, 'sim', 'results.json')))
C=R["cells"]; REPS=R["reps"]; ALPHA=0.05
POL=("random","greedy")
ADVERSE=("adv_reuse","adv_differr","adv_drift","adv_poorchal")
def g(c,p,pol,k): return c["res"][f"{p}|{pol}"][k]

print("="*84); print("REALIZED GAMMA PER CELL  (deterministic floor under top-qN flagging)"); print("="*84)
print(f"  bound: gamma >= 1 - Kq  =  0.980 at q=0.001,  0.900 at q=0.005  (exact, not probabilistic)\n")
seen={}
for c in C: seen.setdefault((c["pi"],c["q"],c["rho"]),c["meta"]["gamma"])
print(f"  {'pi':>6}{'q':>7}{'rho':>5}  {'gamma':>7}  {'1-Kq':>6}  {'slack':>7}  {'ladder':>9}  {'best':>5}")
for (pi,q,rho),gm in sorted(seen.items()):
    cc=[c for c in C if (c["pi"],c["q"],c["rho"])==(pi,q,rho)][0]["meta"]
    bd=1-20*q
    print(f"  {pi:>6}{q:>7}{rho:>5}  {gm:>7.4f}  {bd:>6.3f}  {gm-bd:>+7.4f}  "
          f"{('strict' if cc['ladder_strict'] else str(cc['ladder_inversions'])+' inv'):>9}  {cc['argmax_method']:>5}")
gam=[c["meta"]["gamma"] for c in C]
print(f"\n  min {min(gam):.4f}   max {max(gam):.4f}   all >= 1-Kq: "
      f"{all(c['meta']['gamma'] >= 1-20*c['q']-1e-12 for c in C)}")
fc=all(c["meta"]["flag_counts_equal"] for c in C)
print(f"  every method flags exactly floor(qN) units, all K equal, in every cell: {fc}")

print("\n"+"="*84); print("FALSIFICATION CRITERIA  (PREREG section 8, preregistered order)"); print("="*84)
se=np.sqrt(ALPHA*(1-ALPHA)/REPS)
print(f"\n8.1  LEVEL   threshold alpha + 2SE = {ALPHA+2*se:.4f}")
w=max((g(c,"controlled",p,"p_any_false"),c["cell"],p) for c in C for p in POL)
viol=[1 for c in C for p in POL if g(c,"controlled",p,"p_any_false")>ALPHA+2*se]
print(f"     worst controlled P(any false) = {w[0]:.4f} (cell {w[1]}, {w[2]});  cells exceeding: {len(viol)}")
nz=sum(1 for c in C for p in POL if g(c,"naive",p,"p_any_false")>0)
nzc=sum(1 for c in C for p in POL if g(c,"controlled",p,"p_any_false")>0)
zero_naive=sum(1 for c in C for p in POL if g(c,"naive",p,"p_any_false")==0)
over=sum(1 for c in C for p in POL if g(c,"naive",p,"p_any_false")>ALPHA)
print(f"     -> {'FALSIFIED' if viol else 'NOT FALSIFIED'}")
print(f"     discriminating power: naive exceeds alpha in {over}/96; naive exactly zero in {zero_naive}/96")
print(f"     any false upgrade at all: naive {nz}/96, controlled {nzc}/96")

print(f"\n8.2  VACUITY   trigger: controlled P(zero upgrades) > 0.50")
pz=[g(c,"controlled",p,"p_zero_up") for c in C for p in POL]
bad=sum(1 for x in pz if x>0.50)
print(f"     min {min(pz):.4f}  median {np.median(pz):.4f}  max {max(pz):.4f};  above 0.50: {bad}/96")
for lab,key in [("pi",lambda c:c["pi"]),("q",lambda c:c["q"]),("B",lambda c:c["budget"])]:
    d=defaultdict(list)
    for c in C:
        for p in POL: d[key(c)].append(g(c,"controlled",p,"p_zero_up"))
    print(f"     by {lab:>2}: "+"   ".join(f"{k} -> {np.mean(v):.3f}" for k,v in sorted(d.items())))

print(f"\n8.3  COST   trigger: consumption > 10x budget")
r=[g(c,"controlled",p,"mean_used")/(c["budget"]*20) for c in C for p in POL]
pt=[g(c,"controlled",p,"p_top_quartile") for c in C for p in POL]
print(f"     consumed/supplied: median {np.median(r):.3f} max {max(r):.3f}  -> {'TOO COSTLY' if max(r)>10 else 'NOT TRIGGERED'}")
print(f"     P(top quartile): median {np.median(pt):.3f} min {min(pt):.3f}")

print(f"\n8.4  ADVERSE CONDITIONS")
base={k:np.mean([g(c,"controlled",p,k) for c in C for p in POL]) for k in
      ("p_any_false","mean_ups","p_zero_up","mean_rank")}
print(f"     {'condition':>14} {'P(any false)':>13} {'ups':>7} {'P(zero)':>8} {'rank':>7}   verdict")
print(f"     {'controlled':>14} {base['p_any_false']:>13.4f} {base['mean_ups']:>7.2f} "
      f"{base['p_zero_up']:>8.3f} {base['mean_rank']:>7.2f}   --")
noeff=[]
for adv in ADVERSE:
    v={k:np.mean([g(c,adv,p,k) for c in C for p in POL]) for k in base}
    deg=(v["p_any_false"]>base["p_any_false"]+2*se) or (v["mean_rank"]>base["mean_rank"]+0.5) \
        or (v["mean_ups"]<base["mean_ups"]-0.25)
    if not deg: noeff.append(adv)
    print(f"     {adv:>14} {v['p_any_false']:>13.4f} {v['mean_ups']:>7.2f} "
          f"{v['p_zero_up']:>8.3f} {v['mean_rank']:>7.2f}   {'degrades' if deg else '*** NO EFFECT ***'}")
print(f"     -> {'ADVERSARIAL' if not noeff else 'NOT ADVERSARIAL for: '+', '.join(noeff)}")

print(f"\n8.5  UNION COVERAGE   exclusion threshold gamma < 0.50")
print(f"     realized gamma min {min(gam):.4f} max {max(gam):.4f};  cells excluded: "
      f"{sum(1 for x in gam if x<0.50)}")

print("\n"+"="*84); print("HEADLINE QUANTITIES"); print("="*84)
print(f"  {'procedure':>14} {'P(false)':>9} {'E[false]':>9} {'ups':>6} {'rank':>7} "
      f"{'P(top25)':>9} {'verifs':>7} {'|V|/N <=':>9}")
for k in ("naive","controlled","oracle")+ADVERSE:
    v={x:np.mean([g(c,k,p,x) for c in C for p in POL]) for x in
       ("p_any_false","e_false","mean_ups","mean_rank","p_top_quartile","mean_used","frac_verified_upper")}
    print(f"  {k:>14} {v['p_any_false']:>9.4f} {v['e_false']:>9.4f} {v['mean_ups']:>6.2f} "
          f"{v['mean_rank']:>7.2f} {v['p_top_quartile']:>9.3f} {v['mean_used']:>7.0f} "
          f"{v['frac_verified_upper']:>9.5f}")
print(f"\n  ABSOLUTE UNCERTAINTY -- what the run can state")
print(f"    prevalence identified-set diameter FLOOR (Cor. 12): min {min(gam):.4f} over 48 cells")
print(f"    adjudicated share, upper bound: {np.mean([g(c,'controlled',p,'frac_verified_upper') for c in C for p in POL]):.5f}")
ow=np.mean([g(c,"controlled",p,"oracle_terminal_width") for c in C for p in POL])
iw=np.mean([c["meta"]["initial_width_SUPERSEDED"] for c in C])
print(f"    ORACLE terminal sensitivity width (diagnostic only): {ow:.6f}")
print(f"    initial-incumbent width (the superseded run's defective quantity): {iw:.6f}")
