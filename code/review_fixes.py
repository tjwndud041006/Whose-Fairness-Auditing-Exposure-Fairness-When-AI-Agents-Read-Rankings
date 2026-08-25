"""Review-response computations (items 1-4, all CPU LP):
(1) Leave-one-out sensitivity of the main-8 agent set on ESCI: drop each agent
    curve -> U*, premium, lambda_break of the reduced set.
(2) Human-curve ambiguity: V_H = {DCG, RBP-0.8, RBP-0.9} -> robust floor with
    (a) human set only (no agents), (b) human set + main-8 agents.
(3) Existence-audit dilemma on canonical P_H: acquittal rates when the audit
    family D is (i) geometric decay (paper's 27/200), (ii) all monotone
    nonincreasing curves, (iii) the full simplex (any curve) -- per-query LP.
(4) lambda_break workload band from the saved main_frontier.json rows
    (first a with umax(a) >= U_R per workload; no re-solve needed).
Run: python review_fixes.py
"""
import json
import numpy as np
from scipy.optimize import linprog
from loaders import LOADERS, K
from policy_structure4 import build_canonical
from curveset import load_set

data = LOADERS["ESCI"]()
ctx = build_canonical(data, ntop=15)
vH = ctx["vH"]; agents = ctx["agents"]
names = list(load_set(K, "main").keys())
NQ = len(data)
top = np.array(ctx["top"]); tgt = ctx["tgt"]

def U_of(Ps, v):
    E = np.zeros(len(ctx["brands"]))
    for q, (m, bs) in enumerate(data):
        e = Ps[q] @ v
        np.add.at(E, [ctx["bidx"][bb] for bb in bs], e)
    return float(np.max(np.abs(E[top]/(NQ*v.sum()*tgt[top])-1)))

def lam_break(P_H, P_R, ags):
    prev, lam = None, None
    for a in np.linspace(0, 0.8, 81):
        if a == 0:
            uh, ur = U_of(P_H, vH), U_of(P_R, vH)
        else:
            uh = max(U_of(P_H, (1-a)*vH + a*v) for v in ags)
            ur = max(U_of(P_R, (1-a)*vH + a*v) for v in ags)
        if prev is not None and prev < 0 and uh-ur >= 0 and lam is None: lam = float(a)
        prev = uh-ur
    return lam

out = {}
P_H, U_H = ctx["solve"]([vH])
P_R, U_R = ctx["solve"]([vH]+agents)
print(f"baseline: U_H={U_H:.4f} U_R={U_R:.4f} prem={U_R-U_H:.4f}", flush=True)

# ---------- (1) leave-one-out ----------
loo = {}
for j, nm in enumerate(names):
    ags = [v for i, v in enumerate(agents) if i != j]
    P_j, U_j = ctx["solve"]([vH]+ags)
    lam_j = lam_break(P_H, P_j, ags)
    loo[nm] = {"U": float(U_j), "premium": float(U_j-U_H), "lam": lam_j}
    print(f"LOO -{nm:26s} U*={U_j:.4f} prem={U_j-U_H:.4f} lam={lam_j}", flush=True)
out["loo"] = loo

# ---------- (2) human-curve ambiguity ----------
rbp8 = 0.8**np.arange(K); rbp8 /= rbp8.sum()
rbp9 = 0.9**np.arange(K); rbp9 /= rbp9.sum()
hset = [vH, rbp8, rbp9]
P_hs, U_hs = ctx["solve"](hset)
P_all, U_all = ctx["solve"](hset+agents)
out["human_set"] = {"U_Hset": float(U_hs), "premium_Hset": float(U_hs-U_H),
                    "U_combined": float(U_all), "premium_combined": float(U_all-U_H)}
print(f"human-set only: U*={U_hs:.4f} (prem {U_hs-U_H:+.4f}); combined: U*={U_all:.4f} (prem {U_all-U_H:+.4f})", flush=True)

# ---------- (3) existence-audit dilemma on canonical P_H ----------
def geo(r):
    v = r**np.arange(1, K+1); return v/v.sum()
rs = np.linspace(0.30, 1.0, 36)
DELTAS = [0.10, 0.15, 0.20, 0.25]
vs_all = [vH]+agents

def min_u_simplex(Pq, mm, monotone):
    # min t s.t. |(Pq v)_i / mm_i - 1| <= t, v in simplex (optionally nonincreasing)
    Kv = Pq.shape[1]
    ncols = Kv + 1
    A_ub, b_ub = [], []
    for i in range(Pq.shape[0]):
        row = np.zeros(ncols); row[:Kv] = Pq[i]/mm[i]; row[Kv] = -1
        A_ub.append(row); b_ub.append(1.0)
        row2 = np.zeros(ncols); row2[:Kv] = -Pq[i]/mm[i]; row2[Kv] = -1
        A_ub.append(row2); b_ub.append(-1.0)
    if monotone:
        for k in range(Kv-1):
            row = np.zeros(ncols); row[k] = -1; row[k+1] = 1
            A_ub.append(row); b_ub.append(0.0)
    A_eq = np.zeros((1, ncols)); A_eq[0, :Kv] = 1
    c = np.zeros(ncols); c[Kv] = 1
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, None)]*Kv + [(0, None)], method="highs")
    return float(res.x[Kv]) if res.status == 0 else np.inf

fam = {f: {d: [0, 0] for d in DELTAS} for f in ["geometric", "monotone", "simplex"]}
for q, (m, bs) in enumerate(data):
    mm = np.asarray(m, float); mm = mm/mm.sum()
    Pq = P_H[q]
    def Uq(v): return float(np.max(np.abs((Pq @ v)/(v.sum()*mm)-1)))
    u = {"geometric": min(Uq(geo(r)) for r in rs),
         "monotone": min_u_simplex(Pq, mm, True),
         "simplex": min_u_simplex(Pq, mm, False)}
    u_mx = max(Uq(v) for v in vs_all)
    for f in fam:
        for d in DELTAS:
            if u[f] <= d:
                fam[f][d][0] += 1
                if u_mx > d: fam[f][d][1] += 1
out["audit_dilemma"] = {f: {str(d): {"acquitted": fam[f][d][0], "acquitted_but_unfair": fam[f][d][1]}
                            for d in DELTAS} for f in fam}
for f in fam:
    print(f"audit family={f}: " + " | ".join(
        f"d={d}: acq {fam[f][d][0]}/{NQ} (unfair {fam[f][d][1]})" for d in DELTAS), flush=True)

# ---------- (4) lambda_break band from saved frontier rows ----------
fr = json.load(open("main_frontier.json"))
AGRID = np.linspace(0, 1, 51)
lams = []
for r in fr["rows"]:
    u = np.array(r["umax"]); UR = r["U_R"]
    idx = np.where((u >= UR) & (AGRID > 0))[0]
    lams.append(float(AGRID[idx[0]]) if len(idx) else None)
valid = [l for l in lams if l is not None]
out["lam_band"] = {"per_workload": lams, "mean": float(np.mean(valid)),
                   "p5": float(np.percentile(valid, 5)), "p95": float(np.percentile(valid, 95)),
                   "min": float(min(valid)), "max": float(max(valid)), "n_none": lams.count(None)}
print(f"lambda_break band over 20 workloads: mean {np.mean(valid):.3f} "
      f"[p5 {np.percentile(valid,5):.2f}, p95 {np.percentile(valid,95):.2f}] "
      f"range [{min(valid):.2f},{max(valid):.2f}] none={lams.count(None)}", flush=True)

json.dump(out, open("review_fixes.json", "w"), indent=1)
print("saved review_fixes.json")
