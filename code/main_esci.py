"""Round-9 rebase, step 2: the ESCI deep-dive under MAIN set + canonical policies.
(A) Table 1: Umax(a) at a in {0,.25,1} + worst-type (argmax vertex) for
    relevance-sort / human-cal / average-curve / merit-robust / robust / oracle.
(B) Existence-vs-worst-case audit (geometric-decay family) on canonical P_H.
(C) Saturation: robust floor + lambda_break under MAIN(9) vs FULL(15) vertices;
    leave-one-out of each EXT curve.
Run: python main_esci.py
"""
import json
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from loaders import esci_data, K
from policy_structure4 import build_canonical
from curveset import load_set

rng = np.random.default_rng(5)
data = esci_data()
NQ = len(data)
ctx = build_canonical(data, ntop=15)
vH = ctx["vH"]; agents = ctx["agents"]
anames = list(load_set(K, "main").keys())
ext = load_set(K, "ext")
brands, bidx, top, tgt = ctx["brands"], ctx["bidx"], np.array(ctx["top"]), ctx["tgt"]
vs_all = [vH]+agents; vnames = ["human"]+anames

def U_of(Ps, v):
    E = np.zeros(len(brands))
    for q, (m, bs) in enumerate(data):
        e = Ps[q] @ v
        np.add.at(E, [bidx[b] for b in bs], e)
    return float(np.max(np.abs(E[top]/(NQ*v.sum()*tgt[top])-1)))

def stats(Ps):
    per = {vn: U_of(Ps, v) for vn, v in zip(vnames, vs_all)}
    wv = max(per.values()); am = max(per, key=per.get)
    def at(a):
        if a == 0: return U_of(Ps, vH)
        return max(U_of(Ps, (1-a)*vH + a*v) for v in agents)
    return {"a0": at(0), "a25": at(0.25), "a1": at(1.0), "worst": wv, "argmax": am}

print("== policies ==", flush=True)
P_H, U_H = ctx["solve"]([vH])
P_R, U_R = ctx["solve"]([vH]+agents)
vbar = np.mean(np.stack(vs_all), axis=0)
P_avg, _ = ctx["solve"]([vbar])
P_sort = np.zeros((NQ, K, K))
for q, (m, bs) in enumerate(data):
    o = np.argsort(-np.asarray(m)); P_sort[q][o, np.arange(K)] = 1.0

# merit-robust (wrong axis): robust to 5 perturbed provider targets under vH (vertex LP)
J = 5
tgts = [tgt]
for j in range(J-1):
    Mj = np.zeros(len(brands))
    for m, bs in data:
        pert = np.asarray(m)*np.exp(rng.uniform(-0.2, 0.2, size=len(m)))
        for mi, b in zip(pert, bs): Mj[bidx[b]] += mi
    tgts.append(Mj/Mj.sum())
NV = K*K; NVAR = NQ*NV+1; T = NQ*NV
rows, cols, vals, b_eq = [], [], [], []; r = 0
for q in range(NQ):
    for i in range(K):
        for k in range(K): rows.append(r); cols.append(q*NV+i*K+k); vals.append(1.0)
        b_eq.append(1.0); r += 1
    for k in range(K):
        for i in range(K): rows.append(r); cols.append(q*NV+i*K+k); vals.append(1.0)
        b_eq.append(1.0); r += 1
A_eq = sparse.coo_matrix((vals, (rows, cols)), shape=(r, NVAR)).tocsr()
urows, ucols, uvals, b_ub = [], [], [], []; ur = 0
tot = NQ*vH.sum()
for tg in tgts:
    for b in top:
        for sign in (1.0, -1.0):
            for q, (m, bs) in enumerate(data):
                for i, bb in enumerate(bs):
                    if bidx[bb] == b:
                        for k in range(K): urows.append(ur); ucols.append(q*NV+i*K+k); uvals.append(sign*vH[k])
            urows.append(ur); ucols.append(T); uvals.append(-tot*tg[b])
            b_ub.append(sign*tot*tg[b]); ur += 1
A_ub = sparse.coo_matrix((uvals, (urows, ucols)), shape=(ur, NVAR)).tocsr()
c_obj = np.zeros(NVAR); c_obj[T] = 1
res = linprog(c_obj, A_eq=A_eq, b_eq=np.array(b_eq), A_ub=A_ub, b_ub=np.array(b_ub),
              bounds=[(0, 1)]*(NQ*NV)+[(0, None)], method="highs")
assert res.status == 0
P_MR = res.x[:NQ*NV].reshape(NQ, K, K)

oracle = {vn: ctx["solve"]([v])[1] for vn, v in zip(vnames, vs_all)}
table = {}
for nm, Ps in [("relevance-sort", P_sort), ("human-cal", P_H), ("avg-curve", P_avg),
               ("merit-robust", P_MR), ("robust", P_R)]:
    table[nm] = stats(Ps)
    print(f"  {nm:14s}: {table[nm]['a0']:.3f} | {table[nm]['a25']:.3f} | {table[nm]['a1']:.3f} | "
          f"{table[nm]['worst']:.3f} ({table[nm]['argmax']})", flush=True)
table["oracle"] = {"a0": oracle["human"], "worst": max(oracle.values())}
print(f"  oracle        : {oracle['human']:.3f} | - | - | {max(oracle.values()):.3f}", flush=True)

# lambda_break (canonical, main set)
lam = None; prev = None
for a in np.linspace(0, 0.8, 81):
    uh = stats_dummy = None
    uH = U_of(P_H, vH) if a == 0 else max(U_of(P_H, (1-a)*vH + a*v) for v in agents)
    uR = U_of(P_R, vH) if a == 0 else max(U_of(P_R, (1-a)*vH + a*v) for v in agents)
    if prev is not None and prev < 0 and uH-uR >= 0 and lam is None: lam = float(a)
    prev = uH-uR
print(f"lambda_break (canonical, main) = {lam}", flush=True)

# (B) existence audit on canonical P_H
def geo(r):
    v = r**np.arange(1, K+1); return v/v.sum()
rs = np.linspace(0.30, 1.0, 36)
DELTAS = [0.10, 0.15, 0.20, 0.25]
div = {d: [0, 0, 0] for d in DELTAS}
for q, (m, bs) in enumerate(data):
    mm = np.asarray(m, float); mm = mm/mm.sum()
    def Uq(v):
        e = P_H[q] @ v
        return np.max(np.abs(e/(v.sum()*mm)-1))
    u_ex = min(Uq(geo(r)) for r in rs)
    u_mx = max(Uq(v) for v in vs_all)
    for d in DELTAS:
        div[d][2] += 1
        if u_ex <= d:
            div[d][1] += 1
            if u_mx > d: div[d][0] += 1
audit = {str(d): {"acquitted": div[d][1], "acquitted_but_unfair": div[d][0], "total": div[d][2]} for d in DELTAS}
for d in DELTAS:
    a_, b_, t_ = div[d]
    print(f"  delta={d}: exists-pass {b_}/{t_}, of those max-FAIL {a_}", flush=True)

# (C) saturation main -> full
full_agents = agents + list(ext.values())
P_F, U_F = ctx["solve"]([vH]+full_agents)
lamf = None; prev = None
for a in np.linspace(0, 0.8, 81):
    uh = U_of(P_H, vH) if a == 0 else max(U_of(P_H, (1-a)*vH + a*v) for v in full_agents)
    ur = U_of(P_F, vH) if a == 0 else max(U_of(P_F, (1-a)*vH + a*v) for v in full_agents)
    if prev is not None and prev < 0 and uh-ur >= 0 and lamf is None: lamf = float(a)
    prev = uh-ur
loo = {}
for en, ec in ext.items():
    sub = [v for v in full_agents if v is not ec]
    _, Ue = ctx["solve"]([vH]+sub)
    loo[en] = float(U_F-Ue)
print(f"saturation: U_rob main={U_R:.4f} full={U_F:.4f} lam_full={lamf}", flush=True)
print("  LOO:", {k: round(v, 4) for k, v in loo.items()}, flush=True)

json.dump({"table": table, "oracle": oracle, "lambda_break": lam, "audit": audit,
           "U_rob_main": U_R, "U_rob_full": float(U_F), "lambda_break_full": lamf,
           "loo_ext": loo},
          open("main_esci.json", "w"), indent=1)
print("saved main_esci.json")
