"""M1: epsilon-inflated ambiguity set (measurement-noise / unmeasured-model robustness).
Every AGENT curve v_j is replaced by the TV-ball B_eps(v_j) (sum-preserving, outer-
approximated: max <g_b, delta> <= eps*(max_k g_bk - min_k g_bk), linearized with aux
vars M_b >= g_bk >= m_b). vH stays exact. Stage-1 robust LP gives U*(eps); the fixed
canonical P_H gives Umax(a, eps) and hence lambda_break(eps).
Calibration: (i) sampling-protocol noise TV p95 (multinomial n=3200 bootstrap from
each GPT-5.6 tier curve); (ii) measured TV distances between same-family model
variants at K=10 (the "nearby unmeasured model" scale).
Run: python eps_robust.py
"""
import json
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from loaders import LOADERS, K
from policy_structure4 import build_canonical
from curveset import load_set

EPS_GRID = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]

data = LOADERS["ESCI"]()
ctx = build_canonical(data, ntop=15)
vH = ctx["vH"]; agents = ctx["agents"]
NQ = len(data)
brands = ctx["brands"]; bidx = ctx["bidx"]; top = list(ctx["top"]); tgt = ctx["tgt"]
NV = K * K; T = NQ * NV; ntop = len(top)

# member lists: for each top provider, the (q, i) slots occupied by it
slots = [[] for _ in range(ntop)]
for q, (m, bs) in enumerate(data):
    for i, bb in enumerate(bs):
        if bidx[bb] in top:
            slots[top.index(bidx[bb])].append((q, i))

def eq_block(ncols):
    rows, cols, vals, b_eq = [], [], [], []; r = 0
    for q in range(NQ):
        for i in range(K):
            for k in range(K): rows.append(r); cols.append(q*NV+i*K+k); vals.append(1.0)
            b_eq.append(1.0); r += 1
        for k in range(K):
            for i in range(K): rows.append(r); cols.append(q*NV+i*K+k); vals.append(1.0)
            b_eq.append(1.0); r += 1
    return sparse.coo_matrix((vals, (rows, cols)), shape=(r, ncols)).tocsr(), np.array(b_eq)

def solve_inflated(eps):
    # vars: x (T), t (1), M_b (ntop), m_b (ntop)
    ncols = T + 1 + 2*ntop
    tcol = T
    Mcol = lambda jb: T + 1 + jb
    mcol = lambda jb: T + 1 + ntop + jb
    A_eq, b_eq = eq_block(ncols)
    rows, cols, vals, b_ub = [], [], [], []; r = 0
    curves_infl = [(vH, 0.0)] + [(v, eps) for v in agents]
    for v, e in curves_infl:
        tot = NQ * v.sum()
        for jb in range(ntop):
            b = top[jb]
            for sign in (1.0, -1.0):
                for (q, i) in slots[jb]:
                    for k in range(K):
                        rows.append(r); cols.append(q*NV+i*K+k); vals.append(sign*v[k])
                if e > 0:
                    rows.append(r); cols.append(Mcol(jb)); vals.append(e)
                    rows.append(r); cols.append(mcol(jb)); vals.append(-e)
                rows.append(r); cols.append(tcol); vals.append(-tot*tgt[b])
                b_ub.append(sign*tot*tgt[b]); r += 1
    # M_b >= g_bk, m_b <= g_bk for all k
    if eps > 0:
        for jb in range(ntop):
            for k in range(K):
                for (q, i) in slots[jb]:
                    rows.append(r); cols.append(q*NV+i*K+k); vals.append(1.0)
                rows.append(r); cols.append(Mcol(jb)); vals.append(-1.0)
                b_ub.append(0.0); r += 1
                for (q, i) in slots[jb]:
                    rows.append(r); cols.append(q*NV+i*K+k); vals.append(-1.0)
                rows.append(r); cols.append(mcol(jb)); vals.append(1.0)
                b_ub.append(0.0); r += 1
    A_ub = sparse.coo_matrix((vals, (rows, cols)), shape=(r, ncols)).tocsr()
    c = np.zeros(ncols); c[tcol] = 1
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=np.array(b_ub),
                  bounds=[(0, 1)]*T + [(0, None)]*(1 + 2*ntop), method="highs")
    assert res.status == 0, res.message
    return float(res.x[tcol])

# fixed canonical human-calibrated policy for lambda_break(eps)
P_H, U_H = ctx["solve"]([vH])
E = {}          # E[j][jb]: exposure of top provider jb under curve j (0=vH)
G = np.zeros((ntop, K))
for jb in range(ntop):
    for (q, i) in slots[jb]:
        G[jb] += P_H[q][i]
DELTA = G.max(axis=1) - G.min(axis=1)
allc = [vH] + list(agents)
EX = np.array([[float(G[jb] @ v) for jb in range(ntop)] for v in allc])
den = NQ * tgt[np.array(top)]

def umax(a, eps):
    best = 0.0
    for j in range(1, len(allc)):
        Emix = (1-a)*EX[0] + a*EX[j]
        infl = a*eps*DELTA
        up = (Emix + infl)/den - 1
        lo = 1 - (Emix - infl)/den
        best = max(best, float(np.max(np.maximum(up, lo))))
    return max(best, float(np.max(np.abs(EX[0]/den - 1))) if a == 0 else best)

out = {"eps": {}, "U_H": U_H}
for eps in EPS_GRID:
    U = solve_inflated(eps)
    lam = None
    for a in np.linspace(0, 1, 101):
        if a > 0 and umax(float(a), eps) >= U:
            lam = float(a); break
    out["eps"][str(eps)] = {"U": U, "premium": U - U_H, "lam": lam}
    print(f"eps={eps:.3f}  U*={U:.4f}  prem={U-U_H:.4f}  lam_break={lam}", flush=True)

# calibration (i): sampling-noise TV p95, multinomial n=3200 from each GPT tier curve
rng = np.random.default_rng(3)
samp = {}
full = load_set(20, "full")
for name in ["GPT-5.6-sol", "GPT-5.6-terra", "GPT-5.6-luna"]:
    p = full[name]
    tv = [0.5*np.abs(rng.multinomial(3200, p)/3200 - p).sum() for _ in range(2000)]
    samp[name] = float(np.percentile(tv, 95))
    print(f"sampling-noise TV p95 {name}: {samp[name]:.4f}", flush=True)

# calibration (ii): neighbor TV at K=10 (policy-relevant truncation)
c10 = load_set(10, "full")
pairs = [("GPT-5.6-terra", "GPT-5.6-sol"), ("GPT-5.6-luna", "GPT-5.6-sol"),
         ("GPT-5.6-terra", "GPT-5.6-luna"), ("Gemma-4-12B (nf4)", "Gemma-4-E4B"),
         ("GPT-4o", "GPT-5.6-sol"), ("Mistral-7B", "Mistral-Small-3.2 (nf4)"),
         ("Gemma-2-9B", "Gemma-4-12B (nf4)"), ("Qwen2.5-VL-7B", "Qwen3-VL-8B")]
nb = {}
for a_, b_ in pairs:
    ka = [k for k in c10 if k.startswith(a_.split(" (")[0])][0] if a_ not in c10 else a_
    kb = [k for k in c10 if k.startswith(b_.split(" (")[0])][0] if b_ not in c10 else b_
    tv = float(0.5*np.abs(c10[ka] - c10[kb]).sum())
    nb[f"{a_}|{b_}"] = tv
    print(f"neighbor TV {a_} vs {b_}: {tv:.4f}", flush=True)

out["tv_sampling_p95"] = samp
out["tv_neighbors"] = nb
json.dump(out, open("eps_robust.json", "w"), indent=1)
print("saved eps_robust.json")
