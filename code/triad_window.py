"""Round-24: audit triad where the worst-case verdict is not vacuous.
(A) Workload level: the 20 bootstrap ESCI workloads of main_frontier.py (same seeds), canonical human-calibrated
    policy P_H per workload, amortized top-15 provider U_inf. For tau in a grid spanning the certification window,
    count workloads acquitted by: worst-case over the measured set; existence over geometric decay; existence over
    all monotone curves (LP); existence over the unrestricted simplex (LP); blend a=.25. False-acquittal rate =
    |exist-acquit & worst-fail| / |exist-acquit|.
(B) Query level (canonical workload, item-level U_inf as in Table 3) for delta up to 1.0.
Run: python triad_window.py
"""
import json, numpy as np, pandas as pd
from scipy.optimize import linprog
from loaders import LOADERS, K, esci_table
from policy_structure4 import build_canonical
from curveset import load_set
NQ, NTOP, B = 200, 15, 20
ex = esci_table()
cnt = ex.groupby("query_id").size(); pool = cnt[cnt>=K].index.to_numpy().copy()
vH = 1/np.log2(np.arange(2, K+2)); vH /= vH.sum()
agents = list(load_set(K, "main").values()); vs_all = [vH]+agents; vbar = 0.75*vH+0.25*np.mean(agents, axis=0)
def geo(r): v = r**np.arange(1, K+1); return v/v.sum()
RS = np.linspace(0.30, 1.0, 36)
def exist_lp(A, tgt_vec, tau, monotone):
    """exists v>=0, sum v=1 (monotone: v_k>=v_{k+1}) with |A v / tgt - 1| <= tau row-wise; A: (G,K) exposure per unit v."""
    G = A.shape[0]
    rows = []; b = []
    for g in range(G):
        rows.append(A[g]/tgt_vec[g]); b.append(1+tau)
        rows.append(-A[g]/tgt_vec[g]); b.append(-(1-tau))
    if monotone:
        for k in range(K-1):
            r = np.zeros(K); r[k] = -1; r[k+1] = 1; rows.append(r); b.append(0.0)
    res = linprog(np.zeros(K), A_ub=np.array(rows), b_ub=np.array(b), A_eq=np.ones((1,K)), b_eq=[1.0], bounds=[(0,None)]*K, method="highs")
    return res.status == 0
out = {"workload": {}, "query": {}}
# ---------- (A) workload level ----------
TAUS = [round(x,2) for x in np.arange(0.16, 0.41, 0.02)]
rngb = np.random.default_rng(17)
per_wl = []
for bb in range(B):
    qs = rngb.choice(pool, NQ, replace=False)
    data = []
    for qi in qs:
        g = ex[ex.query_id==qi]
        sub = g.sample(K, random_state=int(qi)%2**31) if len(g)>K else g
        data.append((np.array([{"E":1.0,"S":0.5}[l] for l in sub.esci_label]), sub.product_brand.to_numpy()))
    ctx = build_canonical(data, ntop=NTOP); P_H, U_H = ctx["solve"]([vH])
    top = np.array(ctx["top"]); tgt = ctx["tgt"]; bidx = ctx["bidx"]
    # exposure matrix A[g,k] = sum over slots of provider g of P_H[q][i][k]  -> E_g(v) = A @ v ; normalise by NQ
    A = np.zeros((len(top), K))
    for q, (m, bs) in enumerate(data):
        for i, bbn in enumerate(bs):
            gi = bidx[bbn]
            if gi in top: A[list(top).index(gi)] += P_H[q][i]
    A /= NQ
    U = lambda v: float(np.max(np.abs(A @ v / (v.sum()*tgt[top]) - 1)))
    row = {"U_H": U_H, "U_worst": max(U(v) for v in vs_all), "U_geo_min": min(U(geo(r)) for r in RS), "U_blend": U(vbar)}
    row["mono_ok"] = {str(t): exist_lp(A, tgt[top], t, True) for t in TAUS}
    row["free_ok"] = {str(t): exist_lp(A, tgt[top], t, False) for t in TAUS}
    per_wl.append(row); print(f"wl {bb}: U_H={U_H:.3f} worst={row['U_worst']:.3f} geo_min={row['U_geo_min']:.3f} blend={row['U_blend']:.3f}", flush=True)
for t in TAUS:
    wc = [r["U_worst"] <= t for r in per_wl]; ge = [r["U_geo_min"] <= t for r in per_wl]
    mo = [r["mono_ok"][str(t)] for r in per_wl]; fr = [r["free_ok"][str(t)] for r in per_wl]; bl = [r["U_blend"] <= t for r in per_wl]
    def fa(ex_): n = sum(ex_); f = sum(e and not w for e, w in zip(ex_, wc)); return (f, n)
    out["workload"][str(t)] = {"worst_acq": sum(wc), "geo_acq": sum(ge), "geo_false": fa(ge), "mono_acq": sum(mo), "mono_false": fa(mo),
                               "free_acq": sum(fr), "free_false": fa(fr), "blend_acq": sum(bl), "blend_false": fa(bl)}
    print(t, out["workload"][str(t)], flush=True)
out["per_workload"] = [{k: v for k, v in r.items() if k not in ("mono_ok","free_ok")} for r in per_wl]
# ---------- (B) query level, canonical workload ----------
data = LOADERS["ESCI"](); ctx = build_canonical(data, ntop=NTOP); P_H, U_H = ctx["solve"]([vH])
DELTAS = [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
per_q = []
for q, (m, bs) in enumerate(data):
    mm = np.asarray(m, float); mm = mm/mm.sum(); Pq = P_H[q]
    Uq = lambda v: float(np.max(np.abs((Pq @ v)/(v.sum()*mm) - 1)))
    r = {"worst": max(Uq(v) for v in vs_all), "geo": min(Uq(geo(rr)) for rr in RS), "blend": Uq(vbar)}
    r["mono"] = {str(d): exist_lp(Pq, mm, d, True) for d in DELTAS}; r["free"] = {str(d): exist_lp(Pq, mm, d, False) for d in DELTAS}
    per_q.append(r)
for d in DELTAS:
    wc = [r["worst"] <= d for r in per_q]; ge = [r["geo"] <= d for r in per_q]; mo = [r["mono"][str(d)] for r in per_q]; fr = [r["free"][str(d)] for r in per_q]; bl = [r["blend"] <= d for r in per_q]
    def fa(ex_): n = sum(ex_); f = sum(e and not w for e, w in zip(ex_, wc)); return (f, n)
    out["query"][str(d)] = {"worst_acq": sum(wc), "geo_acq": sum(ge), "geo_false": fa(ge), "mono_acq": sum(mo), "mono_false": fa(mo), "free_acq": sum(fr), "free_false": fa(fr), "blend_acq": sum(bl), "blend_false": fa(bl)}
    print("q", d, out["query"][str(d)], flush=True)
out["query_worst_dist"] = {"min": min(r["worst"] for r in per_q), "p10": float(np.percentile([r["worst"] for r in per_q], 10)), "median": float(np.median([r["worst"] for r in per_q]))}
json.dump(out, open("triad_window.json", "w"), indent=1); print("saved triad_window.json")
