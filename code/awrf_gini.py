"""Round-21: report our Steam/TREC policies on the metrics their literatures use.
TREC Fair 2022: AWRF = 1 - JSD(exposure share over groups, target share) under a curve v.
Steam (P-MMF literature): Gini of provider exposure-to-target ratios; also min provider share (P-MMF's MMF, position-weighted).
Policies: relevance sort (identity), human-calibrated lexicographic optimum P_H, robust P_R (same LPs as the paper);
curves: vH, worst agent vertex, a=0.25 mixture worst.
Run: python awrf_gini.py
"""
import json, sys, numpy as np
from scipy import sparse
from scipy.optimize import linprog
from loaders import LOADERS, K
from policy_structure4 import build_canonical
from curveset import load_set, MAIN
from core import *
from core import solve_lex, lam_break, sign_indet
agents = load_set(10, "main"); names = list(agents); V = [agents[n] for n in names]
def exposure(ctx, data, P, v):
    E = np.zeros(len(ctx["brands"]))
    for q, (m, bs) in enumerate(data):
        e = P[q] @ v
        np.add.at(E, [ctx["bidx"][bb] for bb in bs], e)
    return E
def jsd(p, q):
    p = p/p.sum(); q = q/q.sum(); m = 0.5*(p+q)
    def kl(a, b):
        mask = a > 0; return float(np.sum(a[mask]*np.log2(a[mask]/b[mask])))
    return 0.5*kl(p, m) + 0.5*kl(q, m)
def awrf(ctx, data, P, v): return 1 - jsd(exposure(ctx, data, P, v), ctx["tgt"])
def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return float((2*np.sum((np.arange(1, n+1))*x)/(n*x.sum())) - (n+1)/n) if x.sum() > 0 else 0.0
def gini_ratio(ctx, data, P, v, group):
    E = exposure(ctx, data, P, v); g = np.array(group)
    return gini(E[g]/(ctx["NQ"]*v.sum()*ctx["tgt"][g]))
def minshare(ctx, data, P, v, group):
    E = exposure(ctx, data, P, v); g = np.array(group)
    return float(np.min(E[g]/(ctx["NQ"]*v.sum()*ctx["tgt"][g])))
out = {}
for dom, metric in [("TREC22", "awrf"), ("Steam", "gini")]:
    data = LOADERS[dom](); ctx = make_ctx(data, 15); vH = ctx["vH"]; NQ = ctx["NQ"]
    P_rel = np.array([np.eye(K) for _ in range(NQ)])
    P_H, U_H = solve_lex(ctx, [vH], vH)
    P_R, U_R = solve_lex(ctx, [vH]+V, vH)
    allg = list(range(len(ctx["brands"]))); top = ctx["top"]
    res = {}
    for pname, P in [("relevance", P_rel), ("human_cal", P_H), ("robust", P_R)]:
        row = {"U_inf_vH": U_of(ctx, data, P, vH), "U_inf_worst": max(U_of(ctx, data, P, v) for v in V)}
        if metric == "awrf":
            row.update(awrf_vH=awrf(ctx, data, P, vH), awrf_worst=min(awrf(ctx, data, P, v) for v in V),
                       awrf_a25=min(awrf(ctx, data, P, (1-.25)*vH+.25*v) for v in V),
                       awrf_top15_vH=1-jsd(exposure(ctx,data,P,vH)[np.array(top)], ctx["tgt"][np.array(top)]),
                       awrf_top15_worst=min(1-jsd(exposure(ctx,data,P,v)[np.array(top)], ctx["tgt"][np.array(top)]) for v in V))
        else:
            row.update(gini_vH=gini_ratio(ctx, data, P, vH, top), gini_worst=max(gini_ratio(ctx, data, P, v, top) for v in V),
                       gini_all_vH=gini_ratio(ctx, data, P, vH, allg), gini_all_worst=max(gini_ratio(ctx, data, P, v, allg) for v in V),
                       minshare_vH=minshare(ctx, data, P, vH, top), minshare_worst=min(minshare(ctx, data, P, v, top) for v in V))
        res[pname] = row; print(dom, pname, {k: round(v, 3) for k, v in row.items()}, flush=True)
    out[dom] = dict(policies=res, n_groups=len(allg), U_H=U_H, U_R=U_R)
json.dump(out, open("awrf_gini.json", "w"), indent=1); print("saved awrf_gini.json")
