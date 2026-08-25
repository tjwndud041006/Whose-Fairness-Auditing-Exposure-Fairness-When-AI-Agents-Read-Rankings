"""Shared LP machinery (slots formulation) used by the ESCI/TREC/Steam/LastFM analyses.
Verbatim from the original analysis code (previously shared between scripts by exec of a
common source segment); collected here as an importable module.

  make_ctx(data, ntop)         : build_canonical + per-provider slot lists
  stage1(ctx, curves)          : min_P max_{v in curves} U_inf(Pv)   (an arbitrary solver vertex)
  solve_lex(ctx, curves, vH)   : stage 1, then MAD tie-break under vH (the canonical policy of the paper)
  rtb_solve(ctx, vH, curves)   : stage 1 under vH, then robust tie-break (min worst case over curves)
  U_of / intervals / sign_indet / lam_break : evaluation helpers
"""
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from loaders import K
from policy_structure4 import build_canonical

rbp8 = 0.8**np.arange(K); rbp8 /= rbp8.sum()
rbp9 = 0.9**np.arange(K); rbp9 /= rbp9.sum()

# ---------- generic machinery (slots formulation, as eps_robust.py) ----------
def make_ctx(data, ntop):
    ctx = build_canonical(data, ntop=ntop)
    NQ = len(data)
    top = list(ctx["top"]); bidx = ctx["bidx"]; tgt = ctx["tgt"]
    slots = [[] for _ in range(len(top))]
    for q, (m, bs) in enumerate(data):
        for i, bb in enumerate(bs):
            if bidx[bb] in top:
                slots[top.index(bidx[bb])].append((q, i))
    ctx["slots"] = slots; ctx["NQ"] = NQ
    return ctx

def eq_block(NQ, ncols):
    rows, cols, vals, b_eq = [], [], [], []; r = 0
    NV = K*K
    for q in range(NQ):
        for i in range(K):
            for k in range(K): rows.append(r); cols.append(q*NV+i*K+k); vals.append(1.0)
            b_eq.append(1.0); r += 1
        for k in range(K):
            for i in range(K): rows.append(r); cols.append(q*NV+i*K+k); vals.append(1.0)
            b_eq.append(1.0); r += 1
    return sparse.coo_matrix((vals, (rows, cols)), shape=(r, ncols)).tocsr(), np.array(b_eq)

def dev_rows_slots(ctx, v, var_col, rows, cols, vals, b_ub, r):
    NQ = ctx["NQ"]; top = ctx["top"]; tgt = ctx["tgt"]; slots = ctx["slots"]
    NV = K*K; tot = NQ*v.sum()
    for jb, b in enumerate(top):
        for sign in (1.0, -1.0):
            for (q, i) in slots[jb]:
                for k in range(K): rows.append(r); cols.append(q*NV+i*K+k); vals.append(sign*v[k])
            rows.append(r); cols.append(var_col); vals.append(-tot*tgt[b])
            b_ub.append(sign*tot*tgt[b]); r += 1
    return r

def stage1(ctx, curves):
    """min t s.t. dev(v) <= t for v in curves; returns (P_vertex, t*)."""
    NQ = ctx["NQ"]; T = NQ*K*K; ncols = T+1
    A_eq, b_eq = eq_block(NQ, ncols)
    rows, cols, vals, b_ub = [], [], [], []; r = 0
    for v in curves:
        r = dev_rows_slots(ctx, v, T, rows, cols, vals, b_ub, r)
    A_ub = sparse.coo_matrix((vals, (rows, cols)), shape=(r, ncols)).tocsr()
    c = np.zeros(ncols); c[T] = 1
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=np.array(b_ub),
                  bounds=[(0, 1)]*T+[(0, None)], method="highs")
    assert res.status == 0, res.message
    return res.x[:T].reshape(NQ, K, K), float(res.x[T])

def rtb_solve(ctx, vH, allcurves):
    """Stage 1: min t under vH. Stage 2: t fixed, min s = worst over allcurves."""
    P1, U_H = stage1(ctx, [vH])
    NQ = ctx["NQ"]; T = NQ*K*K; ncols = T+2   # t, s
    A_eq, b_eq = eq_block(NQ, ncols)
    rows, cols, vals, b_ub = [], [], [], []; r = 0
    r = dev_rows_slots(ctx, vH, T, rows, cols, vals, b_ub, r)
    for v in allcurves:
        r = dev_rows_slots(ctx, v, T+1, rows, cols, vals, b_ub, r)
    rows.append(r); cols.append(T); vals.append(1.0); b_ub.append(U_H+1e-8); r += 1
    A_ub = sparse.coo_matrix((vals, (rows, cols)), shape=(r, ncols)).tocsr()
    c = np.zeros(ncols); c[T+1] = 1
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=np.array(b_ub),
                  bounds=[(0, 1)]*T+[(0, None)]*2, method="highs")
    assert res.status == 0, res.message
    return res.x[:T].reshape(NQ, K, K), U_H, float(res.x[T+1]), P1

def U_of(ctx, data, Ps, v, group=None):
    E = np.zeros(len(ctx["brands"]))
    for q, (m, bs) in enumerate(data):
        e = Ps[q] @ v
        np.add.at(E, [ctx["bidx"][bb] for bb in bs], e)
    g = np.array(ctx["top"]) if group is None else np.array(group)
    return float(np.max(np.abs(E[g]/(ctx["NQ"]*v.sum()*ctx["tgt"][g])-1)))

def intervals(ctx, data, Ps, vs):
    devs = {b: [] for b in ctx["top"]}
    for v in vs:
        E = np.zeros(len(ctx["brands"]))
        for q, (m, bs) in enumerate(data):
            e = Ps[q] @ v
            np.add.at(E, [ctx["bidx"][bb] for bb in bs], e)
        tot = ctx["NQ"]*v.sum()
        for b in ctx["top"]:
            devs[b].append(E[b]/(tot*ctx["tgt"][b]) - 1)
    lo = np.array([min(devs[b]) for b in ctx["top"]])
    hi = np.array([max(devs[b]) for b in ctx["top"]])
    return lo, hi

# ---------- lexicographic (canonical) solve and evaluation helpers (as k10_analysis.py) ----------
def solve_lex(ctx, curves, vH, tol=1e-6):
    P1,t1 = stage1(ctx, curves)
    NQ=ctx["NQ"]; T=NQ*K*K; nd=len(ctx["top"]); ncols=T+1+nd
    A_eq,b_eq=eq_block(NQ,ncols); rows,cols,vals,b_ub=[],[],[],[]; r=0
    for v in curves: r=dev_rows_slots(ctx,v,T,rows,cols,vals,b_ub,r)
    tot=NQ*vH.sum(); NV=K*K
    for jb,b in enumerate(ctx["top"]):
        for sign in (1.0,-1.0):
            for (q,i) in ctx["slots"][jb]:
                for k in range(K): rows.append(r); cols.append(q*NV+i*K+k); vals.append(sign*vH[k])
            rows.append(r); cols.append(T+1+jb); vals.append(-tot*ctx["tgt"][b]); b_ub.append(sign*tot*ctx["tgt"][b]); r+=1
    rows.append(r); cols.append(T); vals.append(1.0); b_ub.append(t1+tol); r+=1
    A_ub=sparse.coo_matrix((vals,(rows,cols)),shape=(r,ncols)).tocsr()
    c=np.zeros(ncols); c[T+1:]=1.0
    res=linprog(c,A_eq=A_eq,b_eq=b_eq,A_ub=A_ub,b_ub=np.array(b_ub),bounds=[(0,1)]*T+[(0,None)]*(1+nd),method="highs")
    if res.status!=0: return P1, float(t1)
    return res.x[:T].reshape(NQ,K,K), float(t1)

def lam_break(ctx, data, P_H, vH, verts, U_ref):
    prev, lam = None, None
    for a in np.linspace(0, 1, 101):
        u = U_of(ctx,data,P_H,vH) if a == 0 else max(U_of(ctx,data,P_H,(1-a)*vH+a*v) for v in verts)
        if prev is not None and prev < 0 and u-U_ref >= 0 and lam is None: lam = float(a)
        prev = u-U_ref
    return lam

def sign_indet(ctx, data, P, vs):
    top=ctx["top"]; cnt=0
    for b in top:
        devs=[]
        for v in vs:
            E=0.0
            for (q,i) in ctx["slots"][top.index(b)]: E += float(P[q][i] @ v)
            devs.append(E/(ctx["NQ"]*v.sum()*ctx["tgt"][b])-1)
        if min(devs)<0<max(devs): cnt+=1
    return cnt

