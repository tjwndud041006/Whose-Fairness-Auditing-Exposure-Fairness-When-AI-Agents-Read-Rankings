"""Round-13 (external-plan execution), computations A1/A2/A4/A5 + B2-audit +
hull distances + leave-one-family-out. All existing data, CPU LPs only.

A2 RTB: stage 1 min U_inf under vH; stage 2 among stage-1 optima minimize
        max_{v in V} U_inf(Pv). Compare canonical (MAD tie-break), arbitrary
        solver vertex (stage 1 only), RTB: worst-case, Umax(a) sweep ->
        lambda_break vs robust floor, lambda*_cert(0.26), sign-indet top-15
        and top-50, mean interval width.
A1 : per-domain max_v U*(v) over the 9 main curves (oracle worst-type)
     -> premium decomposition (agent-presence vs identification price).
A4 : 20-workload crosstab (U_H <= 0.20) x (premium == 0) from main_frontier.json.
A5 : per-domain human-set {DCG, RBP.8, RBP.9} floor and combined-set floor.
B2 : aggregate-mixture audit on canonical P_H (ESCI): U_q(P vbar_a) <= delta,
     vbar_a = (1-a) vH + a mean(agents), a = 0.25.
C1 : TV distance of each of the 6 extension curves to conv(main set incl vH).
LOFO: leave-one-family-out robust floors (6 families).
Run: python rtb_main.py
"""
import json
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from loaders import LOADERS, K
from policy_structure4 import build_canonical
from curveset import load_set

out = {}
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

# =============== ESCI primary: A2 (RTB, 3 policies) ===============
data = LOADERS["ESCI"]()
ctx = make_ctx(data, 15)
vH = ctx["vH"]; agents = ctx["agents"]; vs_all = [vH]+agents

P_can, U_H = ctx["solve"]([vH])                     # canonical (MAD tie-break)
P_rob, U_R = ctx["solve"](vs_all)                   # robust
P_rtb, U_H2, wc_rtb, P_vtx = rtb_solve(ctx, vH, vs_all)
assert abs(U_H2-U_H) < 1e-6

def umax_sweep(Ps):
    us = []
    for a in np.linspace(0, 1, 101):
        if a == 0: us.append(U_of(ctx, data, Ps, vH))
        else: us.append(max(U_of(ctx, data, Ps, (1-a)*vH+a*v) for v in agents))
    return np.array(us)

AG = np.linspace(0, 1, 101)
res_pol = {}
for name, P in [("canonical", P_can), ("vertex", P_vtx), ("rtb", P_rtb)]:
    us = umax_sweep(P)
    ib = np.where((us >= U_R-1e-9) & (AG > 0))[0]
    lam = float(AG[ib[0]]) if len(ib) else None
    ic = np.where(us <= 0.26)[0]
    cert26 = float(AG[ic[-1]]) if len(ic) else 0.0
    lo, hi = intervals(ctx, data, P, vs_all)
    res_pol[name] = {"worst_case": float(us.max()), "wc_at1": float(us[-1]),
                     "lam_break": lam, "cert_026": cert26,
                     "sind15": int(np.sum((lo < 0) & (hi > 0))),
                     "width_mean": float((hi-lo).mean())}
    print(f"[A2/15] {name:9s} U_H={U_of(ctx,data,P,vH):.4f} wc={us.max():.4f} "
          f"lam={lam} cert.26={cert26:.2f} s-ind={res_pol[name]['sind15']}/15 "
          f"w={res_pol[name]['width_mean']:.3f}", flush=True)
res_pol["U_H"] = U_H; res_pol["U_R"] = U_R; res_pol["rtb_stage2_wc"] = wc_rtb
out["A2_top15"] = res_pol

# top-50 sign-indet, all three policies
ctx50 = make_ctx(data, 50)
P50_can, _ = ctx50["solve"]([vH])
P50_rtb, _, _, P50_vtx = rtb_solve(ctx50, vH, vs_all)
t50 = {}
for name, P in [("canonical", P50_can), ("vertex", P50_vtx), ("rtb", P50_rtb)]:
    lo, hi = intervals(ctx50, data, P, vs_all)
    t50[name] = {"sind50": int(np.sum((lo < 0) & (hi > 0))),
                 "width_mean": float((hi-lo).mean())}
    print(f"[A2/50] {name:9s} s-ind={t50[name]['sind50']}/50 w={t50[name]['width_mean']:.3f}", flush=True)
out["A2_top50"] = t50

# =============== B2: aggregate-mixture audit on canonical P_H ===============
vbar = 0.75*vH + 0.25*np.mean(np.stack(agents), axis=0)
DELTAS = [0.10, 0.15, 0.20, 0.25]
agg = {d: [0, 0] for d in DELTAS}
for q, (m, bs) in enumerate(data):
    mm = np.asarray(m, float); mm = mm/mm.sum()
    Pq = P_can[q]
    ua = float(np.max(np.abs((Pq @ vbar)/(vbar.sum()*mm)-1)))
    umx = max(float(np.max(np.abs((Pq @ v)/(v.sum()*mm)-1))) for v in vs_all)
    for d in DELTAS:
        if ua <= d:
            agg[d][0] += 1
            if umx > d: agg[d][1] += 1
out["B2_aggregate"] = {str(d): {"acquitted": agg[d][0], "acquitted_but_unfair": agg[d][1]} for d in DELTAS}
print("[B2] aggregate-mixture (a=0.25): " +
      " | ".join(f"d={d}: acq {agg[d][0]}/200 (unfair {agg[d][1]})" for d in DELTAS), flush=True)

# =============== C1: TV distance of extension curves to conv(main) ===============
full = load_set(K, "full"); main = load_set(K, "main")
ext = {k: v for k, v in full.items() if k not in main}
base = [vH]+[np.asarray(v) for v in main.values()]
def tv_to_hull(vnew, base):
    nb = len(base); ncols = nb + 2*K       # weights + pos/neg residual
    A_eq = np.zeros((K+1, ncols)); b_eq = np.zeros(K+1)
    for kk in range(K):
        for j in range(nb): A_eq[kk, j] = base[j][kk]
        A_eq[kk, nb+kk] = 1.0; A_eq[kk, nb+K+kk] = -1.0
        b_eq[kk] = vnew[kk]
    A_eq[K, :nb] = 1.0; b_eq[K] = 1.0
    c = np.zeros(ncols); c[nb:] = 0.5
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)]*ncols, method="highs")
    return float(res.fun)
hd = {}
for name, v in ext.items():
    v = np.asarray(v)/np.asarray(v).sum()
    hd[name] = tv_to_hull(v, base)
    print(f"[C1] TV dist to conv(main) {name}: {hd[name]:.4f}", flush=True)
out["C1_hull_tv"] = hd

# =============== LOFO: leave-one-family-out ===============
FAM = {"Qwen": ["Qwen3-VL-8B", "Qwen3-VL-32B (4bit)"], "Gemma": ["Gemma-4-12B (nf4)"],
       "Llama": ["Llama-3.1-8B"], "Mistral": ["Mistral-Small-3.2 (nf4)"],
       "Phi": ["Phi-4-mini"], "OpenAI": ["GPT-4o", "GPT-5.6-sol"]}
names = list(main.keys())
lofo = {}
for fam, members in FAM.items():
    ags = [np.asarray(main[n]) for n in names if n not in members]
    _, Uf = ctx["solve"]([vH]+[a/np.sum(a) for a in ags])
    lofo[fam] = {"U": float(Uf), "premium": float(Uf-U_H)}
    print(f"[LOFO] -{fam:8s} U*={Uf:.4f} prem={Uf-U_H:.4f}", flush=True)
out["LOFO"] = lofo

# =============== A4: workload crosstab ===============
fr = json.load(open("main_frontier.json"))
ct = {"hard_paid": 0, "hard_free": 0, "easy_paid": 0, "easy_free": 0}
for r in fr["rows"]:
    hard = r["U_H"] > 0.20; free = (r["U_R"]-r["U_H"]) < 1e-4
    key = ("hard" if hard else "easy") + "_" + ("free" if free else "paid")
    ct[key] += 1
out["A4_crosstab"] = ct
print(f"[A4] crosstab (U_H>0.20 x premium==0): {ct}", flush=True)

# =============== A1 + A5 per domain ===============
dom = {}
for name in ["ESCI", "TREC22", "Steam", "LastFM"]:
    d = LOADERS[name]()
    cx = make_ctx(d, 15)
    vHx = cx["vH"]; ags = cx["agents"]; va = [vHx]+ags
    _, UHx = cx["solve"]([vHx])
    _, URx = cx["solve"](va)
    percurve = {}
    for i, v in enumerate(va):
        _, u = cx["solve"]([v])
        percurve[i] = float(u)
    omax = max(percurve.values())
    _, Uhs = cx["solve"]([vHx, rbp8, rbp9])
    _, Uall = cx["solve"]([vHx, rbp8, rbp9]+ags)
    dom[name] = {"U_H": float(UHx), "U_R": float(URx), "oracle_worst": float(omax),
                 "presence_price": float(omax-UHx), "ident_price": float(URx-omax),
                 "U_Hset": float(Uhs), "U_combined": float(Uall),
                 "percurve": percurve}
    print(f"[A1/A5] {name:8s} U_H={UHx:.4f} U_R={URx:.4f} oracle_worst={omax:.4f} "
          f"presence={omax-UHx:+.4f} ident={URx-omax:+.4f} "
          f"Hset={Uhs:.4f} comb={Uall:.4f}", flush=True)
out["domains"] = dom

json.dump(out, open("rtb_main.json", "w"), indent=1)
print("saved rtb_main.json")
