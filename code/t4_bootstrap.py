"""Table 4 (tab:stress), per workload: every perturbation of the measured curve set, evaluated on
the canonical ESCI workload and on each of the B=20 bootstrap workloads of the Table 2 band suite.

t4_table4.py then summarises these runs as the typeset table: each perturbation's canonical value
plus/minus the SD of its PAIRED difference from the baseline across the 20 bootstrap workloads.
Pairing matters: the marginal spread across workloads (U*_H ranges .169-.247, premium 0-.137) is an
order of magnitude larger than any perturbation effect and would make all ten rows indistinguishable.

Workload draws follow main_frontier.py exactly -- one rng(17) stream, qs = rng.choice(pool, 200,
replace=False) per bootstrap -- so the workloads are the same ones behind the Table 2 brackets.

Rows and their original single-workload sources:
  measured baseline / leave-one-out / human-side set          review_fixes.py
  direct K=10 (3 measurement seeds) / real items              k10_analysis.py [--prefix=...]
  access vs choice construct (six models)                     construct_robust.py
  TV ball eps=0.05                                            eps_robust.py
  permutation and Dirichlet nulls / mixture cap wbar=1/4      tierb.py

Run: python t4_bootstrap.py --all              (21 workloads, ~1 h on one core)
     python t4_bootstrap.py --b=7              (one workload; run several in parallel)
     python t4_bootstrap.py --all --ndraw=5    (fewer null draws, for a quick pass)
Writes t4_boot_<b>.json next to this script, one per workload.
"""
import sys, os, json, time
from itertools import combinations
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

import loaders
from loaders import K, NQ as NQ_CFG
from config import CURVES
from policy_structure4 import build_canonical, widths_all
from curveset import load_set

WORKLOADS = ["canon"] + [str(i) for i in range(20)]
NDRAW = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--ndraw=")), 20))
TODO = WORKLOADS if "--all" in sys.argv else \
    [next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--b=")), "canon")]

GAIN = {"E": 1.0, "S": 0.5}
SIX = ["Qwen2.5-VL-7B-Instruct", "Qwen3-VL-8B-Instruct", "Qwen3-VL-32B-Instruct-bnb-4bit",
       "Mistral-7B-Instruct-v0.3", "Llama-3.1-8B-Instruct", "gemma-2-9b-it"]
K10FILES = {
    "Qwen3-VL-8B": "k10_Qwen3-VL-8B-Instruct.json",
    "Qwen3-VL-32B (4bit)": "k10_Qwen3-VL-32B-Instruct-bnb-4bit.json",
    "Gemma-4-12B (nf4)": "k10_gemma-4-12B-it-nf4.json",
    "Llama-3.1-8B": "k10_Llama-3.1-8B-Instruct.json",
    "Mistral-Small-3.2 (nf4)": "k10_Mistral-Small-3.2-24B-Instruct-2506-nf4.json",
    "Phi-4-mini": "k10_Phi-4-mini-instruct.json",
    "GPT-4o": "k10_gpt-4o.json",
    "GPT-5.6-sol": "k10_gpt-5.6-sol.json"}
FAMILY = {"Qwen3-VL-8B": "Qwen", "Qwen3-VL-32B (4bit)": "Qwen", "Gemma-4-12B (nf4)": "Gemma",
          "Llama-3.1-8B": "Llama", "Mistral-Small-3.2 (nf4)": "Mistral", "Phi-4-mini": "Phi",
          "GPT-4o": "GPT", "GPT-5.6-sol": "GPT"}


def build_workload(b):
    """b='canon' -> loaders.esci_data(); b=i -> bootstrap workload i of the main_frontier.py suite."""
    if b == "canon":
        return loaders.esci_data()
    ex = loaders.esci_table()
    cnt = ex.groupby("query_id").size()
    pool = cnt[cnt >= K].index.to_numpy().copy()
    rngb = np.random.default_rng(17)
    for _ in range(int(b) + 1):
        qs = rngb.choice(pool, NQ_CFG, replace=False)
    data = []
    for qi in qs:
        g = ex[ex.query_id == qi]
        sub = g.sample(K, random_state=int(qi) % 2**31) if len(g) > K else g
        data.append((np.array([GAIN[l] for l in sub.esci_label]), sub.product_brand.to_numpy()))
    return data


def remeasured(prefix):
    """The main-8 set re-measured under `prefix`; names any curve that has no such file."""
    curves, fb = [], []
    for name, fn in K10FILES.items():
        fnp = fn.replace("k10_", prefix + "_", 1)
        if not os.path.exists(f"{CURVES}/{fnp}"):
            fnp = fn
            fb.append(name)
        c = np.array(json.load(open(f"{CURVES}/{fnp}"))["results"]["pick"]["curve"], float)[:K]
        curves.append(c / c.sum())
    return curves, fb


def pick_curve(m):
    for pat in ("lcurve", "newcurve"):
        p = f"{CURVES}/{pat}_{m}.json"
        if os.path.exists(p):
            c = np.array(json.load(open(p))["results"]["pick"]["curve"], float)[:K]
            return c / c.sum()
    raise FileNotFoundError(m)


def access_curve(m):
    c = np.array(json.load(open(f"{CURVES}/ccurve_{m}.json"))["acc_by_pos"], float)[:K]
    return c / c.sum()


def run_workload(b):
    t0 = time.time()
    data = build_workload(b)
    ctx = build_canonical(data, ntop=15)
    vH = ctx["vH"]
    AG = [np.asarray(v) for v in ctx["agents"]]
    NAMES = list(load_set(K, "main").keys())
    NQ = len(data); top = np.array(ctx["top"]); tgt = ctx["tgt"]
    rbp8 = 0.8**np.arange(K); rbp8 /= rbp8.sum()
    rbp9 = 0.9**np.arange(K); rbp9 /= rbp9.sum()
    print(f"[{b}] NQ={NQ} providers={len(ctx['brands'])}", flush=True)

    def U_of(Ps, v):
        E = np.zeros(len(ctx["brands"]))
        for q, (m, bs) in enumerate(data):
            np.add.at(E, [ctx["bidx"][bb] for bb in bs], Ps[q] @ v)
        return float(np.max(np.abs(E[top] / (NQ * v.sum() * tgt[top]) - 1)))

    P_H, U_H = ctx["solve"]([vH])

    def lam_break(ags, U_ref):
        """First agent share at which Umax(a) under the fixed human-calibrated policy reaches U_ref
        (the grid and convention of tierb.py / core.lam_break)."""
        prev, lam = None, None
        for a in np.linspace(0, 1, 101):
            u = U_of(P_H, vH) if a == 0 else max(U_of(P_H, (1 - a) * vH + a * v) for v in ags)
            if prev is not None and prev < 0 and u - U_ref >= 0 and lam is None:
                lam = float(a)
            prev = u - U_ref
        return lam

    def sind(curves):
        lo, hi = widths_all(P_H, data, ctx, curves, top)
        return int(np.sum((lo < 0) & (hi > 0)))

    def row(label, agents):
        _, U_R = ctx["solve"]([vH] + list(agents))
        r = {"premium": float(U_R - U_H), "U_R": float(U_R), "U_H": U_H,
             "lam_break": lam_break(agents, U_R), "sind": sind([vH] + list(agents))}
        print(f"  {label:44s} prem={r['premium']:+.4f} lam={r['lam_break']} "
              f"s-ind={r['sind']}/15", flush=True)
        return r

    R = {"b": b, "U_H": U_H, "n_providers": len(ctx["brands"]), "ndraw": NDRAW, "rows": {}}
    R["rows"]["baseline"] = row("measured curves (baseline)", AG)

    R["rows"]["k10_seeds"] = {}
    for tag, pref in [("s42", "k10"), ("s1", "k10s1"), ("s2", "k10s2")]:
        cs, fb = remeasured(pref)
        R["rows"]["k10_seeds"][tag] = row(f"direct K=10 ({pref})", cs)
        R["rows"]["k10_seeds"][tag]["fallback"] = fb
    cs, fb = remeasured("real")
    R["rows"]["real_items"] = row("real ESCI items", cs)
    R["rows"]["real_items"]["fallback"] = fb

    R["rows"]["choice_six"] = row("choice curves, six models", [pick_curve(m) for m in SIX])
    R["rows"]["access_six"] = row("access curves, six models", [access_curve(m) for m in SIX])

    # ---- TV ball eps=0.05: every agent curve replaced by its TV ball, vH exact (eps_robust.py) ----
    NV = K * K; T = NQ * NV; ntop = len(top); tl = list(top)
    slots = [[] for _ in range(ntop)]
    for q, (m, bs) in enumerate(data):
        for i, bb in enumerate(bs):
            if ctx["bidx"][bb] in top:
                slots[tl.index(ctx["bidx"][bb])].append((q, i))

    def eq_block(ncols):
        rows, cols, vals, b_eq = [], [], [], []; r = 0
        for q in range(NQ):
            for i in range(K):
                for k in range(K):
                    rows.append(r); cols.append(q * NV + i * K + k); vals.append(1.0)
                b_eq.append(1.0); r += 1
            for k in range(K):
                for i in range(K):
                    rows.append(r); cols.append(q * NV + i * K + k); vals.append(1.0)
                b_eq.append(1.0); r += 1
        return sparse.coo_matrix((vals, (rows, cols)), shape=(r, ncols)).tocsr(), np.array(b_eq)

    def solve_inflated(eps):
        ncols = T + 1 + 2 * ntop; tcol = T
        Mcol = lambda jb: T + 1 + jb
        mcol = lambda jb: T + 1 + ntop + jb
        A_eq, b_eq = eq_block(ncols)
        rows, cols, vals, b_ub = [], [], [], []; r = 0
        for v, e in [(vH, 0.0)] + [(v, eps) for v in AG]:
            tot = NQ * v.sum()
            for jb in range(ntop):
                bb_ = top[jb]
                for sign in (1.0, -1.0):
                    for (q, i) in slots[jb]:
                        for k in range(K):
                            rows.append(r); cols.append(q * NV + i * K + k); vals.append(sign * v[k])
                    if e > 0:
                        rows.append(r); cols.append(Mcol(jb)); vals.append(e)
                        rows.append(r); cols.append(mcol(jb)); vals.append(-e)
                    rows.append(r); cols.append(tcol); vals.append(-tot * tgt[bb_])
                    b_ub.append(sign * tot * tgt[bb_]); r += 1
        if eps > 0:
            for jb in range(ntop):
                for k in range(K):
                    for (q, i) in slots[jb]:
                        rows.append(r); cols.append(q * NV + i * K + k); vals.append(1.0)
                    rows.append(r); cols.append(Mcol(jb)); vals.append(-1.0)
                    b_ub.append(0.0); r += 1
                    for (q, i) in slots[jb]:
                        rows.append(r); cols.append(q * NV + i * K + k); vals.append(-1.0)
                    rows.append(r); cols.append(mcol(jb)); vals.append(1.0)
                    b_ub.append(0.0); r += 1
        A_ub = sparse.coo_matrix((vals, (rows, cols)), shape=(r, ncols)).tocsr()
        c = np.zeros(ncols); c[tcol] = 1
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=np.array(b_ub),
                      bounds=[(0, 1)] * T + [(0, None)] * (1 + 2 * ntop), method="highs")
        assert res.status == 0, res.message
        return float(res.x[tcol])

    G = np.zeros((ntop, K))
    for jb in range(ntop):
        for (q, i) in slots[jb]:
            G[jb] += P_H[q][i]
    DELTA = G.max(axis=1) - G.min(axis=1)
    EX = np.array([[float(G[jb] @ v) for jb in range(ntop)] for v in [vH] + AG])
    den = NQ * tgt[top]

    def umax_eps(a, eps):
        best = float(np.max(np.abs(EX[0] / den - 1))) if a == 0 else 0.0
        for j in range(1, len(EX)):
            Emix = (1 - a) * EX[0] + a * EX[j]; infl = a * eps * DELTA
            best = max(best, float(np.max(np.maximum((Emix + infl) / den - 1,
                                                     1 - (Emix - infl) / den))))
        return best

    U_eps = solve_inflated(0.05)
    lam_eps = next((float(a) for a in np.linspace(0, 1, 101)
                    if a > 0 and umax_eps(float(a), 0.05) >= U_eps), None)
    e_lo = np.minimum(EX[0] / den - 1, (EX[1:] / den - 1 - 0.05 * DELTA / den).min(0))
    e_hi = np.maximum(EX[0] / den - 1, (EX[1:] / den - 1 + 0.05 * DELTA / den).max(0))
    R["rows"]["tv_ball_eps05"] = {"premium": float(U_eps - U_H), "U_R": U_eps, "U_H": U_H,
                                  "lam_break": lam_eps,
                                  "sind": int(np.sum((e_lo < 0) & (e_hi > 0)))}
    print(f"  {'TV ball eps=0.05':44s} prem={U_eps - U_H:+.4f} lam={lam_eps} "
          f"s-ind={R['rows']['tv_ball_eps05']['sind']}/15", flush=True)

    # ---- null curve sets (tierb.py, rng(11)) ----
    rng = np.random.default_rng(11)
    for kind in ("perm", "dirichlet"):
        rows_ = []
        for _ in range(NDRAW):
            na = [v[rng.permutation(K)].copy() for v in AG] if kind == "perm" \
                else [rng.dirichlet(np.ones(K)) for _ in AG]
            _, Un = ctx["solve"]([vH] + na)
            rows_.append({"premium": float(Un - U_H), "U_R": float(Un),
                          "lam_break": lam_break(na, Un), "sind": sind([vH] + na)})

        def agg(key):
            vals = [r[key] for r in rows_ if r[key] is not None]
            return {"mean": float(np.mean(vals)), "sd": float(np.std(vals)),
                    "n": len(vals), "n_none": NDRAW - len(vals)}
        R["rows"][f"null_{kind}"] = {"draws": rows_, "premium": agg("premium"),
                                     "lam_break": agg("lam_break"), "sind": agg("sind")}
        a_ = R["rows"][f"null_{kind}"]
        print(f"  {'null ' + kind:44s} prem={a_['premium']['mean']:.4f}+-{a_['premium']['sd']:.4f} "
              f"lam={a_['lam_break']['mean']:.3f} s-ind={a_['sind']['mean']:.2f}", flush=True)

    # ---- leave one curve / one family out ----
    loo = {}
    for j, nm in enumerate(NAMES):
        loo[f"-{nm}"] = row(f"LOO -{nm}", [v for i, v in enumerate(AG) if i != j])
    for fam in sorted(set(FAMILY.values())):
        keep = [v for n, v in zip(NAMES, AG) if FAMILY[n] != fam]
        if len(keep) == len(AG):
            continue
        loo[f"-family:{fam}"] = row(f"LOO family -{fam}", keep)
    R["rows"]["loo"] = loo

    # ---- human side as a set ----
    hset = [vH, rbp8, rbp9]
    _, U_hs = ctx["solve"](hset)
    _, U_all = ctx["solve"](hset + AG)
    R["rows"]["human_set"] = {"premium": float(U_all - U_H), "U_R": float(U_all), "U_H": U_H,
                              "U_Hset_only": float(U_hs), "lam_break": lam_break(AG, U_all),
                              "sind": sind(hset + AG)}
    print(f"  {'human side as a set':44s} prem={U_all - U_H:+.4f} "
          f"lam={R['rows']['human_set']['lam_break']} "
          f"s-ind={R['rows']['human_set']['sind']}/15", flush=True)

    # ---- mixture cap wbar = 1/4 (tierb.py M2: vertices are uniform mixtures over 4-subsets) ----
    verts = [np.mean([AG[i] for i in c], axis=0) for c in combinations(range(len(AG)), 4)]
    R["rows"]["mixture_cap_quarter"] = row(f"mixture cap wbar=1/4 ({len(verts)} verts)", verts)

    out = f"t4_boot_{b}.json"
    json.dump(R, open(out, "w"), indent=1)
    print(f"saved {out} ({time.time() - t0:.0f}s)", flush=True)


for b in TODO:
    run_workload(b)
print("done:", " ".join(f"t4_boot_{b}.json" for b in TODO))
