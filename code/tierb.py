"""Round-14 Tier-B: M2 box-constrained ambiguity, M5 null-set baselines, M-b.
M2: W(wbar) = {sum w_k v^Ak : w in simplex, w_k <= wbar}. Vertices = uniform
    mixtures over ceil(1/wbar)-subsets. Robust solve over {vH} + vertices for
    wbar in {1/2, 1/3, 1/4} (wbar=1 is the existing U_R). Also lambda_break.
M5: 20 permutation-null 8-curve sets (entries of each measured curve randomly
    permuted) and 20 Dirichlet(1^K)-null sets: U*, premium, lambda_break,
    sign-indet under canonical P_H. Measured-set row for comparison.
M-b: average-curve policy re-solved with lexicographic tie-break (min U under
     vbar, then MAD under vbar) -> U_inf under vH at a=0.
Run: python tierb.py
"""
import json
from itertools import combinations
import numpy as np
from loaders import LOADERS, K
from policy_structure4 import build_canonical

data = LOADERS["ESCI"]()
ctx = build_canonical(data, ntop=15)
vH = ctx["vH"]; agents = [np.asarray(v) for v in ctx["agents"]]
NQ = len(data); top = np.array(ctx["top"]); tgt = ctx["tgt"]
out = {}

def U_of(Ps, v):
    E = np.zeros(len(ctx["brands"]))
    for q, (m, bs) in enumerate(data):
        e = Ps[q] @ v
        np.add.at(E, [ctx["bidx"][bb] for bb in bs], e)
    return float(np.max(np.abs(E[top]/(NQ*v.sum()*tgt[top])-1)))

P_H, U_H = ctx["solve"]([vH])
_, U_R = ctx["solve"]([vH]+agents)
print(f"baseline U_H={U_H:.4f} U_R={U_R:.4f}", flush=True)

def lam_break(vertex_curves, U_ref):
    prev, lam = None, None
    for a in np.linspace(0, 1, 101):
        u = U_of(P_H, vH) if a == 0 else max(U_of(P_H, (1-a)*vH+a*v) for v in vertex_curves)
        if prev is not None and prev < 0 and u-U_ref >= 0 and lam is None: lam = float(a)
        prev = u-U_ref
    return lam

# ---------------- M2: box-constrained mixture weights ----------------
m2 = {"1": {"U": U_R, "premium": U_R-U_H, "lam": lam_break(agents, U_R), "n_vertices": len(agents)}}
print(f"[M2] wbar=1    U*={U_R:.4f} prem={U_R-U_H:.4f} lam={m2['1']['lam']}", flush=True)
for denom in (2, 3, 4):
    verts = [np.mean([agents[i] for i in c], axis=0) for c in combinations(range(len(agents)), denom)]
    _, Uw = ctx["solve"]([vH]+verts)
    lam = lam_break(verts, Uw)
    m2[f"1/{denom}"] = {"U": float(Uw), "premium": float(Uw-U_H), "lam": lam, "n_vertices": len(verts)}
    print(f"[M2] wbar=1/{denom}  U*={Uw:.4f} prem={Uw-U_H:.4f} lam={lam} ({len(verts)} vertices)", flush=True)
out["M2_box"] = m2

# ---------------- M5: null-set baselines ----------------
def set_stats(null_agents):
    _, Un = ctx["solve"]([vH]+null_agents)
    lam = lam_break(null_agents, Un)
    devs = {b: [] for b in top}
    for v in [vH]+null_agents:
        E = np.zeros(len(ctx["brands"]))
        for q, (m, bs) in enumerate(data):
            e = P_H[q] @ v
            np.add.at(E, [ctx["bidx"][bb] for bb in bs], e)
        tot = NQ*v.sum()
        for b in top: devs[b].append(E[b]/(tot*tgt[b])-1)
    sind = sum(1 for b in top if min(devs[b]) < 0 < max(devs[b]))
    return float(Un), float(Un-U_H), lam, int(sind)

rng = np.random.default_rng(11)
for kind in ("perm", "dirichlet"):
    rows = []
    for s in range(20):
        if kind == "perm":
            na = [v[rng.permutation(K)].copy() for v in agents]
        else:
            na = [rng.dirichlet(np.ones(K)) for _ in range(len(agents))]
        Un, pr, lam, sind = set_stats(na)
        rows.append({"U": Un, "premium": pr, "lam": lam, "sind": sind})
        print(f"[M5-{kind}] set {s}: U*={Un:.4f} prem={pr:.4f} lam={lam} sind={sind}/15", flush=True)
    U_ = [r["U"] for r in rows]; Pm = [r["premium"] for r in rows]
    Lm = [r["lam"] for r in rows if r["lam"] is not None]; Sd = [r["sind"] for r in rows]
    out[f"M5_{kind}"] = {"rows": rows,
        "U_mean": float(np.mean(U_)), "U_sd": float(np.std(U_)),
        "prem_mean": float(np.mean(Pm)), "prem_sd": float(np.std(Pm)),
        "lam_mean": float(np.mean(Lm)) if Lm else None,
        "lam_sd": float(np.std(Lm)) if Lm else None,
        "lam_none": sum(1 for r in rows if r["lam"] is None),
        "sind_mean": float(np.mean(Sd)), "sind_sd": float(np.std(Sd))}
    o = out[f"M5_{kind}"]
    print(f"[M5-{kind}] U {o['U_mean']:.4f}±{o['U_sd']:.4f} prem {o['prem_mean']:.4f}±{o['prem_sd']:.4f} "
          f"lam {o['lam_mean']}±{o['lam_sd']} sind {o['sind_mean']:.1f}±{o['sind_sd']:.1f}", flush=True)

# ---------------- M-b: average-curve with lexicographic tie-break ----------------
vbar = np.mean(np.stack([vH]+agents), axis=0)
ctx_b = build_canonical(data, ntop=15, vH_override=vbar)   # tie-break MAD under vbar
P_avg, _ = ctx_b["solve"]([vbar])
out["Mb_avg_lex"] = {"U_vH_a0": U_of(P_avg, vH),
                    "wc_vertex": max(U_of(P_avg, v) for v in [vH]+agents)}
print(f"[M-b] average-curve (lex tie-break): U_inf under vH at a=0 = {out['Mb_avg_lex']['U_vH_a0']:.4f} "
      f"(paper: 0.327); worst vertex {out['Mb_avg_lex']['wc_vertex']:.4f}", flush=True)

json.dump(out, open("tierb.json", "w"), indent=1)
print("saved tierb.json")
