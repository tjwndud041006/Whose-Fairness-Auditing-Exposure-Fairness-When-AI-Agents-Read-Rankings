"""Round-9 rebase, step 1: Table 2 (all four domains) under the frozen MAIN set
(8 agents + human) and the canonical lexicographic policy for BOTH P_H and P_R.
Outputs per domain: U*_H, U*_rob, premium, lambda_break, sign-ind (top-15),
mean/max interval width, NDCG_H, Delta_rob, and the full U(a) sweeps for Fig 5
(ESCI + TREC).
Run: python main_tables.py
"""
import json
import numpy as np
from loaders import LOADERS, K
from policy_structure4 import build_canonical, widths_all

disc = 1/np.log2(np.arange(2, K+2))

def ndcg(P, data):
    vals = []
    for q, (m, bs) in enumerate(data):
        m = np.asarray(m, dtype=float)
        vals.append(float(m @ (P[q] @ disc))/float(np.sort(m)[::-1] @ disc))
    return float(np.mean(vals))

out = {}
for name, loader in LOADERS.items():
    data = loader()
    ctx = build_canonical(data, ntop=15)      # build_canonical -> port_core.load_curves -> MAIN set
    vH = ctx["vH"]; agents = ctx["agents"]; curves = [vH]+agents
    NQ = len(data); top = np.array(ctx["top"]); tgt = ctx["tgt"]
    P_H, U_H = ctx["solve"]([vH])
    P_R, U_R = ctx["solve"](curves)

    def U_of(Ps, v):
        E = np.zeros(len(ctx["brands"]))
        for q, (m, bs) in enumerate(data):
            e = Ps[q] @ v
            np.add.at(E, [ctx["bidx"][b] for b in bs], e)
        return float(np.max(np.abs(E[top]/(NQ*v.sum()*tgt[top])-1)))

    def at_share(Ps, a):
        if a == 0: return U_of(Ps, vH)
        return max(U_of(Ps, (1-a)*vH + a*v) for v in agents)

    lam = None; prev = None; sweep = []
    for a in np.linspace(0, 0.8, 81):
        uh, ur = at_share(P_H, a), at_share(P_R, a)
        sweep.append((float(a), uh, ur))
        if prev is not None and prev < 0 and uh-ur >= 0 and lam is None: lam = float(a)
        prev = uh-ur

    loH, hiH = widths_all(P_H, data, ctx, curves, top)
    loR, hiR = widths_all(P_R, data, ctx, curves, top)
    nH, nR = ndcg(P_H, data), ndcg(P_R, data)
    out[name] = {"U_H": U_H, "U_rob": U_R, "premium": U_R-U_H, "lambda_break": lam,
                 "sign_ind_H": int(np.sum((loH < 0) & (hiH > 0))),
                 "sign_ind_R": int(np.sum((loR < 0) & (hiR > 0))),
                 "width_H_mean": float((hiH-loH).mean()), "width_H_min": float((hiH-loH).min()),
                 "width_H_max": float((hiH-loH).max()), "width_R_mean": float((hiR-loR).mean()),
                 "ndcg_H": nH, "ndcg_R": nR, "delta_rob": nH-nR, "sweep": sweep}
    print(f"{name:8s} U_H={U_H:.4f} U_R={U_R:.4f} prem={U_R-U_H:.4f} lam={lam} "
          f"s-ind={out[name]['sign_ind_H']}/15 (R {out[name]['sign_ind_R']}) "
          f"wH={out[name]['width_H_mean']:.3f}[{out[name]['width_H_min']:.3f},{out[name]['width_H_max']:.3f}] "
          f"NDCG={nH:.3f} drob={nH-nR:+.4f}", flush=True)
json.dump(out, open("main_tables.json", "w"), indent=1)
print("saved main_tables.json")
