"""Round-9 rebase, step 4: certification frontier under MAIN set + canonical
policies, B=20 bootstrap ESCI workloads. Per workload: U*_H, U*_rob, and the
exact Umax(a) sweep of the canonical human-calibrated policy (no linearity
assumption). Outputs cert rates over a tau grid and lambda*_cert(tau) stats.
Run: python main_frontier.py
"""
import json
import numpy as np
import pandas as pd
from policy_structure4 import build_canonical
from curveset import load_set
from loaders import esci_table

K, NQ, NTOP, B = 10, 200, 15, 20
ex = esci_table()
cnt = ex.groupby("query_id").size(); pool = cnt[cnt>=K].index.to_numpy().copy()
vH = 1/np.log2(np.arange(2, K+2)); vH /= vH.sum()
agents = list(load_set(K, "main").values())

rows = []
rngb = np.random.default_rng(17)
AGRID = np.linspace(0, 1, 51)
for b in range(B):
    qs = rngb.choice(pool, NQ, replace=False)
    data = []
    for qi in qs:
        g = ex[ex.query_id==qi]
        sub = g.sample(K, random_state=int(qi)%2**31) if len(g)>K else g
        data.append((np.array([{"E":1.0,"S":0.5}[l] for l in sub.esci_label]), sub.product_brand.to_numpy()))
    ctx = build_canonical(data, ntop=NTOP)
    P_H, U_H = ctx["solve"]([vH])
    P_R, U_R = ctx["solve"]([vH]+agents)
    top = np.array(ctx["top"]); tgt = ctx["tgt"]
    def U_of(Ps, v):
        E = np.zeros(len(ctx["brands"]))
        for q, (m, bs) in enumerate(data):
            e = Ps[q] @ v
            np.add.at(E, [ctx["bidx"][bb] for bb in bs], e)
        return float(np.max(np.abs(E[top]/(NQ*v.sum()*tgt[top])-1)))
    umax = []
    for a in AGRID:
        if a == 0: umax.append(U_of(P_H, vH))
        else: umax.append(max(U_of(P_H, (1-a)*vH + a*v) for v in agents))
    rows.append({"U_H": float(U_H), "U_R": float(U_R), "umax": umax})
    print(f"workload {b}: U_H={U_H:.3f} U_R={U_R:.3f} umax(1)={umax[-1]:.3f}", flush=True)

taus = np.round(np.arange(0.14, 0.46, 0.02), 3).tolist()
cert_known = [float(np.mean([r["U_H"] <= t for r in rows])) for t in taus]
cert_ambig = [float(np.mean([r["U_R"] <= t for r in rows])) for t in taus]
lam_mean, lam_p5, lam_p95 = [], [], []
for t in taus:
    lams = []
    for r in rows:
        u = np.array(r["umax"])
        ok = np.where(u <= t)[0]
        lams.append(float(AGRID[ok[-1]]) if len(ok) else np.nan)
    lams = np.array(lams, dtype=float)
    valid = ~np.isnan(lams)
    lam_mean.append(float(np.nanmean(lams)) if valid.any() else None)
    lam_p5.append(float(np.nanpercentile(lams, 5)) if valid.any() else None)
    lam_p95.append(float(np.nanpercentile(lams, 95)) if valid.any() else None)
prem = [r["U_R"]-r["U_H"] for r in rows]
out = {"taus": taus, "cert_known": cert_known, "cert_ambig": cert_ambig,
       "lam_mean": lam_mean, "lam_p5": lam_p5, "lam_p95": lam_p95,
       "U_H_mean": float(np.mean([r["U_H"] for r in rows])),
       "U_H_range": [float(min(r["U_H"] for r in rows)), float(max(r["U_H"] for r in rows))],
       "U_R_mean": float(np.mean([r["U_R"] for r in rows])),
       "U_R_range": [float(min(r["U_R"] for r in rows)), float(max(r["U_R"] for r in rows))],
       "premium_range": [float(min(prem)), float(max(prem))],
       "premium_zero_count": int(sum(1 for p in prem if p < 1e-4)),
       "rows": rows}
json.dump(out, open("main_frontier.json", "w"), indent=1)
print("saved main_frontier.json")
print(f"U_H mean {out['U_H_mean']:.3f} {out['U_H_range']}; U_R mean {out['U_R_mean']:.3f} {out['U_R_range']}")
print(f"premium range {out['premium_range']}, zero on {out['premium_zero_count']}/20")
