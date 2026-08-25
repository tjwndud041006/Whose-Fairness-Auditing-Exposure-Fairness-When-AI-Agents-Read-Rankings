"""M3: Steam workload variability, B=20 bootstrap user-workloads (200 users each,
resampled without replacement from the eligible pool). Per workload: U*_H, U*_rob,
premium, lambda_break (same crossover definition as main_tables). Second domain
check that the Table-2 point estimates are not artifacts of one user draw.
Run: python steam_frontier.py
"""
import json
import numpy as np
import scipy.sparse as sp
from policy_structure4 import build_canonical
from curveset import load_set
from loaders import K, NQ

B = 20
from config import DATA
BASE = f"{DATA}/steam"
d = np.load(f"{BASE}/ranking_scores.npz")
S = sp.coo_matrix((d["data"], (d["row"], d["col"])), shape=tuple(d["shape"])).tocsr()
iid2pid = {int(k): int(v) for k, v in
           json.load(open(f"{BASE}/iid2pid.json")).items()}
pool = np.array([u for u in range(S.shape[0]) if S[u].nnz >= K])
print(f"eligible users: {len(pool)}", flush=True)

vH = 1/np.log2(np.arange(2, K+2)); vH /= vH.sum()
agents = list(load_set(K, "main").values())

def workload(users):
    data = []
    for u in users:
        row = S[u]
        items = row.indices; scores = row.data
        topk = items[np.argsort(-scores)[:K]]
        sc = np.sort(scores)[::-1][:K]
        m = np.maximum(sc, 1e-3)
        gs = np.array([f"pub{iid2pid.get(int(i), -1)}" for i in topk])
        data.append((m, gs))
    return data

rows = []
rngb = np.random.default_rng(23)
for b in range(B):
    data = workload(rngb.choice(pool, NQ, replace=False))
    ctx = build_canonical(data, ntop=15)
    P_H, U_H = ctx["solve"]([vH])
    P_R, U_R = ctx["solve"]([vH]+agents)
    top = np.array(ctx["top"]); tgt = ctx["tgt"]
    def U_of(Ps, v):
        E = np.zeros(len(ctx["brands"]))
        for q, (m, bs) in enumerate(data):
            e = Ps[q] @ v
            np.add.at(E, [ctx["bidx"][bb] for bb in bs], e)
        return float(np.max(np.abs(E[top]/(NQ*v.sum()*tgt[top])-1)))
    def at_share(Ps, a):
        if a == 0: return U_of(Ps, vH)
        return max(U_of(Ps, (1-a)*vH + a*v) for v in agents)
    lam = None; prev = None
    for a in np.linspace(0, 0.8, 81):
        uh, ur = at_share(P_H, a), at_share(P_R, a)
        if prev is not None and prev < 0 and uh-ur >= 0 and lam is None: lam = float(a)
        prev = uh-ur
    rows.append({"U_H": float(U_H), "U_R": float(U_R),
                 "premium": float(U_R-U_H), "lam": lam})
    print(f"workload {b}: U_H={U_H:.4f} U_R={U_R:.4f} prem={U_R-U_H:.4f} lam={lam}", flush=True)

prem = [r["premium"] for r in rows]
lams = [r["lam"] for r in rows if r["lam"] is not None]
out = {"rows": rows,
       "U_H_mean": float(np.mean([r["U_H"] for r in rows])),
       "U_H_range": [float(min(r["U_H"] for r in rows)), float(max(r["U_H"] for r in rows))],
       "U_R_mean": float(np.mean([r["U_R"] for r in rows])),
       "U_R_range": [float(min(r["U_R"] for r in rows)), float(max(r["U_R"] for r in rows))],
       "premium_mean": float(np.mean(prem)),
       "premium_range": [float(min(prem)), float(max(prem))],
       "lam_mean": float(np.mean(lams)) if lams else None,
       "lam_range": [float(min(lams)), float(max(lams))] if lams else None,
       "lam_none_count": int(sum(1 for r in rows if r["lam"] is None))}
json.dump(out, open("steam_frontier.json", "w"), indent=1)
print("saved steam_frontier.json")
print(f"U_H mean {out['U_H_mean']:.4f} {out['U_H_range']}; U_R mean {out['U_R_mean']:.4f} {out['U_R_range']}")
print(f"premium mean {out['premium_mean']:.4f} range {out['premium_range']}; "
      f"lam mean {out['lam_mean']} range {out['lam_range']} none={out['lam_none_count']}")
