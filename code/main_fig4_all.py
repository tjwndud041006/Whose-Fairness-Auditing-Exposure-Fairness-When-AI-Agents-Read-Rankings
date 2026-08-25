"""Round-9 rebase, step 3: top-50 provider intervals on ESCI under MAIN set +
canonical policy. Five bootstrap subsets (seeds 100..104, as before); subset
100 also yields Fig-4 rows (intervals, human-value dot, mean rank) and the
strata decomposition (head/mid/tail widths, centers, crossings) plus
r(width, exposure-weighted mean rank).
Run: python main_fig4.py
"""
import json
import numpy as np
import pandas as pd
from policy_structure4 import build_canonical
from curveset import load_set
from loaders import esci_table

K, NQ, NTOP = 10, 200, 50
ex0 = esci_table()
cnt = ex0.groupby("query_id").size(); all_q = cnt[cnt>=K].index.to_numpy().copy()
vH = 1/np.log2(np.arange(2, K+2)); vH /= vH.sum()
agents = list(load_set(K, "main").values())
vs_all = [vH]+agents

def subset_data(seed):
    r2 = np.random.default_rng(seed)
    qs = r2.choice(all_q, NQ, replace=False)
    data = []
    for qi in qs:
        g = ex0[ex0.query_id==qi]
        sub = g.sample(K, random_state=int(qi)%2**31) if len(g)>K else g
        data.append((np.array([{"E":1.0,"S":0.5}[l] for l in sub.esci_label]), sub.product_brand.to_numpy()))
    return data

pooled_cross, pooled_n = 0, 0
fig_rows = None; strata = None; all_rows = []
for s in range(5):
    data = subset_data(100+s)
    ctx = build_canonical(data, ntop=NTOP)
    P, _ = ctx["solve"]([vH])
    top, bidx, tgt, M = ctx["top"], ctx["bidx"], ctx["tgt"], ctx["M"]
    rows = []
    for b in top:
        ds = []
        for v in vs_all:
            E = 0.0
            for q, (m, bs) in enumerate(data):
                e = P[q] @ v
                for i, bb in enumerate(bs):
                    if bidx[bb] == b: E += e[i]
            ds.append(E/(NQ*v.sum()*tgt[b])-1)
        # exposure-weighted mean rank under P
        num, den = 0.0, 0.0
        rankpos = np.arange(1, K+1)
        for q, (m, bs) in enumerate(data):
            for i, bb in enumerate(bs):
                if bidx[bb] == b:
                    pr_ = P[q][i]; num += (pr_*rankpos).sum(); den += pr_.sum()
        rows.append({"dev_lo": min(ds), "dev_hi": max(ds), "dev_human": ds[0],
                     "merit_share": float(tgt[b]), "mean_rank": num/max(den, 1e-9)})
    cross = sum(1 for r in rows if r["dev_lo"] < 0 < r["dev_hi"])
    pooled_cross += cross; pooled_n += len(rows); all_rows.append(rows)
    print(f"subset {100+s}: {cross}/{len(rows)} cross", flush=True)
    if s == 0:
        fig_rows = rows
        rr = sorted(rows, key=lambda r: -r["merit_share"])
        strata = {}
        for nm, sl in (("head", rr[:17]), ("mid", rr[17:33]), ("tail", rr[33:])):
            w = np.array([r["dev_hi"]-r["dev_lo"] for r in sl])
            c = np.array([abs((r["dev_hi"]+r["dev_lo"])/2) for r in sl])
            x = np.array([r["dev_lo"] < 0 < r["dev_hi"] for r in sl])
            strata[nm] = {"n": len(sl), "cross": int(x.sum()), "width": float(w.mean()),
                          "abs_center": float(c.mean())}
        w = np.array([r["dev_hi"]-r["dev_lo"] for r in rows])
        x = np.array([r["dev_lo"] < 0 < r["dev_hi"] for r in rows])
        mr = np.array([r["mean_rank"] for r in rows])
        cw = float(np.corrcoef(mr, w)[0, 1])
        wc = float(w[x].mean()) if x.any() else None
        wn = float(w[~x].mean()) if (~x).any() else None
        extra = {"r_width_meanrank": cw, "width_crossers": wc, "width_noncross": wn}
        print(f"  strata: {strata}", flush=True)
        print(f"  r(width, mean rank)={cw:.2f} width cross/non={wc:.2f}/{wn:.2f}", flush=True)
print(f"pooled: {pooled_cross}/{pooled_n} ({100*pooled_cross/pooled_n:.0f}%)", flush=True)
json.dump({"fig_rows": fig_rows, "all_rows": all_rows, "pooled_cross": pooled_cross, "pooled_n": pooled_n,
           "strata": strata, **extra},
          open("main_fig4_all.json", "w"), indent=1)
print("saved main_fig4.json")
