"""Round-24: 20 bootstrap workloads for TREC22 and LastFM (U_H, U*, premium, lambda_break), seed-parametrised copies of the loaders."""
import json, sys, numpy as np
sys.argv=[sys.argv[0]]
import loaders
from loaders import K, NQ
from policy_structure4 import build_canonical
from curveset import load_set
src=open('loaders.py').read()
def make_loader(fn_name, seed_default, seed):
    a=src.index(f"def {fn_name}("); b=src.index("\ndef ", a+5) if "\ndef " in src[a+5:] else len(src)
    body=src[a:b].replace(f"np.random.default_rng({seed_default})", f"np.random.default_rng({seed})")
    ns=dict(vars(loaders)); exec(body, ns); return ns[fn_name]
vH = 1/np.log2(np.arange(2, K+2)); vH /= vH.sum(); agents=list(load_set(K,"main").values())
out={}
for dom, fn, sd in [("TREC22","trec_data",7),("LastFM","lastfm_data",21)]:
    rows=[]
    for b in range(20):
        seed=1000+b
        data=make_loader(fn, sd, seed)()
        ctx=build_canonical(data, ntop=15); P_H,U_H=ctx["solve"]([vH]); P_R,U_R=ctx["solve"]([vH]+agents)
        def U(P,v):
            E=np.zeros(len(ctx["brands"]))
            for q,(m,bs) in enumerate(data): np.add.at(E,[ctx["bidx"][x] for x in bs],P[q]@v)
            g=np.array(ctx["top"]); return float(np.max(np.abs(E[g]/(len(data)*v.sum()*ctx["tgt"][g])-1)))
        lam=None; prev=None
        for a in np.linspace(0,1,101):
            u=U(P_H,vH) if a==0 else max(U(P_H,(1-a)*vH+a*v) for v in agents)
            if prev is not None and prev<0 and u-U_R>=0 and lam is None: lam=float(a)
            prev=u-U_R
        rows.append(dict(seed=seed,U_H=U_H,U_R=U_R,premium=U_R-U_H,lam=lam)); print(dom,b,rows[-1],flush=True)
    pr=[r["premium"] for r in rows]; lm=[r["lam"] for r in rows if r["lam"] is not None]
    out[dom]=dict(rows=rows, premium_p5=float(np.percentile(pr,5)), premium_p95=float(np.percentile(pr,95)), premium_median=float(np.median(pr)),
                  lam_p5=float(np.percentile(lm,5)) if lm else None, lam_p95=float(np.percentile(lm,95)) if lm else None, lam_median=float(np.median(lm)) if lm else None, lam_none=20-len(lm))
    json.dump(out, open("boot_domains.json","w"), indent=1)
print("saved boot_domains.json")
