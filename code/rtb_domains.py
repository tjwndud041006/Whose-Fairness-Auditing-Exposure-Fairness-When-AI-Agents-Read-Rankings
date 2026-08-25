"""Round-30: tie-break dependence of lambda_break on all four domains.
For each domain: canonical (MAD tie-break) human-calibrated policy, an arbitrary solver vertex (stage-1 only),
and the robust tie-break (RTB, stage 2 minimizing worst case over V); report worst case at a=1 and lambda_break vs U*.
Run: python rtb_domains.py   (CPU, ~minutes per domain)"""
import json, numpy as np
from scipy import sparse
from scipy.optimize import linprog
from loaders import LOADERS, K
from policy_structure4 import build_canonical
from curveset import load_set
from core import *
agents=list(load_set(K,"main").values())
out={}
for dom in ["ESCI","TREC22","Steam","LastFM"]:
    data=LOADERS[dom](); ctx=make_ctx(data,15); vH=ctx["vH"]
    P_can,U_H=ctx["solve"]([vH])
    P_rtb,U_H2,wc_rtb,P_vtx=rtb_solve(ctx,vH,agents)
    _,U_R=ctx["solve"]([vH]+agents)
    def lam(P):
        prev=None
        for a in np.linspace(0,1,101):
            u=U_of(ctx,data,P,vH) if a==0 else max(U_of(ctx,data,P,(1-a)*vH+a*v) for v in agents)
            if prev is not None and prev<0 and u-U_R>=0: return float(a)
            prev=u-U_R
        return None
    row={"U_H":U_H,"U_R":U_R}
    for name,P in [("canonical",P_can),("vertex",P_vtx),("rtb",P_rtb)]:
        row[name]={"worst":max(U_of(ctx,data,P,v) for v in agents),"lam":lam(P)}
    out[dom]=row; print(dom,{k:(v if not isinstance(v,dict) else {kk:(round(vv,3) if isinstance(vv,float) else vv) for kk,vv in v.items()}) for k,v in row.items()},flush=True)
    json.dump(out,open('rtb_domains.json','w'),indent=1)
print('saved rtb_domains.json')
