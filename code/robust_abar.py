"""Round-38: share-bounded robust policy. min_P max_{v in A(abar)} U_inf(Pv) with A(abar)={(1-a)vH + a v : a in [0,abar], v in conv(V^A)};
by convexity the binding vertices are vH and (1-abar)vH + abar*v_j. Report Umax(0), Umax(0.25), worst type (a=1), lambda_break vs U*.
ESCI canonical workload, top-15, lexicographic (stage 1 U_inf, stage 2 MAD under vH) via ctx["solve"]."""
import json, numpy as np
from loaders import LOADERS, K
from policy_structure4 import build_canonical
from curveset import load_set
from scipy import sparse
from scipy.optimize import linprog
from core import *
agents=[np.asarray(v,float) for v in load_set(K,"main").values()]
data=LOADERS["ESCI"](); ctx=make_ctx(data,15); vH=ctx["vH"]
_,U_R=ctx["solve"]([vH]+agents)
out={}
for abar in [0.25, 0.5]:
    curves=[vH]+[(1-abar)*vH+abar*v for v in agents]
    P,Uab=ctx["solve"](curves)
    def Umax(a): return U_of(ctx,data,P,vH) if a==0 else max(U_of(ctx,data,P,(1-a)*vH+a*v) for v in agents)
    worst=max(U_of(ctx,data,P,v) for v in agents); wname=None
    names=list(load_set(K,"main").keys()); wname=names[int(np.argmax([U_of(ctx,data,P,v) for v in agents]))]
    lam=None; prev=None
    for a in np.linspace(0,1,101):
        u=Umax(a)
        if prev is not None and prev<0 and u-U_R>=0 and lam is None: lam=float(a)
        prev=u-U_R
    out[str(abar)]=dict(U_obj=Uab,a0=Umax(0),a25=Umax(0.25),a50=Umax(0.5),worst=worst,worst_name=wname,lam_break=lam)
    print(abar, {k:(round(v,4) if isinstance(v,float) else v) for k,v in out[str(abar)].items()}, flush=True)
json.dump(out,open('robust_abar.json','w'),indent=1); print('saved')
