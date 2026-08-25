import json, sys, numpy as np
from itertools import combinations
from loaders import LOADERS, K
from policy_structure4 import build_canonical, widths_all, brand_exposures
from port_core import load_curves
out={}
# ---------- A) ESCI sign-indeterminacy under canonical human-calibrated policy
data=LOADERS["ESCI"](); ctx=build_canonical(data,ntop=15); vH=ctx["vH"]
_,agents=load_curves(K); names=list(agents); A=[np.asarray(agents[n],float) for n in names]
P_H,U_H=ctx["solve"]([vH]); top=np.array(ctx["top"]); NQ=len(data)
def sind(curves):
    lo,hi=widths_all(P_H,data,ctx,curves,top); return int(np.sum((lo<0)&(hi>0)))
base=sind([vH]+A); out["baseline"]=base; print("baseline s-ind",base,"(Table 2: 4)")
# leave one curve out / leave one family out
fam={n:n.split("-")[0].replace("GPT","GPT") for n in names}
fam={n:("GPT" if n.startswith("GPT") else "Qwen" if n.startswith("Qwen") else n.split("-")[0]) for n in names}
loo={n:sind([vH]+[a for m,a in zip(names,A) if m!=n]) for n in names}
lofo={f:sind([vH]+[a for m,a in zip(names,A) if fam[m]!=f]) for f in sorted(set(fam.values()))}
out["loo"]=loo; out["lofo"]=lofo; print("LOO",loo); print("LOFO",lofo)
# human side as a set
rbp8=0.8**np.arange(K); rbp8/=rbp8.sum(); rbp9=0.9**np.arange(K); rbp9/=rbp9.sum()
out["human_set"]=sind([vH,rbp8,rbp9]+A); print("human set s-ind",out["human_set"])
# mixture cap 1/4: vertices = uniform mixtures over 4-subsets (70)
verts=[np.mean([A[i] for i in c],axis=0) for c in combinations(range(8),4)]
out["cap14"]=sind([vH]+verts); print("cap 1/4 s-ind",out["cap14"],"n_vertices",len(verts))
# TV ball eps=0.05 (outer approx as in eps_robust.py): dev_b(v)=g_b.v-1, range over ball = +-eps*(max_k g_bk - min_k g_bk)
G=np.zeros((len(ctx["brands"]),K))
for q,(m,bs) in enumerate(data):
    for i,bb in enumerate(bs): G[ctx["bidx"][bb]]+=P_H[q][i]
G=G/(NQ*ctx["tgt"][:,None]); G=G[top]  # 15 x K (agent curves sum to 1)
eps=0.05; rng=G.max(1)-G.min(1)
devH=G@vH-1; devA=np.array([G@a-1 for a in A])   # sanity: should equal widths_all
lo=np.minimum(devH,(devA-eps*rng).min(0)); hi=np.maximum(devH,(devA+eps*rng).max(0))
out["tvball05"]=int(np.sum((lo<0)&(hi>0))); print("TV ball .05 s-ind",out["tvball05"], "| exact-ball check:",end=" ")
# exact (simplex-constrained) ball for reference
def exact_range(g,v,eps):
    order=np.argsort(g); hi=g@v; lo=g@v
    # move eps mass from lowest-g positions to highest-g position
    rem=eps; gain=0.0
    for k in order:
        if rem<=0: break
        t=min(v[k],rem); gain+=t*(g[order[-1]]-g[k]); rem-=t
    rem=eps; loss=0.0
    for k in order[::-1]:
        if rem<=0: break
        t=min(v[k],rem); loss+=t*(g[k]-g[order[0]]); rem-=t
    return hi+gain, lo-loss
his=[];los=[]
for b in range(15):
    hb=devH[b]; lb=devH[b]
    for a in A:
        h,l=exact_range(G[b],a,eps); hb=max(hb,h-1); lb=min(lb,l-1)
    his.append(hb); los.append(lb)
print(int(np.sum((np.array(los)<0)&(np.array(his)>0))))
# ---------- B) Steam Gini at a=0.25 (worst over agents), human-cal and robust
from core import *
from core import solve_lex, lam_break, sign_indet
from curveset import load_set
ag=load_set(10,"main"); V=[ag[n] for n in ag]
def exposure(ctx,data,P,v):
    E=np.zeros(len(ctx["brands"]))
    for q,(m,bs) in enumerate(data):
        e=P[q]@v; np.add.at(E,[ctx["bidx"][bb] for bb in bs],e)
    return E
def gini(x):
    x=np.sort(np.asarray(x,float)); n=len(x)
    return float((2*np.sum((np.arange(1,n+1))*x)/(n*x.sum()))-(n+1)/n) if x.sum()>0 else 0.0
def gini_ratio(ctx,data,P,v,group):
    E=exposure(ctx,data,P,v); g=np.array(group); return gini(E[g]/(ctx["NQ"]*v.sum()*ctx["tgt"][g]))
sd=LOADERS["Steam"](); sctx=make_ctx(sd,15); svH=sctx["vH"]; stop=sctx["top"]
PH,_=solve_lex(sctx,[svH],svH); PR,_=solve_lex(sctx,[svH]+V,svH)
for nm,P in [("human_cal",PH),("robust",PR)]:
    r={"gini_vH":gini_ratio(sctx,sd,P,svH,stop),"gini_worst":max(gini_ratio(sctx,sd,P,v,stop) for v in V),
       "gini_a25":max(gini_ratio(sctx,sd,P,0.75*svH+0.25*v,stop) for v in V)}
    out["steam_"+nm]=r; print("Steam",nm,{k:round(x,4) for k,x in r.items()})
json.dump(out,open("fill_sind_gini.json","w"),indent=1); print("saved")
