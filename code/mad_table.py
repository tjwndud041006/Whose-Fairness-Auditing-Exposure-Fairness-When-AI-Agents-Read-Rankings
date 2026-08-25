"""MAD replication of the policy table (ESCI canonical workload, top-15).
MAD(e) = mean_b |e_b/(c m_b) - 1| over top-15 providers. Policies solved under MAD:
human-MAD (min MAD under vH), robust-MAD (min max_v MAD), average-curve-MAD; plus
relevance sort and the U_inf-canonical policies evaluated under MAD. Report MAD at
a=0, a=0.25, worst type; and U_inf of the MAD policies for cross-reading.
Run: python mad_table.py
"""
import json, numpy as np
from scipy import sparse
from scipy.optimize import linprog
from loaders import LOADERS, K
from policy_structure4 import build_canonical
from core import *
data = LOADERS["ESCI"](); ctx = make_ctx(data, 15); vH=ctx["vH"]; agents=ctx["agents"]; vs_all=[vH]+agents
NQ=ctx["NQ"]; ntop=len(ctx["top"]); top=np.array(ctx["top"]); tgt=ctx["tgt"]

def dev_vec(Ps, v):
    E=np.zeros(len(ctx["brands"]))
    for q,(m,bs) in enumerate(data):
        e=Ps[q]@v; np.add.at(E,[ctx["bidx"][bb] for bb in bs],e)
    return E[top]/(NQ*v.sum()*tgt[top])-1
def MAD(Ps,v): return float(np.mean(np.abs(dev_vec(Ps,v))))

def solve_mad(curves):
    """min s s.t. for each curve j: sum_b d_{bj}/ntop <= s, d_{bj} >= +-dev_bj."""
    T=NQ*K*K; J=len(curves); ncols=T+1+J*ntop
    A_eq,b_eq=eq_block(NQ,ncols)
    rows,cols,vals,b_ub=[],[],[],[]; r=0
    for j,v in enumerate(curves):
        # dev rows with var_col = d_{bj}: need per-provider var -> custom
        tot=NQ*v.sum(); NV=K*K
        for jb,b in enumerate(ctx["top"]):
            dcol=T+1+j*ntop+jb
            for sign in (1.0,-1.0):
                for (q,i) in ctx["slots"][jb]:
                    for k in range(K): rows.append(r); cols.append(q*NV+i*K+k); vals.append(sign*v[k])
                rows.append(r); cols.append(dcol); vals.append(-tot*tgt[b]); b_ub.append(sign*tot*tgt[b]); r+=1
        # sum_b d_bj / ntop - s <= 0
        for jb in range(ntop): rows.append(r); cols.append(T+1+j*ntop+jb); vals.append(1.0/ntop)
        rows.append(r); cols.append(T); vals.append(-1.0); b_ub.append(0.0); r+=1
    A_ub=sparse.coo_matrix((vals,(rows,cols)),shape=(r,ncols)).tocsr()
    c=np.zeros(ncols); c[T]=1
    res=linprog(c,A_eq=A_eq,b_eq=b_eq,A_ub=A_ub,b_ub=np.array(b_ub),bounds=[(0,1)]*T+[(0,None)]*(1+J*ntop),method="highs")
    assert res.status==0,res.message
    return res.x[:T].reshape(NQ,K,K), float(res.x[T])

vbar=np.mean(np.stack(vs_all),axis=0)
P_hm,Uhm=solve_mad([vH]); P_rm,Urm=solve_mad(vs_all); P_am,Uam=solve_mad([vbar])
P_can,_=ctx["solve"]([vH]); P_rob,_=ctx["solve"](vs_all)
# relevance sort
P_rel=np.zeros((NQ,K,K))
for q,(m,bs) in enumerate(data):
    order=np.argsort(-np.asarray(m)); 
    for rank,i in enumerate(order): P_rel[q][i][rank]=1.0
out={}
for nm,P in (("relevance",P_rel),("human-MAD",P_hm),("average-MAD",P_am),("robust-MAD",P_rm),("human-Uinf(canonical)",P_can),("robust-Uinf",P_rob)):
    a0=MAD(P,vH); a25=max(MAD(P,0.75*vH+0.25*v) for v in agents); wt=max(MAD(P,v) for v in vs_all)
    wi=max(U_of(ctx,data,P,v) for v in vs_all)
    out[nm]={"mad_a0":a0,"mad_a25":a25,"mad_worst":wt,"uinf_worst":wi}
    print(f"{nm:24s} MAD a=0 {a0:.3f}  a=.25 {a25:.3f}  worst {wt:.3f} | Uinf worst {wi:.3f}",flush=True)
print("floors: human-MAD",round(Uhm,4),"robust-MAD",round(Urm,4),"avg-MAD(obj under vbar)",round(Uam,4))
out["floors"]={"human_mad":Uhm,"robust_mad":Urm}
json.dump(out,open("mad_table.json","w"),indent=1); print("saved mad_table.json")
