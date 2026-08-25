"""(A) mixture-calibrated baselines + merit-definition x premium (ESCI canonical workload)."""
import json, numpy as np, pandas as pd
from loaders import K, NQ, esci_table
from policy_structure4 import build_canonical
from curveset import load_set

from scipy import sparse
from scipy.optimize import linprog
from core import *
def solve_lex(ctx, curves, vH, tol=1e-6):
    """stage1 min t over curves; stage2 t<=t*+tol, min MAD under vH (aux d_b)."""
    P1,t1 = stage1(ctx, curves)
    NQ=ctx["NQ"]; T=NQ*K*K; nd=len(ctx["top"]); ncols=T+1+nd
    A_eq,b_eq=eq_block(NQ,ncols); rows,cols,vals,b_ub=[],[],[],[]; r=0
    for v in curves: r=dev_rows_slots(ctx,v,T,rows,cols,vals,b_ub,r)
    # MAD aux: dev under vH per provider, each with its own var
    tot=NQ*vH.sum(); NV=K*K
    for jb,b in enumerate(ctx["top"]):
        for sign in (1.0,-1.0):
            for (q,i) in ctx["slots"][jb]:
                for k in range(K): rows.append(r); cols.append(q*NV+i*K+k); vals.append(sign*vH[k])
            rows.append(r); cols.append(T+1+jb); vals.append(-tot*ctx["tgt"][b]); b_ub.append(sign*tot*ctx["tgt"][b]); r+=1
    rows.append(r); cols.append(T); vals.append(1.0); b_ub.append(t1+tol); r+=1
    A_ub=sparse.coo_matrix((vals,(rows,cols)),shape=(r,ncols)).tocsr()
    c=np.zeros(ncols); c[T+1:]=1.0
    res=linprog(c,A_eq=A_eq,b_eq=b_eq,A_ub=A_ub,b_ub=np.array(b_ub),bounds=[(0,1)]*T+[(0,None)]*(1+nd),method="highs")
    if res.status!=0: return P1, float(t1)
    return res.x[:T].reshape(NQ,K,K), float(t1)

def esci_gain(S_gain):
    rng = np.random.default_rng(0); GAIN = {"E": 1.0, "S": S_gain}
    ex = esci_table()
    cnt = ex.groupby("query_id").size(); all_q = cnt[cnt>=K].index.to_numpy().copy()
    rng.shuffle(all_q); qs = all_q[:NQ]
    data = []
    for qi in qs:
        g = ex[ex.query_id==qi]
        sub = g.sample(K, random_state=int(qi)%2**31) if len(g)>K else g
        data.append((np.array([GAIN[l] for l in sub.esci_label]), sub.product_brand.to_numpy()))
    return data


out = {}
# ---- mixture-calibrated baselines on canonical data ----
data = esci_gain(0.5); ctx = make_ctx(data, 15); vH = ctx["vH"]; agents = ctx["agents"]; vs_all=[vH]+agents
vbarA = np.mean(np.stack(agents), axis=0)
for a in (0.25, 0.5):
    vmix = (1-a)*vH + a*vbarA
    P, _ = solve_lex(ctx, [vmix], vH)
    row = {"a0": U_of(ctx,data,P,vH),
           "a25": max(U_of(ctx,data,P,0.75*vH+0.25*v) for v in agents),
           "worst": max(U_of(ctx,data,P,v) for v in vs_all),
           "worst_curve": int(np.argmax([U_of(ctx,data,P,v) for v in vs_all]))}
    out[f"mixcal_{a}"] = row
    print(f"[mixcal a={a}] U a=0 {row['a0']:.3f} a=.25 {row['a25']:.3f} worst {row['worst']:.3f} (vertex idx {row['worst_curve']})", flush=True)
# ---- merit definition x premium ----
for name, Sg in (("graded_S0.5",0.5),("steep_S0.25",0.25),("binary_S1.0",1.0)):
    d = esci_gain(Sg); cx = make_ctx(d, 15); vh = cx["vH"]; ags = cx["agents"]
    P_H,UH = solve_lex(cx,[vh],vh); P_R,UR = solve_lex(cx,[vh]+ags,vh)
    prev=None; lam=None
    for a in np.linspace(0,0.8,81):
        uh = U_of(cx,d,P_H,vh) if a==0 else max(U_of(cx,d,P_H,(1-a)*vh+a*v) for v in ags)
        ur = U_of(cx,d,P_R,vh) if a==0 else max(U_of(cx,d,P_R,(1-a)*vh+a*v) for v in ags)
        if prev is not None and prev<0 and uh-ur>=0 and lam is None: lam=float(a)
        prev=uh-ur
    out[f"merit_{name}"] = {"U_H":float(UH),"U_R":float(UR),"premium":float(UR-UH),"lam":lam}
    print(f"[merit {name}] U_H={UH:.4f} U_R={UR:.4f} prem={UR-UH:.4f} lam={lam}", flush=True)
json.dump(out, open("mixcal_merit.json","w"), indent=1); print("saved mixcal_merit.json")
