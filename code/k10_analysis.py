"""Round-17: direct K=10 main-set re-measurement vs K=20 truncation.
Per-model r/TV/top-3; hull diameter (pairwise TV) truncated vs direct; LP on four
domains with the direct set: U_H, U*_trunc, U*_direct, premium, lambda_break,
sign-indeterminacy, and transfer of the truncated-set robust policy to the direct set.
Run: python k10_analysis.py
"""
import json, sys, numpy as np
from scipy import sparse
from scipy.optimize import linprog
from loaders import LOADERS, K
from policy_structure4 import build_canonical
from curveset import load_set, MAIN
from core import make_ctx, eq_block, dev_rows_slots, stage1, U_of, intervals, rbp8, rbp9
from config import CURVES as PILOT
K10FILES = {
 "Qwen3-VL-8B":"k10_Qwen3-VL-8B-Instruct.json", "Qwen3-VL-32B (4bit)":"k10_Qwen3-VL-32B-Instruct-bnb-4bit.json",
 "Gemma-4-12B (nf4)":"k10_gemma-4-12B-it-nf4.json", "Llama-3.1-8B":"k10_Llama-3.1-8B-Instruct.json",
 "Mistral-Small-3.2 (nf4)":"k10_Mistral-Small-3.2-24B-Instruct-2506-nf4.json", "Phi-4-mini":"k10_Phi-4-mini-instruct.json",
 "GPT-4o":"k10_gpt-4o.json", "GPT-5.6-sol":"k10_gpt-5.6-sol.json"}
PREFIX = next((a.split("=",1)[1] for a in sys.argv if a.startswith("--prefix=")), "k10")
import os
trunc = load_set(10, "main"); direct = {}; meta = {}
for name, fn in K10FILES.items():
    fnp = fn.replace("k10_", PREFIX+"_", 1)
    if not os.path.exists(f"{PILOT}/{fnp}"): print("fallback to seed-42 file for", name, flush=True); fnp = fn
    try: d = json.load(open(f"{PILOT}/{fnp}"))
    except FileNotFoundError: print("MISSING", fn); continue
    c = np.array(d["results"]["pick"]["curve"], float); direct[name] = c/c.sum()
    meta[name] = {"cheap_acc": d["results"].get("cheap",{}).get("cheap_acc"), "label_mass": d["results"]["pick"].get("label_mass", d["results"]["pick"].get("compliance"))}
names = [n for n in trunc if n in direct]
if "--all" in sys.argv: assert len(names)==8, names
vH = 1/np.log2(np.arange(2, K+2)); vH /= vH.sum()
TV = lambda a,b: 0.5*float(np.abs(a-b).sum())
out = {"n_models": len(names), "per_model": {}}
print(f"{'model':26s} {'r':>6s} {'TV':>5s} {'top3 trunc':>10s} {'top3 dir':>8s} {'cheap':>6s}")
for n in names:
    t, d = trunc[n], direct[n]; r = float(np.corrcoef(t, d)[0,1])
    out["per_model"][n] = {"r": r, "TV": TV(t,d), "top3_trunc": float(t[:3].sum()), "top3_direct": float(d[:3].sum()),
                           "r_vH_trunc": float(np.corrcoef(t,vH)[0,1]), "r_vH_direct": float(np.corrcoef(d,vH)[0,1]), **meta[n]}
    print(f"{n:26s} {r:6.2f} {TV(t,d):5.2f} {t[:3].sum():10.3f} {d[:3].sum():8.3f} {str(meta[n]['cheap_acc'])[:5]:>6s}")
def hull(curves):
    cs = list(curves.values()); pw = [TV(a,b) for i,a in enumerate(cs) for b in cs[i+1:]]
    return {"max_pairTV": max(pw), "mean_pairTV": float(np.mean(pw)), "mean_TV_to_vH": float(np.mean([TV(c,vH) for c in cs])),
            "max_TV_to_vH": max(TV(c,vH) for c in cs), "top3_range": [min(float(c[:3].sum()) for c in cs), max(float(c[:3].sum()) for c in cs)]}
out["hull_trunc"] = hull({n:trunc[n] for n in names}); out["hull_direct"] = hull({n:direct[n] for n in names})
print("hull trunc :", {k:(np.round(v,3).tolist() if isinstance(v,list) else round(v,3)) for k,v in out["hull_trunc"].items()})
print("hull direct:", {k:(np.round(v,3).tolist() if isinstance(v,list) else round(v,3)) for k,v in out["hull_direct"].items()})

def solve_lex(ctx, curves, vH, tol=1e-6):
    P1,t1 = stage1(ctx, curves)
    NQ=ctx["NQ"]; T=NQ*K*K; nd=len(ctx["top"]); ncols=T+1+nd
    A_eq,b_eq=eq_block(NQ,ncols); rows,cols,vals,b_ub=[],[],[],[]; r=0
    for v in curves: r=dev_rows_slots(ctx,v,T,rows,cols,vals,b_ub,r)
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
def lam_break(ctx, data, P_H, vH, verts, U_ref):
    prev, lam = None, None
    for a in np.linspace(0, 1, 101):
        u = U_of(ctx,data,P_H,vH) if a == 0 else max(U_of(ctx,data,P_H,(1-a)*vH+a*v) for v in verts)
        if prev is not None and prev < 0 and u-U_ref >= 0 and lam is None: lam = float(a)
        prev = u-U_ref
    return lam
def sign_indet(ctx, data, P, vs):
    top=ctx["top"]; cnt=0
    for b in top:
        devs=[]
        for v in vs:
            E=0.0
            for (q,i) in ctx["slots"][top.index(b)]: E += float(P[q][i] @ v)
            devs.append(E/(ctx["NQ"]*v.sum()*ctx["tgt"][b])-1)
        if min(devs)<0<max(devs): cnt+=1
    return cnt
doms = ["ESCI","TREC22","Steam","LastFM"] if "--all" in sys.argv or "--domains" in sys.argv else ["ESCI"]
out["domains"]={}
for dom in doms:
    data = LOADERS[dom](); ctx = make_ctx(data, 15); vH = ctx["vH"]
    tr = [trunc[n] for n in names]; di = [direct[n] for n in names]
    P_H, U_H = solve_lex(ctx, [vH], vH)
    P_RT, U_RT = solve_lex(ctx, [vH]+tr, vH)
    P_RD, U_RD = solve_lex(ctx, [vH]+di, vH)
    row = {"U_H": U_H, "U_trunc": U_RT, "U_direct": U_RD, "premium_trunc": U_RT-U_H, "premium_direct": U_RD-U_H,
           "lam_trunc": lam_break(ctx,data,P_H,vH,tr,U_RT), "lam_direct": lam_break(ctx,data,P_H,vH,di,U_RD),
           "worst_PH_direct": max(U_of(ctx,data,P_H,v) for v in [vH]+di),
           "transfer_PRT_on_direct": max(U_of(ctx,data,P_RT,v) for v in [vH]+di),
           "transfer_PRD_on_trunc": max(U_of(ctx,data,P_RD,v) for v in [vH]+tr),
           "sind_trunc": sign_indet(ctx,data,P_H,[vH]+tr), "sind_direct": sign_indet(ctx,data,P_H,[vH]+di)}
    out["domains"][dom]=row
    print(dom, {k:(round(v,4) if isinstance(v,float) else v) for k,v in row.items()}, flush=True)
outfn = "k10_analysis.json" if PREFIX=="k10" else f"k10_analysis_{PREFIX}.json"
json.dump(out, open(outfn,"w"), indent=1); print("saved", outfn)
