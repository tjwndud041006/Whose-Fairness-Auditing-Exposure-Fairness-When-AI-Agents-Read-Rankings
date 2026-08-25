"""Regenerate crossing.pdf with the RTB policy added on the ESCI panel."""
import json, numpy as np, matplotlib
from scipy import sparse
from scipy.optimize import linprog
from loaders import LOADERS, K
from policy_structure4 import build_canonical
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from core import *
PILOT="."; from config import FIGS
data = LOADERS["ESCI"](); ctx = make_ctx(data, 15); vH=ctx["vH"]; agents=ctx["agents"]; vs_all=[vH]+agents
P_rtb,_,_,_ = rtb_solve(ctx,vH,vs_all)
AG=np.linspace(0,0.8,81)
rtb=[U_of(ctx,data,P_rtb,vH) if a==0 else max(U_of(ctx,data,P_rtb,(1-a)*vH+a*v) for v in agents) for a in AG]
json.dump({"a":AG.tolist(),"rtb":rtb},open(f"{PILOT}/rtb_sweep.json","w"))
mt=json.load(open(f"{PILOT}/main_tables.json"))
plt.rcParams.update({"font.size":7.5})
fig,axes=plt.subplots(1,2,figsize=(6.6,2.5))
for ax,dom,lab in ((axes[0],"ESCI","ESCI (shopping)"),(axes[1],"TREC22","TREC 2022 (Wikipedia)")):
    d=mt[dom]; a=[s[0] for s in d["sweep"]]; h=[s[1] for s in d["sweep"]]; r=[s[2] for s in d["sweep"]]
    ax.plot(a,h,color="#c05050",lw=1.6,label="human-calibrated (MAD tie-break)")
    if dom=="ESCI":
        ax.plot(AG,rtb,color="#c08a30",lw=1.4,ls="-.",label="human-calibrated (RTB tie-break)")
    ax.plot(a,r,color="#3a7ca5",lw=1.6,label="robust policy")
    ax.axhline(d["U_H"],color="k",ls=":",lw=0.9,label=r"oracle $U^{*}_{H}$")
    lb=d["lambda_break"]
    if lb is not None:
        ax.axvline(lb,color="gray",ls="--",lw=0.9)
        ymid=min(h)+0.55*(max(max(h),max(r))-min(h))
        ax.annotate(rf"$\lambda_{{\rm break}}={lb:.2f}$",(lb,ymid),xytext=(lb+0.03,ymid),fontsize=7,color="dimgray")
    if dom=="ESCI":
        ax.axvline(0.48,color="#c08a30",ls="--",lw=0.8)
        ax.annotate(r"RTB $0.48$",(0.48,min(h)+0.25*(max(h)-min(h))),xytext=(0.50,min(h)+0.25*(max(h)-min(h))),fontsize=6.5,color="#8a6020")
    ax.set_xlabel("agent traffic share $a$"); ax.set_title(lab,fontsize=8)
axes[0].set_ylabel(r"worst-case unfairness $U$")
axes[0].legend(fontsize=6,frameon=False,loc="upper left")
fig.tight_layout(); fig.savefig(f"{FIGS}/crossing.pdf"); print("saved crossing.pdf with RTB")
