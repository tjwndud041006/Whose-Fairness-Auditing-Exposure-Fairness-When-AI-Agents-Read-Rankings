"""Figure 4: worst-case unfairness U_max(a) versus agent share on ESCI (TREC 2022 inset).
Inputs: main_tables.json (canonical human-calibrated and robust sweeps), rtb_sweep.json (RTB tie-break sweep).
Run: python fig4_crossing.py
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from config import FIGS
plt.rcParams.update({"font.size":7.5,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150,"savefig.bbox":"tight"})
sw=json.load(open("rtb_sweep.json")); AG=sw["a"]; rtb=sw["rtb"]; mt=json.load(open("main_tables.json"))
fig,ax=plt.subplots(figsize=(3.4,2.7))
d=mt["ESCI"]; a=[s[0] for s in d["sweep"]]; h=[s[1] for s in d["sweep"]]; r=[s[2] for s in d["sweep"]]
ax.plot(a,h,color="#c05050",lw=1.8,label="human-calibrated (MAD tie-break)"); ax.plot(AG,rtb,color="#c08a30",lw=1.5,ls="-.",label="human-calibrated (RTB tie-break)")
ax.plot(a,r,color="#3a7ca5",lw=1.8,label="robust policy (Thm. 3.5)"); ax.axhline(d["U_H"],color="k",ls=":",lw=0.9,label="human-only floor $U^*_H$")
ax.axvline(0.42,color="gray",ls="--",lw=0.9); ax.text(0.415,0.268,"$\\lambda_{\\rm break}=0.42$",ha="right",fontsize=7,color="dimgray")
ax.axvline(0.48,color="#c08a30",ls="--",lw=0.8); ax.text(0.49,0.200,"RTB $0.48$",fontsize=6.5,color="#8a6020")
ax.set_xlabel("agent traffic share $a$"); ax.set_ylabel("worst-case unfairness $U_{\\max}(a)$"); ax.set_title("ESCI (inset: TREC 2022, $U^*_H{=}0$)",fontsize=8); ax.legend(fontsize=6,frameon=False,loc="upper center",bbox_to_anchor=(0.5,-0.22),ncol=2,columnspacing=1.0,handletextpad=0.4)
ins=ax.inset_axes([0.15,0.56,0.33,0.38]); dt=mt["TREC22"]; at=[s[0] for s in dt["sweep"]]; ht=[s[1] for s in dt["sweep"]]; rt=[s[2] for s in dt["sweep"]]
ins.plot(at,ht,color="#c05050",lw=1.2); ins.plot(at,rt,color="#3a7ca5",lw=1.2); ins.axvline(dt["lambda_break"],color="gray",ls="--",lw=0.7)
ins.set_title("TREC 2022",fontsize=6); ins.tick_params(labelsize=5,pad=1.5); ins.set_xticks([0,0.33,0.8]); ins.set_xticklabels(["0",".33",".8"]); ins.set_yticks([0,0.2,0.4]); ins.set_yticklabels(["0",".2",".4"])
fig.tight_layout(); fig.savefig(f"{FIGS}/crossing.pdf"); plt.close(fig); print("crossing ok")
