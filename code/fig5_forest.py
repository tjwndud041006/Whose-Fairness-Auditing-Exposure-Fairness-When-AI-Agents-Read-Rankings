import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from config import FIGS; PILOT="."
plt.rcParams.update({"font.size":7,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150,"savefig.bbox":"tight"})
allrows=json.load(open(f"{PILOT}/main_fig4_all.json"))["all_rows"]
widths_ind=[]; widths_det=[]; tot_c=0; tot_n=0
for rows in allrows:
    for r in rows:
        ind=r["dev_lo"]<0<r["dev_hi"]; (widths_ind if ind else widths_det).append(r["dev_hi"]-r["dev_lo"]); tot_c+=ind; tot_n+=1
rows=sorted(allrows[0],key=lambda r:-r["merit_share"])
fig,(ax,ax2)=plt.subplots(1,2,figsize=(3.4,2.15),gridspec_kw={"width_ratios":[2.2,0.75],"wspace":0.55})
c=0
for y,r in enumerate(rows):
    ind=r["dev_lo"]<0<r["dev_hi"]; c+=ind
    ax.plot([r["dev_lo"],r["dev_hi"]],[y,y],color="crimson" if ind else "#4878a8",lw=1.5,alpha=0.95 if ind else 0.75,solid_capstyle="butt"); ax.plot(r["dev_human"],y,"k.",ms=2.5)
ax.axvline(0,color="k",lw=0.7); ax.set_ylim(len(rows)-0.5,-0.5); ax.set_yticks([0,9,19,29,39,49]); ax.set_yticklabels([1,10,20,30,40,50])
ax.set_ylabel("provider (merit rank)"); ax.set_xlabel("deviation $d_b(v)$ over $\\mathcal{V}$")
ax.set_title(f"(a) one subset: {c}/50 sign-indet.;\npooled over five: {tot_c}/{tot_n}",fontsize=6.5)
rng=np.random.default_rng(0)
ax2.scatter(rng.uniform(-0.18,0.18,len(widths_det)),widths_det,s=5,color="#4878a8",alpha=0.6); ax2.scatter(1+rng.uniform(-0.18,0.18,len(widths_ind)),widths_ind,s=6,color="crimson",alpha=0.8)
for xx_,ws in ((0,widths_det),(1,widths_ind)): ax2.hlines(np.median(ws),xx_-0.25,xx_+0.25,color="k",lw=1.2)
ax2.axhline(0.05,color="0.4",ls=":",lw=0.8); ax2.text(1.3,0.06,"0.05",fontsize=5,color="0.4",ha="right")
ax2.set_xticks([0,1]); ax2.set_xticklabels([f"determ.\n(n={len(widths_det)})",f"indet.\n(n={len(widths_ind)})"],fontsize=5.8); ax2.set_ylabel("interval width (pooled)",fontsize=6.5)
mat=sum(1 for w in widths_ind if w>0.05); ax2.set_title(f"(b) widths; {mat}/{len(widths_ind)}\nindet. exceed 0.05",fontsize=6.5)
fig.savefig(f"{FIGS}/forest.pdf"); plt.close(fig); print("forest ok",tot_c,tot_n,mat)
