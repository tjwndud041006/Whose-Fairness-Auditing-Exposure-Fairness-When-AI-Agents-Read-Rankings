import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from curveset import load_set
from config import FIGS
plt.rcParams.update({"font.size":7,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150,"savefig.bbox":"tight"})
K=20; vH=1/np.log2(np.arange(2,K+2)); vH/=vH.sum(); x=np.arange(1,K+1)
MAIN={k:np.asarray(v,float) for k,v in load_set(K,"main").items()}
def short(n): return n.replace(" (nf4)","").replace(" (4bit)","").replace("-Instruct","")
def top3(c): c=np.asarray(c); return float(c[:3].sum()/c.sum())
def rH(c): return float(np.corrcoef(np.asarray(c),vH)[0,1])
ORDER=["Mistral-Small-3.2 (nf4)","GPT-5.6-sol","GPT-4o","Qwen3-VL-8B","Gemma-4-12B (nf4)","Phi-4-mini","Qwen3-VL-32B (4bit)","Llama-3.1-8B"]
STY={"Mistral-Small-3.2 (nf4)":("#6a3d9a","-"),"GPT-5.6-sol":("#c0392b","-"),"GPT-4o":("#c0392b","--"),"Qwen3-VL-8B":("#1f77b4","-"),
     "Gemma-4-12B (nf4)":("#2e7d32","-"),"Phi-4-mini":("#8c564b","-"),"Qwen3-VL-32B (4bit)":("#1f77b4","--"),"Llama-3.1-8B":("#ff7f0e","-")}
fig=plt.figure(figsize=(7.0,1.9)); gs=fig.add_gridspec(1,2,width_ratios=[1.3,1],wspace=0.3,left=0.06,right=0.99,top=0.88,bottom=0.2)
ax=fig.add_subplot(gs[0]); axb=fig.add_subplot(gs[1])
ax.plot(x,vH,"k--",lw=1.8,label="human (DCG)",zorder=6)
for n in ORDER:
    c,ls=STY[n]; ax.plot(x,MAIN[n],ls,color=c,lw=1.1,marker="o",ms=1.6,label=short(n))
ax.set_xlabel("rank position"); ax.set_xticks([1,5,10,15,20]); ax.set_ylabel("examination share $v_k$")
ax.set_title("(a) main-set curves ($K{=}20$) vs. human DCG",fontsize=6.8)
ax.legend(fontsize=5.4,frameon=False,ncol=3,loc="upper right",handlelength=2.0,columnspacing=0.9,handletextpad=0.4,labelspacing=0.3)
ax.set_ylim(0,0.33)
ax=axb
OFF={"Mistral-Small-3.2 (nf4)":(0.012,0.02,"left"),"Gemma-4-12B (nf4)":(0.012,0.02,"left"),"GPT-5.6-sol":(0.012,0.03,"left"),
     "Qwen3-VL-8B":(0.012,-0.10,"left"),"Phi-4-mini":(-0.010,0.06,"right"),"GPT-4o":(-0.010,-0.09,"right"),
     "Qwen3-VL-32B (4bit)":(0.012,0.05,"left"),"Llama-3.1-8B":(0.012,-0.12,"left")}
for n in ORDER:
    c,ls=STY[n]; ax.scatter(top3(MAIN[n]),rH(MAIN[n]),s=18,color=c,marker="o" if ls=="-" else "s",zorder=3)
    dx,dy,ha=OFF[n]; ax.text(top3(MAIN[n])+dx,rH(MAIN[n])+dy,short(n),fontsize=5.5,color=c,ha=ha,va="center")
ax.scatter(top3(vH),1.0,marker="*",s=60,color="k",zorder=4); ax.text(top3(vH)-0.01,0.93,"human DCG",fontsize=6,ha="right")
ax.axhline(0,color="0.8",lw=0.6); ax.set_xlabel("top-3 mass (position concentration)",fontsize=6.5); ax.set_ylabel("$r_H$ (shape correlation with human)",fontsize=6.5)
ax.set_xlim(0,0.42); ax.set_ylim(-1.0,1.1); ax.set_title("(b) main set in (top-3 mass, $r_H$) space: none near human DCG",fontsize=6.8)
fig.savefig(f"{FIGS}/instrument.pdf"); plt.close(fig); print("instrument ok")
