import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from config import FIGS; PILOT="."
plt.rcParams.update({"font.size":7,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150,"savefig.bbox":"tight"})
fr=json.load(open(f"{PILOT}/main_frontier.json")); tw=json.load(open(f"{PILOT}/triad_window.json"))["workload"]
taus=np.array(fr["taus"]); ck=np.array(fr["cert_known"]); ca=np.array(fr["cert_ambig"])
fig,(a1,a2,a3)=plt.subplots(3,1,figsize=(3.4,5.45),sharex=True,gridspec_kw={"hspace":0.30})
a1.step(taus,ck,where="post",color="#2471a3",lw=1.5,label="curve known ($v^H$)"); a1.step(taus,ca,where="post",color="#c0392b",lw=1.5,label="curve ambiguous (worst case over $\\mathcal{V}$)")
a1.axvspan(0.18,0.32,color="gold",alpha=0.18); a1.text(0.200,0.58,"certification\nwindow",ha="center",fontsize=5.5,color="#806000")
a1.set_ylabel("fraction of workloads\ncertifiable at $\\tau$ ($n{=}20$)",fontsize=6.5); a1.set_ylim(-0.03,1.05); a1.legend(fontsize=5.6,frameon=False,loc="lower right",bbox_to_anchor=(1.01,0.03),handlelength=1.3,handletextpad=0.35,borderaxespad=0.1)
a1.set_title("(a) what the curve decides: unanimous outside the window",fontsize=7)
T=sorted(float(t) for t in tw)
def get(t,k):
    for key in (str(t),f"{t:.2f}",str(round(t,2))):
        if key in tw: return tw[key][k]
    raise KeyError(t)
W=[get(t,"worst_acq")/20 for t in T]; G=[get(t,"geo_acq")/20 for t in T]; M=[get(t,"mono_acq")/20 for t in T]; Bl=[get(t,"blend_acq")/20 for t in T]
F=[ (get(t,"geo_false")[0]/get(t,"geo_false")[1]) if get(t,"geo_false")[1]>0 else np.nan for t in T]
a2.plot(T,G,"-o",ms=2.5,color="#c0392b",lw=1.3,label="existence: geometric decay (Viable-$\\Lambda$)"); a2.plot(T,M,"--",color="#c0392b",lw=1.0,label="existence: all monotone / unrestricted")
a2.plot(T,Bl,"-",color="0.55",lw=1.0,label="aggregate blend, $a{=}0.25$"); a2.plot(T,W,"-s",ms=2.5,color="#2471a3",lw=1.5,label="worst case over $\\mathcal{V}$ (ours)")
a2.axvspan(0.18,0.32,color="gold",alpha=0.18)
for t,f,gg in zip(T,F,G):
    if gg>0 and abs(t*100%4)<1e-6:
        if gg<0.95: a2.text(t-0.006,gg+0.02,f"{100*f:.0f}%",ha="right",va="bottom",fontsize=5.2,color="#c0392b")
        else: a2.text(t,1.05,f"{100*f:.0f}%",ha="center",fontsize=5.2,color="#c0392b")
a2.set_ylabel("fraction of workloads\nacquitted ($n{=}20$)",fontsize=6.5); a2.set_ylim(-0.03,1.16)
a2.set_title("(b) audit triad vs. $\\tau$; labels = share of existence acquittals that are false",fontsize=6.5)
a2.legend(fontsize=5.2,frameon=False,loc="center right",bbox_to_anchor=(1.0,0.62))
# (c) certificate expiry
mt=json.load(open(f"{PILOT}/main_tables.json"))["ESCI"]; sw=json.load(open(f"{PILOT}/rtb_sweep.json")); rm=json.load(open(f"{PILOT}/rtb_main.json"))["A2_top15"]
a=np.array([s[0] for s in mt["sweep"]]); h=np.array([s[1] for s in mt["sweep"]])
A=np.append(np.array(sw["a"]),1.0); R=np.append(np.array(sw["rtb"]),rm["rtb"]["wc_at1"])  # RTB sweep to 0.8 + measured a=1 endpoint
def cert(aa,u,t):
    ok=u<=t+1e-9
    return float(aa[ok].max()) if ok[0] else np.nan
tt=np.arange(0.14,0.4401,0.005)
lc=np.array([cert(a,h,t) for t in tt]); lr=np.array([cert(A,R,t) for t in tt])
f2=lambda v: np.array([np.nan if x is None else x for x in v],float)
a3.fill_between(taus,f2(fr["lam_p5"]),f2(fr["lam_p95"]),color="#c05050",alpha=0.18,lw=0,label="human-calibrated, mean and 5--95% over 20 workloads")
a3.plot(taus,f2(fr["lam_mean"]),color="#c05050",lw=1.0,ls=":")
a3.plot(tt,lc,color="#c05050",lw=1.6,label="human-calibrated (MAD tie-break), canonical")
a3.plot(tt,lr,color="#c08a30",lw=1.4,ls="-.",label="human-calibrated (RTB tie-break), canonical")
a3.axvspan(0.18,0.32,color="gold",alpha=0.18)
c026=rm["canonical"]["cert_026"]; r026=rm["rtb"]["cert_026"]
a3.plot([0.26,0.26],[c026,r026],color="k",lw=0.7,ls=":"); a3.scatter([0.26,0.26],[c026,r026],s=12,color=["#c05050","#c08a30"],zorder=5)
a3.text(0.267,c026-0.04,f"{c026:.2f}",fontsize=5.5,color="#c05050",va="top",ha="left",zorder=7,bbox=dict(boxstyle="square,pad=0.15",fc="white",ec="none",alpha=0.85)); a3.text(0.267,r026+0.01,f"{r026:.2f}",fontsize=5.5,color="#8a6020",va="bottom",zorder=7,bbox=dict(boxstyle="square,pad=0.15",fc="white",ec="none",alpha=0.85))
a3.set_xlabel("tolerance $\\tau$"); a3.set_ylabel("certificate expiry\n$\\lambda^{*}_{\\rm cert}(\\tau)$ (agent share)",fontsize=6.5); a3.set_ylim(-0.03,1.05)
a3.set_title("(c) certificate expiry: the tie-break moves it, at $\\tau{=}0.26$ from 0.47 to 0.83",fontsize=6.5)
a3.legend(fontsize=5.4,frameon=False,loc="upper center",bbox_to_anchor=(0.5,-0.32),ncol=1,handlelength=1.8,labelspacing=0.3)
fig.savefig(f"{FIGS}/frontier.pdf"); plt.close(fig); print("frontier ok", c026, r026)
