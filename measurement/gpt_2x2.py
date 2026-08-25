"""(C) M4 2x2 for GPT-4o via API logprobs: serialization {fwd, rev} x rank marker {on, off}.
Seed-42 trials as gate_logit_order.py (K=20). Curves read over search RANKS.
Usage: gpt_2x2.py [model]   -> x2curve_<model>.json
"""
import json, sys, time, urllib.request
import numpy as np
from concurrent.futures import ThreadPoolExecutor
MODEL = sys.argv[1] if len(sys.argv)>1 else "gpt-4o"
N_TRIALS, K = 400, 20
rng = np.random.default_rng(42)
LET = [chr(65+i) for i in range(K)]
ADJ = ["Pro","Lite","Max","Air","Ultra","Mini","Plus","Neo","Prime","Core"]
BRAND = ["Sonix","Auralis","Beatric","Clarion","Dynavox","Echora","Fonora","Galex","Harmona","Ionix","Jivra","Kelvox","Lumure","Mavik","Nexcel","Oberon","Pulsar","Quorra","Rivon","Sondel"]
def make_trial(i):
    items=[]
    for b in range(K):
        items.append((f"{BRAND[b]} {ADJ[rng.integers(len(ADJ))]} Wireless Earbuds", round(float(rng.uniform(19,129)),2), round(float(rng.uniform(3.5,4.9)),1), int(rng.integers(50,5000))))
    order=rng.permutation(K); letmap=rng.permutation(K)
    body=[f"{items[o][0]} — ${items[o][1]}, {items[o][2]}★ ({items[o][3]} reviews)" for o in order]
    return body, letmap
trials=[make_trial(i) for i in range(N_TRIALS)]
KEY=None
for line in open(".env"):
    if line.strip().startswith("OPENAI_API_KEY"): KEY=line.split("=",1)[1].strip().strip('"').strip("'")
assert KEY
def call(prompt, retries=6):
    body=json.dumps({"model":MODEL,"max_tokens":1,"temperature":0,"logprobs":True,"top_logprobs":20,"messages":[{"role":"user","content":prompt}]}).encode()
    for r in range(retries):
        try:
            req=urllib.request.Request("https://api.openai.com/v1/chat/completions",data=body,headers={"Authorization":f"Bearer {KEY}","Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=60) as resp: return json.load(resp)
        except Exception as e:
            if r==retries-1: raise
            time.sleep(2**r+np.random.default_rng().uniform(0,1))
def prompt_for(body, letmap, marker, rev):
    lines=[]
    for j,desc in enumerate(body):
        tag=f"[rank {j+1}] " if marker else ""
        lines.append(f"{LET[letmap[j]]}) {tag}{desc}")
    if rev: lines=lines[::-1]
    head="Here are search results for 'wireless earbuds'"+(" (each item shows its search rank)" if marker else "")+":\n"
    return head+"\n".join(lines)+"\n\nPick ONE item to buy. Answer with the letter only."
results={}
for marker in (0,1):
    for rev in (0,1):
        cond=f"marker{marker}_{'rev' if rev else 'fwd'}"
        def one(t):
            body,letmap=t
            d=call(prompt_for(body,letmap,marker,rev))
            top=d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            p=np.zeros(K)
            for e in top:
                s=e["token"].strip().upper()
                if len(s)==1 and "A"<=s<="T": p[ord(s)-65]+=np.exp(e["logprob"])
            return p
        agg=np.zeros(K); mass=[]
        with ThreadPoolExecutor(max_workers=6) as ex:
            for t,p_let in zip(trials, ex.map(one,trials)):
                mass.append(p_let.sum()); p=p_let/max(p_let.sum(),1e-9)
                letmap=t[1]; pr=np.zeros(K)
                for j in range(K): pr[j]=p[letmap[j]]
                agg+=pr
        curve=(agg/len(trials)).tolist()
        results[cond]={"curve_over_ranks":curve,"label_mass":float(np.mean(mass)),"top3":float(sum(curve[:3]))}
        print(f"[{MODEL}] {cond}: mass={np.mean(mass):.3f} top3={sum(curve[:3]):.3f}", flush=True)
        print("  curve:",np.round(curve,3).tolist(), flush=True)
def r(a,b): return float(np.corrcoef(np.array(a),np.array(b))[0,1])
c=results
results["summary"]={"r_fwd_rev_marker":r(c["marker1_fwd"]["curve_over_ranks"],c["marker1_rev"]["curve_over_ranks"]),
 "r_fwd_revflip_marker":r(c["marker1_fwd"]["curve_over_ranks"],c["marker1_rev"]["curve_over_ranks"][::-1]),
 "r_fwd_rev_nomarker":r(c["marker0_fwd"]["curve_over_ranks"],c["marker0_rev"]["curve_over_ranks"]),
 "r_fwd_revflip_nomarker":r(c["marker0_fwd"]["curve_over_ranks"],c["marker0_rev"]["curve_over_ranks"][::-1]),
 "r_marker_effect_fwd":r(c["marker0_fwd"]["curve_over_ranks"],c["marker1_fwd"]["curve_over_ranks"])}
print("summary:",json.dumps(results["summary"],indent=1))
json.dump({"model":MODEL,"results":results},open(f"./x2curve_{MODEL}.json","w"))
print("saved")
