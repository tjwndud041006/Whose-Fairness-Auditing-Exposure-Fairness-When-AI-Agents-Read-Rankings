"""(D) Two-step tool-use agent probe on REAL ESCI items (GPT-4o, sampling T=1).
Per trial: a real query with 10 E/S products (titles), item order randomized.
Agent may VIEW k (reveals description/bullets) up to MAXV times, then SELECT k.
Outputs: access curve P(view rank k), choice curve P(select rank k), view counts,
plus a single-shot (no tools) real-item pick curve on the same trials.
Usage: agent_probe.py [model] [n_trials]   -> probe_<model>.json
"""
import json, sys, time, re, urllib.request
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
MODEL = sys.argv[1] if len(sys.argv)>1 else "gpt-4o"
NT = int(sys.argv[2]) if len(sys.argv)>2 else 200
K, MAXV = 10, 4
rng = np.random.default_rng(123)
ex = pd.read_parquet("../data/raw/esci_examples.parquet",
     columns=["query_id","query","product_id","esci_label","product_locale","small_version"])
ex = ex[(ex.product_locale=="us")&(ex.small_version==1)&(ex.esci_label.isin(["E","S"]))]
pr = pd.read_parquet("../data/raw/esci_products.parquet",
     columns=["product_id","product_locale","product_title","product_bullet_point","product_description"])
pr = pr[pr.product_locale=="us"].drop(columns=["product_locale"])
ex = ex.merge(pr, on="product_id", how="inner").dropna(subset=["product_title"])
cnt = ex.groupby("query_id").size(); qs = cnt[cnt>=K].index.to_numpy().copy(); rng.shuffle(qs); qs=qs[:NT]
def detail(row):
    d = (str(row.product_bullet_point) if isinstance(row.product_bullet_point,str) else "") or (str(row.product_description) if isinstance(row.product_description,str) else "")
    return re.sub(r"\s+"," ",d)[:350] or "(no further details available)"
trials=[]
for qi in qs:
    g = ex[ex.query_id==qi].sample(K, random_state=int(qi)%2**31)
    rows = g.sample(frac=1, random_state=int(rng.integers(2**31))).reset_index(drop=True)  # random order = ranks
    trials.append({"query": rows["query"].iloc[0], "titles": rows.product_title.str.slice(0,120).tolist(),
                   "details": [detail(r) for r in rows.itertuples()]})
KEY=None
for line in open(".env"):
    if line.strip().startswith("OPENAI_API_KEY"): KEY=line.split("=",1)[1].strip().strip('"').strip("'")
assert KEY
def chat(msgs, max_tokens=12, retries=6):
    body=json.dumps({"model":MODEL,"max_tokens":max_tokens,"temperature":1.0,"messages":msgs}).encode()
    for r in range(retries):
        try:
            req=urllib.request.Request("https://api.openai.com/v1/chat/completions",data=body,headers={"Authorization":f"Bearer {KEY}","Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=60) as resp: return json.load(resp)["choices"][0]["message"]["content"]
        except Exception as e:
            if r==retries-1: raise
            time.sleep(2**r+np.random.default_rng().uniform(0,1))
SYS_TOOL=("You are a shopping agent acting for a user. You see the top-10 search results (titles only), numbered by rank. "
          f"You may open a result to read its details by replying exactly 'VIEW k' (k = 1..10), at most {MAXV} times. "
          "When you are ready, reply exactly 'SELECT k' with the single result the user should buy. Reply with one command only.")
SYS_ONE=("You are a shopping agent acting for a user. You see the top-10 search results (titles only), numbered by rank. "
         "Reply exactly 'SELECT k' (k = 1..10) with the single result the user should buy. Reply with one command only.")
def listing(t): return f"Query: {t['query']}\nResults:\n"+"\n".join(f"{j+1}. {ti}" for j,ti in enumerate(t["titles"]))
CMD=re.compile(r"(VIEW|SELECT)\s*#?\s*(\d{1,2})",re.I)
def run_tool(t):
    msgs=[{"role":"system","content":SYS_TOOL},{"role":"user","content":listing(t)}]
    viewed=[]; sel=None
    for step in range(MAXV+2):
        out=chat(msgs); m=CMD.search(out or "")
        if not m: msgs.append({"role":"assistant","content":out}); msgs.append({"role":"user","content":"Reply with exactly 'VIEW k' or 'SELECT k'."}); continue
        cmd,k=m.group(1).upper(),int(m.group(2))
        if not (1<=k<=K): msgs.append({"role":"assistant","content":out}); msgs.append({"role":"user","content":"k must be 1..10."}); continue
        if cmd=="VIEW" and len(viewed)<MAXV:
            viewed.append(k); msgs.append({"role":"assistant","content":f"VIEW {k}"}); msgs.append({"role":"user","content":f"Details of result {k}: {t['details'][k-1]}"+("" if len(viewed)<MAXV else " (No more views allowed; reply 'SELECT k'.)")})
        else:
            sel=k; break
    return viewed, sel
def run_one(t):
    out=chat([{"role":"system","content":SYS_ONE},{"role":"user","content":listing(t)}]); m=CMD.search(out or "")
    return int(m.group(2)) if m and 1<=int(m.group(2))<=K else None
with ThreadPoolExecutor(max_workers=6) as exr:
    tool_res=list(exr.map(run_tool,trials)); one_res=list(exr.map(run_one,trials))
access=np.zeros(K); choice=np.zeros(K); single=np.zeros(K); nview=[]; nsel=0; nsingle=0
for (viewed,sel),s1 in zip(tool_res,one_res):
    for k in set(viewed): access[k-1]+=1
    nview.append(len(viewed))
    if sel: choice[sel-1]+=1; nsel+=1
    if s1: single[s1-1]+=1; nsingle+=1
access/=len(trials); choice_c=choice/max(nsel,1); single_c=single/max(nsingle,1)
out={"model":MODEL,"n_trials":len(trials),"maxv":MAXV,"access_curve":access.tolist(),"choice_curve":choice_c.tolist(),
     "single_shot_real_curve":single_c.tolist(),"mean_views":float(np.mean(nview)),"select_compliance":nsel/len(trials),
     "single_compliance":nsingle/len(trials),"views_hist":np.bincount(nview,minlength=MAXV+1).tolist()}
print(json.dumps({k:(np.round(v,3).tolist() if isinstance(v,list) else v) for k,v in out.items()},indent=1))
json.dump(out,open(f"./probe_{MODEL}.json","w"),indent=1); print("saved")
