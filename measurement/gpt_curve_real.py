"""R1: frontier-model curve via OpenAI API. Same trials as gate_logit (seed 42).
Uses top_logprobs=20 on the first completion token -> near-logit readout of the
letter distribution (comparable to the local logit protocol, plus a sampling check).
Reads OPENAI_API_KEY from .env.
Usage: gpt_curve.py [model] (default gpt-4o)
"""
import json, sys, time
import urllib.request
import numpy as np
from concurrent.futures import ThreadPoolExecutor

MODEL = next((a for a in sys.argv[1:] if not a.startswith("--")), "gpt-4o")
N_TRIALS, K = 400, 10
SEED = int(next((a.split("=",1)[1] for a in sys.argv if a.startswith("--seed=")), "42"))
PREFIX = next((a.split("=",1)[1] for a in sys.argv if a.startswith("--prefix=")), "real")
rng = np.random.default_rng(SEED)
from real_trials import build as build_real, PROMPTS as REAL_PROMPTS, LET
PROMPTS = {"pick": REAL_PROMPTS["pick"]}
KEY = None
for line in open(".env"):
    if line.strip().startswith("OPENAI_API_KEY"):
        KEY = line.split("=",1)[1].strip().strip('"').strip("'")
assert KEY, "no OPENAI_API_KEY in .env"

def call(prompt, retries=5):
    body = json.dumps({"model": MODEL, "max_tokens": 1, "temperature": 0,
                       "logprobs": True, "top_logprobs": 20,
                       "messages": [{"role":"user","content":prompt}]}).encode()
    for r in range(retries):
        try:
            req = urllib.request.Request("https://api.openai.com/v1/chat/completions",
                data=body, headers={"Authorization": f"Bearer {KEY}",
                                    "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except Exception as e:
            if r == retries-1: raise
            time.sleep(2**r + np.random.default_rng().uniform(0,1))

trials = build_real(seed=SEED if SEED!=42 else 123); N_TRIALS=len(trials)
results = {}
for pname, tmpl in PROMPTS.items():
    def one(t):
        d = call(tmpl.format(L=t[0]))
        top = d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        p_let = np.zeros(K)
        for e in top:
            s = e["token"].strip().upper()
            if len(s) == 1 and "A" <= s <= chr(64+K):
                p_let[ord(s)-65] += np.exp(e["logprob"])
        return p_let
    aggs = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, p_let in enumerate(ex.map(one, trials)):
            aggs.append(p_let)
            if (i+1) % 100 == 0: print(f"[{pname}] {i+1}/{N_TRIALS}", flush=True)
    agg = np.zeros(K); mass = []; cheap_prob = []
    for t, p_let in zip(trials, aggs):
        mass.append(p_let.sum())
        p = p_let/max(p_let.sum(), 1e-9)
        letmap = t[2]
        p_pos = np.zeros(K)
        for j in range(K): p_pos[j] = p[letmap[j]]
        agg += p_pos
        if pname == "cheap": cheap_prob.append(p_pos[t[1]])
    curve = (agg/len(trials)).tolist()
    results[pname] = {"curve": curve, "label_mass": float(np.mean(mass)),
                      "cheap_acc": float(np.mean(cheap_prob)) if pname=="cheap" else None}
    print(f"[{MODEL}] {pname}: label-mass={np.mean(mass):.3f}"
          + (f" cheap-prob={np.mean(cheap_prob):.2f}" if pname=="cheap" else ""))
    print("  curve:", np.round(curve, 3).tolist())
json.dump({"model": MODEL, "protocol": "api-top20-logprobs", "results": results},
          open(f"./{PREFIX}_{MODEL}.json", "w"))
print(f"saved {PREFIX}_"+MODEL)
