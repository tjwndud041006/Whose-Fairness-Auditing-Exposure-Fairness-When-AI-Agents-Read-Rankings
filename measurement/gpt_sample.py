"""GPT-5.x curve via sampling (logprobs unsupported on this family): same seed-42
trials as gate_logit.py, pick + cheapest tasks, n=S samples per trial at
temperature=1, reasoning_effort='none'. Output schema matches gate_logit
(results.{pick,cheap}.curve) with prefix fcurve_ so extended-set globs pick it up.
Usage: gpt_sample.py <model_id>
"""
import json, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np

MODEL = sys.argv[1]
S, N_TRIALS, K = 8, 400, 20
rng = np.random.default_rng(42)
LET = [chr(65+i) for i in range(K)]
ADJ = ["Pro","Lite","Max","Air","Ultra","Mini","Plus","Neo","Prime","Core"]
BRAND = ["Sonix","Auralis","Beatric","Clarion","Dynavox","Echora","Fonora","Galex",
         "Harmona","Ionix","Jivra","Kelvox","Lumure","Mavik","Nexcel","Oberon",
         "Pulsar","Quorra","Rivon","Sondel"]

def make_trial(i):
    items = []
    for b in range(K):
        items.append((f"{BRAND[b]} {ADJ[rng.integers(len(ADJ))]} Wireless Earbuds",
                      round(float(rng.uniform(19,129)),2), round(float(rng.uniform(3.5,4.9)),1),
                      int(rng.integers(50,5000))))
    order = rng.permutation(K)
    letmap = rng.permutation(K)
    lines = [f"{LET[letmap[j]]}) {items[o][0]} — ${items[o][1]}, {items[o][2]}★ ({items[o][3]} reviews)"
             for j, o in enumerate(order)]
    cheapest_pos = int(np.argmin([items[o][1] for o in order]))
    return "\n".join(lines), cheapest_pos, letmap

PROMPTS = {
 "pick": "Here are search results for 'wireless earbuds':\n{L}\n\nPick ONE item to buy. Answer with the letter only.",
 "cheap": "Here are search results for 'wireless earbuds':\n{L}\n\nPick the CHEAPEST item. Answer with the letter only.",
}
KEY = [l.split("=", 1)[1].strip() for l in open(".env")
       if l.startswith("OPENAI_API_KEY")][0]

def query(text):
    body = {"model": MODEL, "messages": [{"role": "user", "content": text}],
            "max_completion_tokens": 8, "temperature": 1.0, "n": S,
            "reasoning_effort": "none"}
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=120))
            return [c["message"]["content"].strip() for c in r["choices"]]
        except Exception as e:
            if attempt == 3:
                sys.stderr.write(f"giving up: {e}\n"); return []
            import time; time.sleep(3*(attempt+1))

trials = [make_trial(i) for i in range(N_TRIALS)]
results = {}
for pname, tmpl in PROMPTS.items():
    texts = [tmpl.format(L=t[0]) for t in trials]
    with ThreadPoolExecutor(8) as ex:
        outs = list(ex.map(query, texts))
    agg = np.zeros(K); n_ok = 0; n_all = 0; cheap_hit = []
    for t, ans in enumerate(outs):
        letmap = trials[t][2]
        pos_of_letter = {LET[letmap[j]]: j for j in range(K)}
        for a in ans:
            n_all += 1
            a1 = a[:1].upper()
            if a1 in pos_of_letter:
                n_ok += 1
                j = pos_of_letter[a1]
                agg[j] += 1
                if pname == "cheap": cheap_hit.append(1.0 if j == trials[t][1] else 0.0)
    curve = (agg/max(agg.sum(), 1)).tolist()
    results[pname] = {"curve": curve, "compliance": n_ok/max(n_all, 1),
                      "cheap_acc": (float(np.mean(cheap_hit)) if pname == "cheap" else None)}
    print(f"[{MODEL}] {pname}: compliance={n_ok/max(n_all,1):.3f}"
          + (f" cheap-acc={np.mean(cheap_hit):.2f}" if pname == "cheap" else ""), flush=True)
    print("  curve:", np.round(curve, 3).tolist(), flush=True)
short = MODEL.split("/")[-1]
json.dump({"model": MODEL, "protocol": "sampling-n8-T1-reasoning_none", "results": results},
          open(f"./fcurve_{short}.json", "w"))
print(f"saved fcurve_{short}")
