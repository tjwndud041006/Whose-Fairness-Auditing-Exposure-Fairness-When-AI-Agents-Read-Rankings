"""E1b: sampling-protocol curve (generate-and-parse) on the SAME trials as gate_logit
(seed 42), for the logit-vs-sampling correlation table.
Usage: gate_sample.py <model_id>   -> scurve_<model>.json
"""
import json, sys, re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText

MODEL = sys.argv[1]
N_TRIALS, K, S = 400, 20, 8
rng = np.random.default_rng(42)
LET = [chr(65+i) for i in range(K)]
ADJ = ["Pro","Lite","Max","Air","Ultra","Mini","Plus","Neo","Prime","Core"]
BRAND = ["Sonix","Auralis","Beatric","Clarion","Dynavox","Echora","Fonora","Galex",
         "Harmona","Ionix","Jivra","Kelvox","Lumure","Mavik","Nexcel","Oberon",
         "Pulsar","Quorra","Rivon","Sondel"]
def make_trial(i):
    items=[]
    for b in range(K):
        items.append((f"{BRAND[b]} {ADJ[rng.integers(len(ADJ))]} Wireless Earbuds",
                      round(float(rng.uniform(19,129)),2), round(float(rng.uniform(3.5,4.9)),1),
                      int(rng.integers(50,5000))))
    order = rng.permutation(K)
    letmap = rng.permutation(K)
    lines=[f"{LET[letmap[j]]}) {items[o][0]} — ${items[o][1]}, {items[o][2]}★ ({items[o][3]} reviews)"
           for j,o in enumerate(order)]
    cheapest_pos=int(np.argmin([items[o][1] for o in order]))
    return "\n".join(lines), cheapest_pos, letmap
PROMPT = "Here are search results for 'wireless earbuds':\n{L}\n\nPick ONE item to buy. Answer with the letter only."

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
kw = dict(torch_dtype=torch.bfloat16, device_map="cuda:0")
try: model = AutoModelForCausalLM.from_pretrained(MODEL, **kw)
except Exception: model = AutoModelForImageTextToText.from_pretrained(MODEL, **kw)
model.eval()

trials = [make_trial(i) for i in range(N_TRIALS)]
agg = np.zeros(K); n_ok = 0; n_tot = 0
BS = 4
texts = [tok.apply_chat_template([{"role":"user","content":PROMPT.format(L=t[0])}],
         tokenize=False, add_generation_prompt=True) for t in trials]
for b in range(0, len(texts), BS):
    batch = tok(texts[b:b+BS], return_tensors="pt", padding=True, padding_side="left").to("cuda:0")
    with torch.no_grad():
        gen = model.generate(**batch, do_sample=True, temperature=1.0, top_p=1.0,
                             max_new_tokens=3, num_return_sequences=S,
                             pad_token_id=tok.pad_token_id)
    new = gen[:, batch["input_ids"].shape[1]:]
    outs = tok.batch_decode(new, skip_special_tokens=True)
    for bi in range(len(texts[b:b+BS])):
        letmap = trials[b+bi][2]
        for s in range(S):
            n_tot += 1
            m = re.search(r"[A-T]", outs[bi*S+s])
            if not m: continue
            li = ord(m.group(0))-65
            pos = int(np.where(letmap == li)[0][0])
            agg[pos] += 1; n_ok += 1
curve = (agg/max(agg.sum(),1)).tolist()
short = MODEL.split("/")[-1]
json.dump({"model": MODEL, "protocol": "sampling", "n_samples": n_tot,
           "compliance": n_ok/max(n_tot,1), "curve": curve},
          open(f"./scurve_{short}.json", "w"))
print(f"[{MODEL}] sampling curve done: compliance={n_ok/max(n_tot,1):.3f}")
print("  curve:", np.round(curve,3).tolist())
