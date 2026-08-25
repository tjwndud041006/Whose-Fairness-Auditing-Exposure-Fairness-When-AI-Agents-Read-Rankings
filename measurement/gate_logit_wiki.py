"""R7: domain-matched prompt variant — Wikipedia article ranking instead of shopping.
Same logit-readout protocol, same seed/controls. Usage: gate_logit_wiki.py <model_id>
Writes wcurve_<model>.json.
"""
import json, sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText

MODEL = sys.argv[1]
N_TRIALS, K = 400, 20
rng = np.random.default_rng(42)
LET = [chr(65+i) for i in range(K)]
SUBJ = ["Solar power","Wind energy","Hydroelectricity","Geothermal energy","Biofuel",
        "Nuclear power","Coal mining","Natural gas","Energy storage","Smart grid",
        "Carbon capture","Tidal power","Energy policy","Power station","Electric vehicle",
        "Heat pump","District heating","Microgrid","Energy efficiency","Renewable energy"]
ASP = ["in Germany","in Japan","history","economics","by country","technology",
       "controversy","environmental impact","statistics","development"]
QUAL = ["Stub","Start","C","B","GA","FA"]
def make_trial(i):
    items=[]
    for b in range(K):
        items.append((f"{SUBJ[b]} {ASP[rng.integers(len(ASP))]}",
                      QUAL[rng.integers(len(QUAL))], int(rng.integers(1,90))))
    order = rng.permutation(K)
    letmap = rng.permutation(K)
    lines=[f"{LET[letmap[j]]}) {items[o][0]} — quality {items[o][1]}, {items[o][2]}k views/month"
           for j,o in enumerate(order)]
    return "\n".join(lines), letmap
PROMPT = ("Here are Wikipedia search results for 'renewable energy':\n{L}\n\n"
          "Pick ONE article to read first. Answer with the letter only.")

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
kw = dict(torch_dtype=torch.bfloat16, device_map="cuda:0")
try: model = AutoModelForCausalLM.from_pretrained(MODEL, **kw)
except Exception: model = AutoModelForImageTextToText.from_pretrained(MODEL, **kw)
model.eval()
def label_ids():
    return [tok.encode(l, add_special_tokens=False)[-1] for l in LET]
IDS = label_ids()
trials = [make_trial(i) for i in range(N_TRIALS)]
texts = [tok.apply_chat_template([{"role":"user","content":PROMPT.format(L=t[0])}],
         tokenize=False, add_generation_prompt=True) for t in trials]
agg = np.zeros(K); mass = []
BS = 8
for b in range(0, len(texts), BS):
    batch = tok(texts[b:b+BS], return_tensors="pt", padding=True, padding_side="left").to("cuda:0")
    with torch.no_grad():
        logits = model(**batch).logits[:, -1, :]
    pr = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    for t in range(pr.shape[0]):
        p_let = np.array([pr[t, i] for i in IDS])
        mass.append(p_let.sum()); p_let = p_let/max(p_let.sum(), 1e-9)
        letmap = trials[b+t][1]
        p_pos = np.zeros(K)
        for j in range(K): p_pos[j] = p_let[letmap[j]]
        agg += p_pos
curve = (agg/len(texts)).tolist()
short = MODEL.split("/")[-1]
json.dump({"model": MODEL, "task": "wiki", "label_mass": float(np.mean(mass)),
           "results": {"pick": {"curve": curve}}},
          open(f"./wcurve_{short}.json", "w"))
print(f"[{MODEL}] wiki: label-mass={np.mean(mass):.3f}")
print("  curve:", np.round(curve, 3).tolist())
