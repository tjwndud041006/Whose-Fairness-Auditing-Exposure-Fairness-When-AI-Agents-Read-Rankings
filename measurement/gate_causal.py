"""Causal examination proxy: P(find the cheapest | cheapest sits at rank k).
Same seed-42 trials as gate_logit.py, cheapest task only. If the model never
examines rank k, it cannot find the cheapest item placed there — so the
position-conditioned accuracy curve is a causal proxy for reading probability,
independent of the *selection* channel that the pick curve measures.
Output: ccurve_<model>.json with
  acc_by_pos[k]  = mean p(correct) over trials whose cheapest item sits at rank k
  n_by_pos[k]    = number of such trials (~400/20)
  pick_matrix    = 20x20: mean p(choose rank j | cheapest at rank k)
Usage: gate_causal.py <model_id> [--4bit] [--bs=N]
"""
import json, sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig

MODEL = sys.argv[1]
USE4BIT = "--4bit" in sys.argv
BS = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--bs=")), "4" if USE4BIT else "8"))
N_TRIALS, K = 400, 20
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

TMPL = "Here are search results for 'wireless earbuds':\n{L}\n\nPick the CHEAPEST item. Answer with the letter only."

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
kw = dict(torch_dtype=torch.bfloat16, device_map="cuda:0")
if USE4BIT:
    kw = dict(device_map="cuda:0",
              quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                  bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16))
try: model = AutoModelForCausalLM.from_pretrained(MODEL, **kw)
except Exception: model = AutoModelForImageTextToText.from_pretrained(MODEL, **kw)
model.eval()
IDS = [tok.encode(l, add_special_tokens=False)[-1] for l in LET]

trials = [make_trial(i) for i in range(N_TRIALS)]
texts = [tok.apply_chat_template([{"role":"user","content":TMPL.format(L=t[0])}],
         tokenize=False, add_generation_prompt=True) for t in trials]
acc_sum = np.zeros(K); n_pos = np.zeros(K); M = np.zeros((K, K)); mass = []
for b in range(0, len(texts), BS):
    batch = tok(texts[b:b+BS], return_tensors="pt", padding=True, padding_side="left").to("cuda:0")
    with torch.no_grad():
        logits = model(**batch).logits[:, -1, :]
    pr = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    for t in range(pr.shape[0]):
        p_let = np.array([pr[t, i] for i in IDS])
        mass.append(p_let.sum()); p_let = p_let/max(p_let.sum(), 1e-9)
        _, cpos, letmap = trials[b+t]
        p_pos = np.zeros(K)
        for j in range(K): p_pos[j] = p_let[letmap[j]]
        acc_sum[cpos] += p_pos[cpos]; n_pos[cpos] += 1; M[cpos] += p_pos
acc = (acc_sum/np.maximum(n_pos, 1)).tolist()
M = (M/np.maximum(n_pos, 1)[:, None]).tolist()
print(f"[{MODEL}] label-mass={np.mean(mass):.3f} overall-acc={acc_sum.sum()/n_pos.sum():.3f}")
print("  acc_by_pos:", np.round(acc, 2).tolist())
print("  n_by_pos:", n_pos.astype(int).tolist())
short = MODEL.split("/")[-1]+("-nf4" if USE4BIT else "")
json.dump({"model": MODEL, "acc_by_pos": acc, "n_by_pos": n_pos.astype(int).tolist(),
           "pick_matrix": M, "label_mass": float(np.mean(mass))},
          open(f"./ccurve_{short}.json", "w"))
print(f"saved ccurve_{short}")
