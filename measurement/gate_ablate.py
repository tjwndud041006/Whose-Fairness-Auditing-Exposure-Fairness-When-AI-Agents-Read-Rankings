"""Ablation-influence curve: I_k = TV( p(pick | x), p(pick | x with the rank-k item
resampled) ). If the model never reads rank k, replacing its content cannot move the
output distribution, so I_k ~ 0. Complements gate_causal.py (planted-target curve)
as a second counterfactual-influence measure that does not rely on a task with a
correct answer. First N_TRIALS trials of the seed-42 sequence, 21 forward passes per
trial (1 base + 20 single-position resamples).
Usage: gate_ablate.py <model_id> [--bs=N] [--n=100]
Output: acurve_<model>.json with tv_by_pos[k] = mean TV over trials.
"""
import json, sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText

MODEL = sys.argv[1]
BS = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--bs=")), "8"))
N_TRIALS = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--n=")), "100"))
K = 20
rng = np.random.default_rng(42)
LET = [chr(65+i) for i in range(K)]
ADJ = ["Pro","Lite","Max","Air","Ultra","Mini","Plus","Neo","Prime","Core"]
BRAND = ["Sonix","Auralis","Beatric","Clarion","Dynavox","Echora","Fonora","Galex",
         "Harmona","Ionix","Jivra","Kelvox","Lumure","Mavik","Nexcel","Oberon",
         "Pulsar","Quorra","Rivon","Sondel"]
ALT = ["Vextra","Wavion","Xelora","Yonda","Zephra"]

def make_items():
    items = []
    for b in range(K):
        items.append((f"{BRAND[b]} {ADJ[rng.integers(len(ADJ))]} Wireless Earbuds",
                      round(float(rng.uniform(19,129)),2), round(float(rng.uniform(3.5,4.9)),1),
                      int(rng.integers(50,5000))))
    order = rng.permutation(K)
    letmap = rng.permutation(K)
    return items, order, letmap

def render(items, order, letmap, repl=None):
    lines = []
    for j, o in enumerate(order):
        it = items[o] if (repl is None or j != repl[0]) else repl[1]
        lines.append(f"{LET[letmap[j]]}) {it[0]} — ${it[1]}, {it[2]}★ ({it[3]} reviews)")
    return "\n".join(lines)

TMPL = "Here are search results for 'wireless earbuds':\n{L}\n\nPick ONE item to buy. Answer with the letter only."

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
try: model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
except Exception: model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
model.eval()
IDS = [tok.encode(l, add_special_tokens=False)[-1] for l in LET]

def p_positions(texts, letmaps):
    out = []
    for b in range(0, len(texts), BS):
        batch = tok(texts[b:b+BS], return_tensors="pt", padding=True, padding_side="left").to("cuda:0")
        with torch.no_grad():
            logits = model(**batch).logits[:, -1, :]
        pr = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        for t in range(pr.shape[0]):
            p_let = np.array([pr[t, i] for i in IDS]); p_let /= max(p_let.sum(), 1e-9)
            lm = letmaps[b+t]
            out.append(np.array([p_let[lm[j]] for j in range(K)]))
    return out

tv_sum = np.zeros(K); n = 0
for t in range(N_TRIALS):
    items, order, letmap = make_items()
    # a fresh replacement item per position, alt brand so content genuinely changes
    repls = [(f"{ALT[rng.integers(len(ALT))]} {ADJ[rng.integers(len(ADJ))]} Wireless Earbuds",
              round(float(rng.uniform(19,129)),2), round(float(rng.uniform(3.5,4.9)),1),
              int(rng.integers(50,5000))) for _ in range(K)]
    texts = [tok.apply_chat_template([{"role":"user","content":TMPL.format(L=render(items,order,letmap))}],
             tokenize=False, add_generation_prompt=True)]
    for j in range(K):
        texts.append(tok.apply_chat_template([{"role":"user","content":TMPL.format(L=render(items,order,letmap,(j,repls[j])))}],
                     tokenize=False, add_generation_prompt=True))
    ps = p_positions(texts, [letmap]*(K+1))
    base = ps[0]
    for j in range(K):
        tv_sum[j] += 0.5*np.abs(ps[1+j]-base).sum()
    n += 1
    if (t+1) % 20 == 0: print(f"{t+1}/{N_TRIALS}", flush=True)
tv = (tv_sum/n).tolist()
print(f"[{MODEL}] tv_by_pos:", np.round(tv, 3).tolist())
short = MODEL.split("/")[-1]
json.dump({"model": MODEL, "tv_by_pos": tv, "n_trials": n},
          open(f"./acurve_{short}.json", "w"))
print(f"saved acurve_{short}")
