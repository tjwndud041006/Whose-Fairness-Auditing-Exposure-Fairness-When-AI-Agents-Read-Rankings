"""Rank-vs-context dissociation: same seed-42 trials as gate_logit.py, but each line
carries an explicit rank marker, serialized either forward (rank 1 first — mirrors the
main protocol, control for the marker itself) or reversed (rank 20 first). The letter
->rank mapping is fixed per trial, so the choice distribution can be read over RANKS
in both variants. If curve(fwd) ~ curve(rev): the model keys on the rank marker.
If curve(fwd) ~ reverse(curve(rev)): the model keys on context position (serialization).
Pick task only (the curve used in the ambiguity set).
Usage: gate_logit_2x2.py <model_id> [--bs=N] [--4bit]
Output: x2curve_<model>.json  (2x2: marker on/off x fwd/rev)  (does not join the lcurve_* ambiguity-set glob)
"""
import json, sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig
USE4BIT = "--4bit" in sys.argv

MODEL = sys.argv[1]
BS = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--bs=")), "8"))
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
    letmap = rng.permutation(K)  # rank j is labeled LET[letmap[j]]
    body = [f"{items[o][0]} — ${items[o][1]}, {items[o][2]}★ ({items[o][3]} reviews)" for o in order]
    return body, letmap

def prompt_for(body, letmap, marker, rev):
    lines = [f"{LET[letmap[j]]}) " + (f"[rank {j+1}] " if marker else "") + d for j, d in enumerate(body)]
    if rev: lines = lines[::-1]
    head = "Here are search results for 'wireless earbuds'" + (" (each item shows its search rank)" if marker else "") + ":\n"
    return head + "\n".join(lines) + "\n\nPick ONE item to buy. Answer with the letter only."

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
kw = dict(torch_dtype=torch.bfloat16, device_map="cuda:0")
if USE4BIT:
    kw = dict(device_map="cuda:0", quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16))
try: model = AutoModelForCausalLM.from_pretrained(MODEL, **kw)
except Exception: model = AutoModelForImageTextToText.from_pretrained(MODEL, **kw)
model.eval()
IDS = [tok.encode(l, add_special_tokens=False)[-1] for l in LET]

trials = [make_trial(i) for i in range(N_TRIALS)]
results = {}
CONDS = [(m, r) for m in (0, 1) for r in (0, 1)]
for marker, rev in CONDS:
    variant = f"marker{marker}_{'rev' if rev else 'fwd'}"
    texts = []
    for body, letmap in trials:
        texts.append(tok.apply_chat_template([{"role":"user","content":prompt_for(body, letmap, marker, rev)}],
                     tokenize=False, add_generation_prompt=True))
    agg = np.zeros(K); mass = []
    for b in range(0, len(texts), BS):
        batch = tok(texts[b:b+BS], return_tensors="pt", padding=True, padding_side="left").to("cuda:0")
        with torch.no_grad():
            logits = model(**batch).logits[:, -1, :]
        pr = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        for t in range(pr.shape[0]):
            p_let = np.array([pr[t, i] for i in IDS])
            mass.append(p_let.sum()); p_let = p_let/max(p_let.sum(), 1e-9)
            letmap = trials[b+t][1]
            p_rank = np.zeros(K)
            for j in range(K): p_rank[j] = p_let[letmap[j]]
            agg += p_rank
    curve = (agg/len(texts)).tolist()
    results[variant] = {"curve_over_ranks": curve, "label_mass": float(np.mean(mass))}
    print(f"[{MODEL}] {variant}: label-mass={np.mean(mass):.3f}")
    print("  curve:", np.round(curve, 3).tolist())

def rr(a,b): return float(np.corrcoef(np.array(a),np.array(b))[0,1])
c = results
results["summary"] = {"r_fwd_rev_marker": rr(c["marker1_fwd"]["curve_over_ranks"], c["marker1_rev"]["curve_over_ranks"]),
  "r_fwd_revflip_marker": rr(c["marker1_fwd"]["curve_over_ranks"], c["marker1_rev"]["curve_over_ranks"][::-1]),
  "r_fwd_rev_nomarker": rr(c["marker0_fwd"]["curve_over_ranks"], c["marker0_rev"]["curve_over_ranks"]),
  "r_fwd_revflip_nomarker": rr(c["marker0_fwd"]["curve_over_ranks"], c["marker0_rev"]["curve_over_ranks"][::-1]),
  "r_marker_effect_fwd": rr(c["marker0_fwd"]["curve_over_ranks"], c["marker1_fwd"]["curve_over_ranks"]),
  "top3": {k: float(sum(v["curve_over_ranks"][:3])) for k, v in c.items()}}
print("summary:", json.dumps(results["summary"], indent=1))

short = MODEL.split("/")[-1] + ("-nf4" if USE4BIT else "")
json.dump({"model": MODEL, "results": results},
          open(f"./x2curve_{short}.json", "w"))
print(f"saved x2curve_{short}")
