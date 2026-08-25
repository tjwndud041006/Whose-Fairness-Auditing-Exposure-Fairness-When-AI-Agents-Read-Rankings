"""Rank-vs-context dissociation: same seed-42 trials as gate_logit.py, but each line
carries an explicit rank marker, serialized either forward (rank 1 first — mirrors the
main protocol, control for the marker itself) or reversed (rank 20 first). The letter
->rank mapping is fixed per trial, so the choice distribution can be read over RANKS
in both variants. If curve(fwd) ~ curve(rev): the model keys on the rank marker.
If curve(fwd) ~ reverse(curve(rev)): the model keys on context position (serialization).
Pick task only (the curve used in the ambiguity set).
Usage: gate_logit_order.py <model_id> [--bs=N]
Output: ocurve_<model>.json  (does not join the lcurve_* ambiguity-set glob)
"""
import json, sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText

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
    lines = [f"{LET[letmap[j]]}) [rank {j+1}] {items[o][0]} — ${items[o][1]}, {items[o][2]}★ ({items[o][3]} reviews)"
             for j, o in enumerate(order)]
    return lines, letmap

TMPL = ("Here are search results for 'wireless earbuds' (each item shows its search rank):\n"
        "{L}\n\nPick ONE item to buy. Answer with the letter only.")

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
kw = dict(torch_dtype=torch.bfloat16, device_map="cuda:0")
try: model = AutoModelForCausalLM.from_pretrained(MODEL, **kw)
except Exception: model = AutoModelForImageTextToText.from_pretrained(MODEL, **kw)
model.eval()
IDS = [tok.encode(l, add_special_tokens=False)[-1] for l in LET]

trials = [make_trial(i) for i in range(N_TRIALS)]
results = {}
for variant in ("fwd", "rev"):
    texts = []
    for lines, letmap in trials:
        L = "\n".join(lines if variant == "fwd" else lines[::-1])
        texts.append(tok.apply_chat_template([{"role":"user","content":TMPL.format(L=L)}],
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

f = np.array(results["fwd"]["curve_over_ranks"]); r = np.array(results["rev"]["curve_over_ranks"])
r_rank = float(np.corrcoef(f, r)[0, 1])
r_ctx = float(np.corrcoef(f, r[::-1])[0, 1])
results["r_rank"] = r_rank; results["r_context"] = r_ctx
print(f"r(fwd,rev)={r_rank:+.3f} [rank-driven]   r(fwd,reversed(rev))={r_ctx:+.3f} [context-driven]")

short = MODEL.split("/")[-1]
json.dump({"model": MODEL, "results": results},
          open(f"./ocurve_{short}.json", "w"))
print(f"saved ocurve_{short}")
