"""gate_logit variant for MistralCommonTokenizer models (e.g. Mistral-Small-3.x):
build input ids via apply_chat_template(tokenize=True) (the only safe path for
mistral-common), left-pad manually, read next-token logits over label tokens.
Prints a 2-trial probe (top-5 tokens) before the full run so a broken template
is caught immediately.
Usage: gate_logit_mc.py <model_id> [--bs=N] [--prefix=cand]
"""
import json, sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig

MODEL = sys.argv[1]
BS = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--bs=")), "2"))
PREFIX = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--prefix=")), "cand")
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

PROMPTS = {
 "pick": "Here are search results for 'wireless earbuds':\n{L}\n\nPick ONE item to buy. Answer with the letter only.",
 "cheap": "Here are search results for 'wireless earbuds':\n{L}\n\nPick the CHEAPEST item. Answer with the letter only.",
 "recommend": "You are a shopping assistant. Here are search results for 'wireless earbuds':\n{L}\n\nA user asks you to recommend one good option. Answer with the letter only.",
}
tok = AutoTokenizer.from_pretrained(MODEL)
model = None
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="cuda:0",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16))
except Exception:
    model = AutoModelForImageTextToText.from_pretrained(MODEL, device_map="cuda:0",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16))
model.eval()
PAD = tok.pad_token_id if tok.pad_token_id is not None else 0
IDS = []
for l in LET:
    e = tok.encode(l, add_special_tokens=False)
    IDS.append(e[-1])

def encode(text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True)

def batch_logits(id_lists):
    mx = max(len(a) for a in id_lists)
    inp = torch.full((len(id_lists), mx), PAD, dtype=torch.long)
    att = torch.zeros((len(id_lists), mx), dtype=torch.long)
    for i, a in enumerate(id_lists):
        inp[i, mx-len(a):] = torch.tensor(a); att[i, mx-len(a):] = 1
    with torch.no_grad():
        out = model(input_ids=inp.to("cuda:0"), attention_mask=att.to("cuda:0")).logits[:, -1, :]
    return torch.softmax(out.float(), dim=-1).cpu().numpy()

trials = [make_trial(i) for i in range(N_TRIALS)]
# probe
probe = [encode(PROMPTS["pick"].format(L=trials[t][0])) for t in range(2)]
pr = batch_logits(probe)
for t in range(2):
    topi = np.argsort(-pr[t])[:5]
    print("probe top-5:", [(tok.decode([int(i)]), round(float(pr[t, i]), 3)) for i in topi], flush=True)
    print("label mass:", round(float(sum(pr[t, i] for i in IDS)), 3), flush=True)

results = {}
for pname, tmpl in PROMPTS.items():
    ids_all = [encode(tmpl.format(L=t[0])) for t in trials]
    agg = np.zeros(K); mass = []; cheap_prob = []
    for b in range(0, len(ids_all), BS):
        pr = batch_logits(ids_all[b:b+BS])
        for t in range(pr.shape[0]):
            p_let = np.array([pr[t, i] for i in IDS])
            mass.append(p_let.sum()); p_let = p_let/max(p_let.sum(), 1e-9)
            letmap = trials[b+t][2]
            p_pos = np.zeros(K)
            for j in range(K): p_pos[j] = p_let[letmap[j]]
            agg += p_pos
            if pname == "cheap": cheap_prob.append(p_pos[trials[b+t][1]])
    curve = (agg/len(ids_all)).tolist()
    results[pname] = {"curve": curve, "label_mass": float(np.mean(mass)),
                      "cheap_acc": (float(np.mean(cheap_prob)) if pname == "cheap" else None)}
    print(f"[{MODEL}] {pname}: label-mass={np.mean(mass):.3f}"
          + (f" cheap-prob={np.mean(cheap_prob):.2f}" if pname == "cheap" else ""), flush=True)
    print("  curve:", np.round(curve, 3).tolist(), flush=True)
short = MODEL.split("/")[-1]+"-nf4"
json.dump({"model": MODEL, "quant": "nf4", "results": results},
          open(f"./{PREFIX}_{short}.json", "w"))
print(f"saved {PREFIX}_{short}")
