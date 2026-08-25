"""Logit-readout position-curve measurement: items labeled A..T (single tokens),
read the next-token distribution over the 20 label tokens directly. Compliance = 100%
by construction; no sampling noise; kills the selection-bias critique.
Usage: gate_logit.py <model_id> [--4bit] [--prefix=lcurve]
--4bit now applies real bnb nf4 quantization (for fp16-vs-4bit contrast).
--prefix controls the output filename prefix so variant runs do not join the
lcurve_* glob that downstream scripts treat as the ambiguity set.
"""
import json, sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig

MODEL = sys.argv[1]
USE4BIT = "--4bit" in sys.argv
PREFIX = next((a.split("=",1)[1] for a in sys.argv if a.startswith("--prefix=")), "k10")
N_TRIALS, K = 400, 10
SEED = int(next((a.split("=",1)[1] for a in sys.argv if a.startswith("--seed=")), "42"))
rng = np.random.default_rng(SEED)
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
    letmap = rng.permutation(K)  # position j is labeled LET[letmap[j]]
    lines=[f"{LET[letmap[j]]}) {items[o][0]} — ${items[o][1]}, {items[o][2]}★ ({items[o][3]} reviews)"
           for j,o in enumerate(order)]
    cheapest_pos=int(np.argmin([items[o][1] for o in order]))
    return "\n".join(lines), cheapest_pos, letmap
PROMPTS = {
 "pick":"Here are search results for 'wireless earbuds':\n{L}\n\nPick ONE item to buy. Answer with the letter only.",
 "cheap":"Here are search results for 'wireless earbuds':\n{L}\n\nPick the CHEAPEST item. Answer with the letter only.",
 "recommend":"You are a shopping assistant. Here are search results for 'wireless earbuds':\n{L}\n\nA user asks you to recommend one good option. Answer with the letter only.",
}
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
# label token ids: try "A" and " A" variants, pick the one the model would emit after template
def label_ids(prefix=""):
    ids=[]
    for l in LET:
        t = tok.encode(prefix+l, add_special_tokens=False)
        ids.append(t[-1] if len(t)>=1 else None)
    return ids
IDS_bare = label_ids("")
trials = [make_trial(i) for i in range(N_TRIALS)]
results={}
BS = int(next((a.split("=",1)[1] for a in sys.argv if a.startswith("--bs=")), "4" if USE4BIT else "8"))
def chat(text):
    msgs = [{"role":"user","content":text}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except (ValueError, TypeError):
        return tok.apply_chat_template(msgs, tokenize=False)
for pname,tmpl in PROMPTS.items():
    texts=[chat(tmpl.format(L=t[0])) for t in trials]
    agg=np.zeros(K); mass=[]; cheap_prob=[]; letter_agg=np.zeros(K)
    for b in range(0,len(texts),BS):
        batch=tok(texts[b:b+BS],return_tensors="pt",padding=True,padding_side="left").to("cuda:0")
        with torch.no_grad():
            logits=model(**batch).logits[:,-1,:]
        pr=torch.softmax(logits.float(),dim=-1).cpu().numpy()
        for t in range(pr.shape[0]):
            p_let=np.array([pr[t,i] for i in IDS_bare])
            mass.append(p_let.sum()); p_let=p_let/max(p_let.sum(),1e-9)
            letter_agg+=p_let
            letmap=trials[b+t][2]
            p_pos=np.zeros(K)
            for j in range(K): p_pos[j]=p_let[letmap[j]]
            agg+=p_pos
            if pname=="cheap": cheap_prob.append(p_pos[trials[b+t][1]])
    curve=(agg/len(texts)).tolist()
    results[pname]={"curve":curve,"label_mass":float(np.mean(mass)),
                    "letter_bias":(letter_agg/len(texts)).tolist(),
                    "cheap_acc":(float(np.mean(cheap_prob)) if pname=="cheap" else None)}
    print(f"[{MODEL}] {pname}: label-mass={np.mean(mass):.3f}"
          +(f" cheap-prob={np.mean(cheap_prob):.2f}" if pname=="cheap" else ""))
    print("  curve:", np.round(curve,3).tolist())
short=MODEL.split("/")[-1]+("-nf4" if USE4BIT else "")
json.dump({"model":MODEL,"quant":"nf4" if USE4BIT else "bf16","results":results},
          open(f"./{PREFIX}_{short}.json","w"))
print(f"saved {PREFIX}_{short}")
