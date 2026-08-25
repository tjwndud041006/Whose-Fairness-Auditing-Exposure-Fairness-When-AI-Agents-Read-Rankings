"""REAL-ITEM variant (round-24): real ESCI queries/titles, letter labels, K=10. Logit-readout position-curve measurement: items labeled A..T (single tokens),
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
CPU = "--cpu" in sys.argv
DEV = "cpu" if CPU else "cuda:0"
PREFIX = next((a.split("=",1)[1] for a in sys.argv if a.startswith("--prefix=")), "real")
N_TRIALS, K = 400, 10
SEED = int(next((a.split("=",1)[1] for a in sys.argv if a.startswith("--seed=")), "42"))
rng = np.random.default_rng(SEED)
from real_trials import build as build_real, PROMPTS as REAL_PROMPTS, LET
PROMPTS = {"pick": REAL_PROMPTS["pick"]} if "--pick-only" in sys.argv else REAL_PROMPTS
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
kw = dict(torch_dtype=(torch.float32 if CPU else torch.bfloat16), device_map=DEV)
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
trials = build_real(seed=SEED if SEED!=42 else 123)
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
        batch=tok(texts[b:b+BS],return_tensors="pt",padding=True,padding_side="left").to(DEV)
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
