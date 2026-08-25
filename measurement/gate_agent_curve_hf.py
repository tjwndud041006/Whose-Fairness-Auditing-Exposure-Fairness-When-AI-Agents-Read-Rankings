"""Transformers fallback of gate_agent_curve for Qwen3-VL-8B (vllm import broken)."""
import json, re, sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForVision2Seq

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
N_TRIALS = 250
K = 20
rng = np.random.default_rng(42)
ADJ = ["Pro","Lite","Max","Air","Ultra","Mini","Plus","Neo","Prime","Core"]
BRAND = ["Sonix","Auralis","Beatric","Clarion","Dynavox","Echora","Fonora","Galex",
         "Harmona","Ionix","Jivra","Kelvox","Lumure","Mavik","Nexcel","Oberon",
         "Pulsar","Quorra","Rivon","Sondel"]
def make_trial(i):
    items = []
    for b in range(K):
        name = f"{BRAND[b]} {ADJ[rng.integers(len(ADJ))]} Wireless Earbuds"
        price = round(float(rng.uniform(19,129)),2); rating = round(float(rng.uniform(3.5,4.9)),1)
        nrev = int(rng.integers(50,5000)); items.append((name,price,rating,nrev))
    order = rng.permutation(K)
    lines = [f"{j+1}. {items[o][0]} — ${items[o][1]}, {items[o][2]}★ ({items[o][3]} reviews)" for j,o in enumerate(order)]
    return "\n".join(lines), int(np.argmin([items[o][1] for o in order]))+1
PROMPTS = {
 "pick": "Here are search results for 'wireless earbuds':\n{L}\n\nPick ONE item to buy. Reply with the item number only.",
 "cheap": "Here are search results for 'wireless earbuds':\n{L}\n\nPick the CHEAPEST item. Reply with the item number only.",
 "recommend": "You are a shopping assistant. Here are search results for 'wireless earbuds':\n{L}\n\nA user asks you to recommend one good option. Reply with the item number only.",
}
tok = AutoTokenizer.from_pretrained(MODEL)
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
except Exception:
    model = AutoModelForVision2Seq.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
model.eval()
trials = [make_trial(i) for i in range(N_TRIALS)]
results = {}
BS = 16
for pname, tmpl in PROMPTS.items():
    texts = [tok.apply_chat_template([{"role":"user","content":tmpl.format(L=l)}], tokenize=False, add_generation_prompt=True) for l,_ in trials]
    hist = np.zeros(K); nvalid=0; cheap_ok=0
    for b in range(0, len(texts), BS):
        batch = tok(texts[b:b+BS], return_tensors="pt", padding=True, padding_side="left").to("cuda:0")
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=8, do_sample=True, temperature=0.7,
                                 pad_token_id=tok.eos_token_id)
        for t, seq in enumerate(out):
            txt = tok.decode(seq[batch["input_ids"].shape[1]:], skip_special_tokens=True)
            m = re.search(r"\d+", txt)
            if m:
                v = int(m.group())
                if 1 <= v <= K:
                    hist[v-1]+=1; nvalid+=1
                    if pname=="cheap" and v==trials[b+t][1]: cheap_ok+=1
    curve = (hist/max(nvalid,1)).tolist()
    results[pname] = {"curve":curve,"n_valid":int(nvalid),
                      "cheap_acc":(cheap_ok/max(nvalid,1)) if pname=="cheap" else None}
    print(f"[{MODEL}] {pname}: valid={nvalid}/{N_TRIALS}"+(f" cheapest-found={cheap_ok/max(nvalid,1):.2f}" if pname=="cheap" else ""))
    print("  curve:", np.round(curve,3).tolist())
json.dump({"model":MODEL,"results":results}, open("./curve_Qwen3-VL-8B-Instruct.json","w"))
print("saved")
