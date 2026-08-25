"""Frozen curve-set manifest (round-9 rebase). MAIN = 8 representative agents
(one per family/slot, contemporary where measurable); EXT = within-family and
prior-generation variants used only for generational and saturation analyses.
Llama 4 (109B MoE) and Mistral Small 4 (119B) exceed the 24GB measurement
budget; the largest measurable current models of those families stand in.
"""
import json
import numpy as np

from config import CURVES as PILOT

MAIN = [
    ("lcurve_Qwen3-VL-8B-Instruct.json",                    "Qwen3-VL-8B"),
    ("lcurve_Qwen3-VL-32B-Instruct-bnb-4bit.json",          "Qwen3-VL-32B (4bit)"),
    ("newcurve_gemma-4-12B-it-nf4.json",                    "Gemma-4-12B (nf4)"),
    ("newcurve_Llama-3.1-8B-Instruct.json",                 "Llama-3.1-8B"),
    ("cand_Mistral-Small-3.2-24B-Instruct-2506-nf4.json",   "Mistral-Small-3.2 (nf4)"),
    ("newcurve_Phi-4-mini-instruct.json",                   "Phi-4-mini"),
    ("fcurve_gpt-4o.json",                                  "GPT-4o"),
    ("fcurve_gpt-5.6-sol.json",                             "GPT-5.6-sol"),
]
EXT = [
    ("lcurve_Qwen2.5-VL-7B-Instruct.json",                  "Qwen2.5-VL-7B"),
    ("lcurve_Mistral-7B-Instruct-v0.3.json",                "Mistral-7B"),
    ("newcurve_gemma-2-9b-it.json",                         "Gemma-2-9B"),
    ("newcurve_gemma-4-E4B-it.json",                        "Gemma-4-E4B"),
    ("fcurve_gpt-5.6-terra.json",                           "GPT-5.6-terra"),
    ("fcurve_gpt-5.6-luna.json",                            "GPT-5.6-luna"),
]

def load_set(K, which="main"):
    files = MAIN if which == "main" else (EXT if which == "ext" else MAIN+EXT)
    out = {}
    for fn, name in files:
        d = json.load(open(f"{PILOT}/{fn}"))
        c = np.array(d["results"]["pick"]["curve"], dtype=float)[:K]
        assert c.sum() > 0, fn
        out[name] = c/c.sum()
    return out
