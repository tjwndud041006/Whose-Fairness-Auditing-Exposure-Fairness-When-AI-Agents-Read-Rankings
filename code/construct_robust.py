"""Reviewer #1 (construct): does the ambiguity phenomenon depend on identifying
choice probability with examination probability? Recompute the ESCI core numbers
with ambiguity sets built from three constructs over the SAME six open models
(the ones with all measurements): (a) choice curves (pick), (b) access curves
(planted-cheapest recovery, normalized), (c) influence curves (ablation TV,
normalized; 2 models only -- reported as a partial check).
Canonical solve (lexicographic) for the deployed policy; report U*_H, U*_rob,
premium, lambda_break, sign-ind (top-15).
Run: python construct_robust.py
"""
import json, glob
import numpy as np
from loaders import esci_data, K
from config import CURVES
from policy_structure4 import build_canonical, widths_all

SIX = ["Qwen2.5-VL-7B-Instruct", "Qwen3-VL-8B-Instruct", "Qwen3-VL-32B-Instruct-bnb-4bit",
       "Mistral-7B-Instruct-v0.3", "Llama-3.1-8B-Instruct", "gemma-2-9b-it"]

def pick_curve(m):
    for pat in ("lcurve", "newcurve"):
        try:
            d = json.load(open(f"{CURVES}/{pat}_{m}.json"))
            c = np.array(d["results"]["pick"]["curve"], dtype=float)[:K]
            return c/c.sum()
        except FileNotFoundError:
            continue
    raise FileNotFoundError(m)

def access_curve(m):
    d = json.load(open(f"{CURVES}/ccurve_{m}.json"))
    c = np.array(d["acc_by_pos"], dtype=float)[:K]
    return c/c.sum()

def tv_curve(m):
    d = json.load(open(f"{CURVES}/acurve_{m}.json"))
    c = np.array(d["tv_by_pos"], dtype=float)[:K]
    return c/c.sum()

data = esci_data()
ctx = build_canonical(data, ntop=15)
vH = ctx["vH"]
NQ = len(data); top = np.array(ctx["top"]); tgt = ctx["tgt"]

def U_of(Ps, v):
    E = np.zeros(len(ctx["brands"]))
    for q, (m, bs) in enumerate(data):
        e = Ps[q] @ v
        np.add.at(E, [ctx["bidx"][b] for b in bs], e)
    return float(np.max(np.abs(E[top]/(NQ*v.sum()*tgt[top])-1)))

def run_set(name, agents):
    curves = [vH]+agents
    P_H, U_H = ctx["solve"]([vH])
    P_R, U_R = ctx["solve"](curves)
    def at_share(Ps, a):
        if a == 0: return U_of(Ps, vH)
        return max(U_of(Ps, (1-a)*vH + a*v) for v in agents)
    lam = None; prev = None
    for a in np.linspace(0, 0.8, 81):
        d = at_share(P_H, a) - at_share(P_R, a)
        if prev is not None and prev < 0 and d >= 0 and lam is None: lam = float(a)
        prev = d
    lo, hi = widths_all(P_H, data, ctx, curves, top)
    si = int(np.sum((lo < 0) & (hi > 0)))
    print(f"{name:12s} U*_H={U_H:.3f} U*_rob={U_R:.3f} prem={U_R-U_H:.3f} "
          f"lam_break={lam} sign-ind={si}/15 width={float((hi-lo).mean()):.3f}")
    return {"U_H": U_H, "U_rob": U_R, "premium": U_R-U_H, "lambda_break": lam,
            "sign_ind": si, "mean_width": float((hi-lo).mean())}

out = {}
out["choice"] = run_set("choice", [pick_curve(m) for m in SIX])
out["access"] = run_set("access", [access_curve(m) for m in SIX])
out["influence2"] = run_set("influence2", [tv_curve(m) for m in
                            ["Qwen2.5-VL-7B-Instruct", "Mistral-7B-Instruct-v0.3"]])
json.dump(out, open("construct_robust.json", "w"), indent=1)
print("saved construct_robust.json")
