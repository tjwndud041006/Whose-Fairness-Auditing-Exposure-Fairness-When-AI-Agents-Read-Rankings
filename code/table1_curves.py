"""Table 1: the main-set ambiguity set at K=20 (top-3 mass, argmax rank, r_H, instrumental-task accuracy)
from the frozen measured curves in data/curves (curveset.MAIN).
Run: python table1_curves.py
"""
import json, numpy as np
from config import CURVES, TABLES
from curveset import MAIN
K = 20
vH = 1/np.log2(np.arange(2, K+2)); vH /= vH.sum()
lines = ["Model & Top-3 & argmax rank & r_H & Cheapest"]
out = {}
for fn, name in MAIN:
    d = json.load(open(f"{CURVES}/{fn}"))
    c = np.array(d["results"]["pick"]["curve"], float)[:K]; c = c/c.sum()
    cheap = d["results"].get("cheap", {}).get("cheap_acc")
    row = {"top3": float(c[:3].sum()), "argmax_rank": int(np.argmax(c))+1, "r_H": float(np.corrcoef(c, vH)[0, 1]), "cheapest": cheap}
    out[name] = row
    lines.append(f"{name} & {row['top3']:.2f} & {row['argmax_rank']} & {row['r_H']:+.2f} & {cheap if cheap is None else f'{cheap:.2f}'}")
txt = "\n".join(lines); print(txt)
open(f"{TABLES}/table1_curves.txt", "w").write(txt+"\n"); json.dump(out, open("table1_curves.json", "w"), indent=1)
