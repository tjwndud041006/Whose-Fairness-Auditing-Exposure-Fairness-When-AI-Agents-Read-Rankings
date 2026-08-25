"""Seed replication summary of the direct K=10 re-measurement (paper Table 4 row 2; supplement Table S5).
Reads k10_analysis.json (trial seed 42), k10_analysis_k10s1.json (seed 1), k10_analysis_k10s2.json (seed 2)
and reports mean +- sd per domain of premium, lambda_break, sign-indeterminacy and the human-calibrated
policy's worst type on the direct set.
Run: python k10_seeds_summary.py
"""
import json, numpy as np
files = ["k10_analysis.json", "k10_analysis_k10s1.json", "k10_analysis_k10s2.json"]
runs = [json.load(open(f)) for f in files]
out = {"files": files, "domains": {}}
for dom in ["ESCI", "TREC22", "Steam", "LastFM"]:
    row = {}
    for key in ["premium_direct", "lam_direct", "sind_direct", "worst_PH_direct"]:
        vals = [r["domains"][dom][key] for r in runs]
        row[key] = {"vals": vals, "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))}
    out["domains"][dom] = row
    print(f"{dom:8s} premium {row['premium_direct']['mean']:.3f}+-{row['premium_direct']['sd']:.3f} [{runs[0]['domains'][dom]['premium_direct']:.3f}]"
          f"  lam {row['lam_direct']['mean']:.2f}+-{row['lam_direct']['sd']:.2f}"
          f"  s-ind {row['sind_direct']['mean']:.1f}+-{row['sind_direct']['sd']:.1f}"
          f"  worst {row['worst_PH_direct']['mean']:.3f}+-{row['worst_PH_direct']['sd']:.3f}")
json.dump(out, open("k10_seeds_summary.json", "w"), indent=1); print("saved k10_seeds_summary.json")
