"""Assemble paper Table 4 (tab:stress) from the per-workload runs of t4_bootstrap.py.

Each perturbation row is reported as its value on the canonical ESCI workload plus/minus the SD of
its PAIRED difference from the baseline across the B=20 bootstrap workloads. The pairing removes the
workload effect, which is common to a perturbation and its baseline and is an order of magnitude
larger than any perturbation effect (baseline premium ranges 0 to .137 over the suite, and vanishes
on six workloads). The baseline row instead carries its own 5-95% band over the suite, the same
convention as the Table 2 brackets.

Writes outputs/tables/table4.tex (ready to \\input as tabs_stress.tex) and t4_table4.json.
Run: python t4_table4.py            (after t4_bootstrap.py --all)
"""
import json, glob, os, re
import numpy as np
from config import TABLES

runs = {}
for f in glob.glob("t4_boot_*.json"):
    runs[re.match(r"t4_boot_(.+)\.json$", os.path.basename(f)).group(1)] = json.load(open(f))
bs = sorted([t for t in runs if t != "canon"], key=int)
assert "canon" in runs, "missing t4_boot_canon.json -- run t4_bootstrap.py --all"
print(f"workloads: canonical + {len(bs)} bootstrap")
if len(bs) < 20:
    print(f"WARNING: only {len(bs)}/20 bootstrap workloads present; SDs are provisional")
fb = sorted({m for t in runs for s in runs[t]["rows"]["k10_seeds"].values()
             for m in s.get("fallback", [])})
print(f"direct K=10 curves falling back to the seed-42 file: {fb or 'none'}")
rfb = sorted({m for t in runs for m in runs[t]["rows"]["real_items"].get("fallback", [])})
print(f"real-item curves falling back to the synthetic-item file: {rfb or 'none'}")


def get(run, row, field):
    """One scalar per workload for a Table-4 row."""
    R = run["rows"]
    if row == "k10":                                    # mean over the three measurement seeds
        v = [R["k10_seeds"][t][field] for t in ("s42", "s1", "s2")]
        return None if any(x is None for x in v) else float(np.mean(v))
    if row in ("null_perm", "null_dirichlet"):          # mean over the null draws
        return R[row][field]["mean"]
    if row == "loo":                                    # mean over the 14 reduced sets
        v = [r[field] for r in R["loo"].values() if r[field] is not None]
        return float(np.mean(v)) if v else None
    return R[row][field]


ROWS = [
    ("baseline",            "Measured curves (baseline)"),
    ("k10",                 "Direct $K{=}10$ re-measurement, 3 seeds"),
    ("real_items",          "Real ESCI items"),
    ("access_six",          "Access curves in place of choice"),
    ("tv_ball_eps05",       "TV ball $\\varepsilon{=}0.05$ around every curve"),
    ("null_perm",           "Position-permuted curves (null)"),
    ("null_dirichlet",      "Dirichlet random curves (null)"),
    ("loo",                 "Leave one curve or one family out"),
    ("human_set",           "Human side as a set (DCG, RBP$_{.8}$, RBP$_{.9}$)"),
    ("mixture_cap_quarter", "Mixture cap $\\bar w=1/4$ on any agent curve"),
]
FIELDS = [("premium", 3), ("lam_break", 2), ("sind", 0)]


def fmt(x, nd):
    return f"{x:.{nd}f}".lstrip("0") if nd else f"{x:.0f}"


def paired_sd(key, fld):
    d = [get(runs[b], key, fld) - get(runs[b], "baseline", fld) for b in bs
         if get(runs[b], key, fld) is not None and get(runs[b], "baseline", fld) is not None]
    return (float(np.std(d, ddof=1)) if len(d) > 1 else None), len(d)


bp = [get(runs[b], "baseline", "premium") for b in bs]
band = f"[{fmt(max(float(np.percentile(bp, 5)), 0.0), 2)},{fmt(float(np.percentile(bp, 95)), 2)}]"

out, body = {}, []
for key, label in ROWS:
    out[key] = {"label": label}
    cells = []
    for fld, nd in FIELDS:
        canon = get(runs["canon"], key, fld)
        sd, n = paired_sd(key, fld)
        marg = [get(runs[b], key, fld) for b in bs]
        marg = [v for v in marg if v is not None]
        out[key][fld] = {"canon": canon, "paired_sd": sd, "n_paired": n,
                         "marginal_mean": float(np.mean(marg)) if marg else None,
                         "marginal_sd": float(np.std(marg, ddof=1)) if len(marg) > 1 else None,
                         "p5": float(np.percentile(marg, 5)) if marg else None,
                         "p95": float(np.percentile(marg, 95)) if marg else None}
        if key == "baseline":
            cells.append(f"${fmt(canon, nd)}$~{band}" if fld == "premium" else f"${fmt(canon, nd)}$")
        elif sd is None:
            cells.append(f"${fmt(canon, nd)}$")
        elif fld == "sind":
            cells.append(f"${canon:.0f}{{\\pm}}{sd:.1f}$")
        else:
            s = fmt(sd, nd)
            cells.append(f"${fmt(canon, nd)}{{\\pm}}{s}$" if float(s) > 0
                         else f"${fmt(canon, nd)}{{\\pm}}{{<}}{fmt(10**-nd, nd)}$")
    body.append(f"{label} & " + " & ".join(cells) + "\\\\")

CAPTION = ("Sensitivity to the ambiguity set, canonical ESCI workload.\n"
           "s-ind.: sign-indeterminate providers among the top 15 under\n"
           "human calibration.\n"
           "Each perturbation gives its canonical value $\\pm$ the SD of its\n"
           "paired difference from the baseline across the 20 bootstrap\n"
           "workloads, so the workload effect common to both cancels;\n"
           "brackets on the baseline are its own 5--95\\% band there.\n"
           "Nulls average 20 curve draws per workload and the\n"
           "leave-one-out row the 14 reduced sets.\n"
           "Real-item curves cover seven of the eight models\n"
           "(GPT-5.6-sol keeps its synthetic-item curve).\n"
           "Access comparison choice baseline: $.036/.41/4$.")

tex = ("% tabs_stress.tex -- Table 4 (tab:stress), input in 5.3.\n"
       "% Canonical-workload value +- SD of the paired difference from the baseline over the\n"
       "% B=20 bootstrap ESCI workloads (the main_frontier.py rng(17) suite, as the Table 2 bands).\n"
       "% Generated by t4_bootstrap.py --all then t4_table4.py; do not edit cells by hand.\n"
       "\\begin{table}[t]\n"
       f"\\caption{{{CAPTION}}}\n"
       "\\label{tab:stress}\n"
       "\\small\n"
       "\\setlength{\\tabcolsep}{3pt}\n"
       "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lccc@{}}\n"
       "\\toprule\n"
       "Curve set & premium & $\\lambda_{\\mathrm{break}}$ & s-ind.\\\\\n"
       "\\midrule\n" + "\n".join(body) + "\n"
       "\\bottomrule\n"
       "\\end{tabular*}\n"
       "\\end{table}\n")

print(f"\nbaseline premium 5-95% band over the suite: {band}")
print(f"{'row':46s} {'canon':>8s} {'paired sd':>10s}  {'marginal sd':>11s}")
for key, label in ROWS:
    for fld, nd in FIELDS:
        d = out[key][fld]
        ps = "--" if d["paired_sd"] is None else f"{d['paired_sd']:.4f}"
        ms = "--" if d["marginal_sd"] is None else f"{d['marginal_sd']:.4f}"
        print(f"  {label[:34]:34s} {fld:10s} {d['canon']:8.4f} {ps:>10s}  {ms:>11s}")
print()
print(tex)
json.dump({"band": band, "n_bootstrap": len(bs), "k10_fallback": fb, "real_fallback": rfb,
           "rows": out}, open("t4_table4.json", "w"), indent=1)
open(f"{TABLES}/table4.tex", "w", encoding="utf-8").write(tex)
print(f"saved t4_table4.json and {TABLES}/table4.tex")
