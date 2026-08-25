"""Assemble Tables 2-4 of the paper from the analysis JSONs and check every cell against the values
printed in the paper. Writes outputs/tables/table{2,3,4}.txt and outputs/tables/check.txt.
Run: python tables_assemble.py
"""
import json, numpy as np
from config import TABLES
L = lambda f: json.load(open(f))
mt, rm, fr, bd, sf, rd = L("main_tables.json"), L("rtb_main.json"), L("main_frontier.json"), L("boot_domains.json"), L("steam_frontier.json"), L("rtb_domains.json")
me, mx, ra, md, ag, pm = L("main_esci.json"), L("mixcal_merit.json"), L("robust_abar.json"), L("mad_table.json"), L("awrf_gini.json"), L("pmmf_steam.json")
rf, tb, ep, ks, kr, cr, fs = L("review_fixes.json"), L("tierb.json"), L("eps_robust.json"), L("k10_seeds_summary.json"), L("k10_analysis_real.json"), L("construct_robust.json"), L("fill_sind_gini.json")
checks = []
def chk(name, got, exp, tol):
    ok = (got is not None) and abs(got-exp) <= tol
    checks.append((name, got, exp, ok)); return ok
def f3(x): return f"{x:.3f}"[1:] if x < 1 else f"{x:.3f}"
def f2(x): return f"{x:.2f}"[1:]
# ---------------- Table 2 ----------------
DOMS = ["ESCI", "TREC22", "Steam", "LastFM"]
PAPER2 = {"ESCI": (.192,.231,.039,.007,.42,.48,4,2,.530,.936,.005), "TREC22": (0,.181,.181,.021,.33,.56,11,10,.555,.887,.001),
          "Steam": (0,.031,.031,0,.13,.59,11,14,.250,.931,.005), "LastFM": (0,.209,.209,.053,.42,.58,10,5,.498,.968,.001)}
t2 = ["Domain & U*_H & U* & premium [5-95%] & U*_orc-U*_H & lambda_break [5-95%] & RTB & s-ind H/R & worst type & NDCG_H / Delta_rob"]
for dom in DOMS:
    m = mt[dom]; d = rm["domains"][dom]
    if dom == "ESCI":
        pr = [r["U_R"]-r["U_H"] for r in fr["rows"]]; band_p = (np.percentile(pr,5), np.percentile(pr,95)); lb = rf["lam_band"]; band_l = (lb["p5"], lb["p95"])
        # paper: lambda bracket = range over workloads with positive premium
        lam_pos = [w for w in lb["per_workload"] if w is not None]; band_l = (min(lam_pos), max(lam_pos)) if lam_pos else band_l
    elif dom == "Steam":
        pr = [r["premium"] for r in sf["rows"]]; lm = [r["lam"] for r in sf["rows"] if r["lam"] is not None]
        band_p = (np.percentile(pr,5), np.percentile(pr,95)); band_l = (np.percentile(lm,5), np.percentile(lm,95))
    else:
        b = bd[dom]; band_p = (b["premium_p5"], b["premium_p95"]); band_l = (b["lam_p5"], b["lam_p95"])
    worst = rd[dom]["canonical"]["worst"]; rtb = rd[dom]["rtb"]["lam"]
    vals = (m["U_H"], m["U_rob"], m["premium"], d["presence_price"], m["lambda_break"], rtb, m["sign_ind_H"], m["sign_ind_R"], worst, m["ndcg_H"], m["delta_rob"])
    t2.append(f"{dom} & {f3(vals[0])} & {f3(vals[1])} & {f3(vals[2])} [{f2(band_p[0])},{f2(band_p[1])}] & {f3(vals[3])} & {f2(vals[4])} [{f2(band_l[0])},{f2(band_l[1])}] & {f2(vals[5])} & {vals[6]}/{vals[7]} & {f3(vals[8])} & {f3(vals[9])}/{f3(vals[10])}")
    for name, got, exp, tol in zip(["U_H","U_R","premium","presence","lam","lam_rtb","sind_H","sind_R","worst","ndcg","drob"], vals, PAPER2[dom], [.0006]*4+[.006,.006,.5,.5,.0006,.0006,.0006]):
        chk(f"T2 {dom} {name}", got, exp, tol)
# ---------------- Table 3 ----------------
tb3 = me["table"]
t3 = ["Policy & a=0 & a=0.25 & worst type"]
rows3 = [("Relevance sort", tb3["relevance-sort"]["a0"], tb3["relevance-sort"]["a25"], tb3["relevance-sort"]["worst"], (.287,.286,.464)),
         ("Human-calibrated", tb3["human-cal"]["a0"], tb3["human-cal"]["a25"], tb3["human-cal"]["worst"], (.192,.213,.530)),
         ("Average-curve", tb3["avg-curve"]["a0"], tb3["avg-curve"]["a25"], tb3["avg-curve"]["worst"], (.327,.304,.353)),
         ("Mixture-cal. (a=0.25)", mx["mixcal_0.25"]["a0"], mx["mixcal_0.25"]["a25"], mx["mixcal_0.25"]["worst"], (.192,.213,.530)),
         ("Mixture-cal. (a=0.5)", mx["mixcal_0.5"]["a0"], mx["mixcal_0.5"]["a25"], mx["mixcal_0.5"]["worst"], (.197,.211,.530)),
         ("Merit-robust", tb3["merit-robust"]["a0"], tb3["merit-robust"]["a25"], tb3["merit-robust"]["worst"], (.222,.231,.358)),
         ("Robust", tb3["robust"]["a0"], tb3["robust"]["a25"], tb3["robust"]["worst"], (.231,.231,.231)),
         ("Robust, share<=0.25", ra["0.25"]["a0"], ra["0.25"]["a25"], ra["0.25"]["worst"], (.197,.211,.530)),
         ("Oracle", tb3["oracle"]["a0"], None, tb3["oracle"]["worst"], (.192,None,.200))]
for name, a0, a25, w, exp in rows3:
    t3.append(f"{name} & {f3(a0)} & {'---' if a25 is None else f3(a25)} & {f3(w)}")
    for k, g, e in zip(["a0","a25","worst"], (a0,a25,w), exp):
        if e is not None: chk(f"T3 {name} {k}", g, e, .0006)
hm, rmad = md["human-MAD"], md["robust-MAD"]; aw = ag["TREC22"]["policies"]; gn = ag["Steam"]["policies"]
t3.append("Other indices, human-calibrated/robust")
t3.append(f"MAD, ESCI & {f3(hm['mad_a0'])}/{f3(rmad['mad_a0'])} & {f3(hm['mad_a25'])}/{f3(rmad['mad_a25'])} & {f3(hm['mad_worst'])}/{f3(rmad['mad_worst'])}")
t3.append(f"AWRF, TREC'22 & {aw['human_cal']['awrf_vH']:.3f}/{aw['robust']['awrf_vH']:.3f} & {aw['human_cal']['awrf_a25']:.3f}/{aw['robust']['awrf_a25']:.3f} & {aw['human_cal']['awrf_worst']:.3f}/{aw['robust']['awrf_worst']:.3f}")
t3.append(f"Gini, Steam & {f3(gn['human_cal']['gini_vH'])}/{f3(gn['robust']['gini_vH'])} & {f3(fs['steam_human_cal']['gini_a25'])}/{f3(fs['steam_robust']['gini_a25'])} & {f3(gn['human_cal']['gini_worst'])}/{f3(gn['robust']['gini_worst'])}")
pk = "pmmf_lambda_0.1_eta_0.001"; p = pm["results"][pk]
t3.append(f"P-MMF share, Steam & {p['MMF_vH']:.2f} & {p['MMF_a25']:.2f} & {p['MMF_worst']:.2f} ({p['MMF_worst_name']}); count-exposure MMF {p['MMF']:.2f}")
for name, g, e in [("MAD hum a0",hm["mad_a0"],.060),("MAD hum worst",hm["mad_worst"],.251),("MAD rob",rmad["mad_worst"],.132),("AWRF hum worst",aw["human_cal"]["awrf_worst"],.992),("AWRF rob worst",aw["robust"]["awrf_worst"],.997),
                   ("Gini hum worst",gn["human_cal"]["gini_worst"],.053),("Gini rob worst",gn["robust"]["gini_worst"],.014),("Gini hum a25",fs["steam_human_cal"]["gini_a25"],.014),("Gini rob a25",fs["steam_robust"]["gini_a25"],.007),
                   ("PMMF vH",p["MMF_vH"],.38),("PMMF a25",p["MMF_a25"],.42),("PMMF worst",p["MMF_worst"],.26),("PMMF count",p["MMF"],.60)]:
    chk(f"T3 {name}", g, e, .006 if name.startswith("PMMF") else .0006)   # P-MMF shares are printed with two decimals
# ---------------- Table 4 ----------------
e = ks["domains"]["ESCI"]; real = kr["domains"]["ESCI"]; loo = rf["loo"]; lofo = rm["LOFO"]
loo_p = [v["premium"] for v in loo.values()]+[v["premium"] for v in lofo.values()]; loo_l = [v["lam"] for v in loo.values()]
t4 = ["Curve set & premium & lambda_break & s-ind",
      f"Measured curves (baseline) & {f3(mt['ESCI']['premium'])} & {f2(mt['ESCI']['lambda_break'])} & {mt['ESCI']['sign_ind_H']}",
      f"Direct K=10 re-measurement, 3 seeds & {f3(e['premium_direct']['mean'])}+-{f3(e['premium_direct']['sd'])} & {f2(e['lam_direct']['mean'])}+-{f2(e['lam_direct']['sd'])} & {e['sind_direct']['mean']:.0f}+-{e['sind_direct']['sd']:.0f}",
      f"Real ESCI items (7 of 8 re-measured) & {f3(real['premium_direct'])} & {f2(real['lam_direct'])} & {real['sind_direct']}",
      f"Access curves in place of choice & {f3(cr['access']['premium'])} & {f2(cr['access']['lambda_break'])} & {cr['access']['sign_ind']}   (choice baseline {f3(cr['choice']['premium'])}/{f2(cr['choice']['lambda_break'])}/{cr['choice']['sign_ind']})",
      f"TV ball eps=0.05 & {f3(ep['eps']['0.05']['premium'])} & {f2(ep['eps']['0.05']['lam'])} & {fs['tvball05']}",
      f"Position-permuted curves (null) & {f3(tb['M5_perm']['prem_mean'])}+-{f3(tb['M5_perm']['prem_sd'])} & {f2(tb['M5_perm']['lam_mean'])}+-{f2(tb['M5_perm']['lam_sd'])} & {tb['M5_perm']['sind_mean']:.1f}+-{tb['M5_perm']['sind_sd']:.1f}",
      f"Dirichlet random curves (null) & {f3(tb['M5_dirichlet']['prem_mean'])}+-{f3(tb['M5_dirichlet']['prem_sd'])} & {f2(tb['M5_dirichlet']['lam_mean'])}+-{f2(tb['M5_dirichlet']['lam_sd'])} & {tb['M5_dirichlet']['sind_mean']:.1f}+-{tb['M5_dirichlet']['sind_sd']:.1f}",
      f"Leave one curve or one family out & {f3(min(loo_p))}--{f3(max(loo_p))} & {f2(min(loo_l))}--{f2(max(loo_l))} & {min(fs['loo'].values())}--{max(fs['loo'].values())}",
      f"Human side as a set (DCG, RBP.8, RBP.9) & {f3(rf['human_set']['premium_combined'])} & {f2(mt['ESCI']['lambda_break'])} & {fs['human_set']}",
      f"Mixture cap w=1/4 & {f3(tb['M2_box']['1/4']['premium'])} & {f2(tb['M2_box']['1/4']['lam'])} & {fs['cap14']}"]
for name, g, e_, tol in [("k10 premium",e["premium_direct"]["mean"],.050,.0006),("k10 lam",e["lam_direct"]["mean"],.47,.006),("real premium",real["premium_direct"],.037,.0006),("real lam",real["lam_direct"],.61,.006),("real sind",real["sind_direct"],4,.5),
                         ("access premium",cr["access"]["premium"],.051,.0006),("access lam",cr["access"]["lambda_break"],.38,.006),("access sind",cr["access"]["sign_ind"],3,.5),("tv premium",ep["eps"]["0.05"]["premium"],.064,.0006),("tv lam",ep["eps"]["0.05"]["lam"],.39,.006),("tv sind",fs["tvball05"],5,.5),
                         ("perm premium",tb["M5_perm"]["prem_mean"],.047,.0006),("perm lam",tb["M5_perm"]["lam_mean"],.56,.006),("dir premium",tb["M5_dirichlet"]["prem_mean"],.017,.0006),("dir lam",tb["M5_dirichlet"]["lam_mean"],.13,.006),("dir sind",tb["M5_dirichlet"]["sind_mean"],7.7,.06),
                         ("loo min premium",min(loo_p),.033,.0006),("loo sind",max(fs["loo"].values()),4,.5),("hset premium",rf["human_set"]["premium_combined"],.039,.0006),("hset sind",fs["human_set"],6,.5),("cap premium",tb["M2_box"]["1/4"]["premium"],.038,.0006),("cap lam",tb["M2_box"]["1/4"]["lam"],.50,.006),("cap sind",fs["cap14"],4,.5)]:
    chk(f"T4 {name}", g, e_, tol)
for name, rows in [("table2", t2), ("table3", t3), ("table4", t4)]:
    open(f"{TABLES}/{name}.txt", "w").write("\n".join(rows)+"\n"); print("\n".join(rows)); print()
rep = [f"{'PASS' if ok else 'FAIL'}  {n:32s} got={g if g is None else round(g,4)}  paper={e}" for n, g, e, ok in checks]
nfail = sum(1 for c in checks if not c[3])
rep.append(f"\n{len(checks)-nfail}/{len(checks)} cells match the paper")
open(f"{TABLES}/check.txt", "w").write("\n".join(rep)+"\n"); print("\n".join(rep[-1:])); print("see", f"{TABLES}/check.txt")
