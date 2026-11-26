#!/usr/bin/env bash
# Reproduce every table and figure of the main paper from the frozen measured curves and the derived data.
# No API or GPU calls: the examination curves are read from data/curves (see measurement/ for how they were measured).
# Usage: bash run_all.sh            (from the repository root; ~1-2 h on one CPU core, mostly LP solves)
#        PY=/path/to/python bash run_all.sh
set -euo pipefail
PY=${PY:-python}
cd "$(dirname "$0")/code"
LOG=../outputs/run_all.log; : > "$LOG"
run() { echo "== $(date +%H:%M:%S) $*" | tee -a "$LOG"; "$PY" "$@" >> "$LOG" 2>&1 || { echo "FAILED: $*  (see $LOG)"; exit 1; }; }

# ---- Table 1: the measured curve set
run table1_curves.py
# ---- Table 2 (four domains) and Figure 4 sweeps
run main_tables.py            # U*_H, U*, premium, lambda_break, sign-indeterminacy, NDCG, U_max(a) sweeps
run main_frontier.py          # 20 bootstrap ESCI workloads: certification frontier, lambda*_cert(tau)   [Fig 3a,c; Table 2 band]
run rtb_main.py               # oracle / presence price, robust tie-break, top-50 incidence, hull distances, leave-one-family-out
run rtb_domains.py            # tie-break dependence of the break-even on all four domains       [Table 2 RTB column]
run boot_domains.py           # 20 bootstrap workloads for TREC22 and LastFM                        [Table 2 bands]
run steam_frontier.py         # 20 bootstrap user-workloads for Steam                              [Table 2 band]
run fig_crossing_rtb.py       # RTB U_max(a) sweep                                                  [Fig 4]
# ---- Section 5.2: audit triad
run triad_window.py           # existence audits vs worst-case verdict over the 20 workloads       [Fig 3b]
# ---- Table 3: ESCI policy table and other indices
run main_esci.py              # relevance / human-calibrated / average / merit-robust / robust / oracle; 27/200 query-level audit
run mixcal_merit.py           # mixture-calibrated baselines; merit-scale (binary / graded / steep)
run robust_abar.py            # share-bounded robust policy
run mad_table.py              # MAD replication
run awrf_gini.py              # AWRF (TREC22) and Gini (Steam)
run pmmf_steam.py             # P-MMF audited post hoc on the Steam workload
# ---- Table 4: perturbations of the measured set
run review_fixes.py           # leave-one-out, human-side set, existence-audit dilemma, lambda band
run tierb.py                  # mixture cap, permutation and Dirichlet nulls
run eps_robust.py             # TV-ball inflation
run k10_analysis.py --all                    # direct K=10 set, seed 42
run k10_analysis.py --prefix=k10s1 --all     # seed 1
run k10_analysis.py --prefix=k10s2 --all     # seed 2
run k10_seeds_summary.py
run k10_analysis.py --prefix=real --all      # real-item re-measurement
run construct_robust.py       # access vs choice curves
run fill_sind.py              # sign-indeterminacy for the remaining Table 4 rows; Steam Gini at a=0.25
# ---- Table 4 as typeset: every row re-evaluated on the canonical + 20 bootstrap workloads.
# ~1 h single-core. To parallelise, replace the --all line with one run per workload, e.g.
#   seq 0 19 | xargs -P 7 -I{} "$PY" t4_bootstrap.py --b={}   (plus --b=canon)
run t4_bootstrap.py --all     # all ten Table 4 rows per workload       -> t4_boot_<b>.json
run t4_table4.py              # canonical value +- paired SD vs baseline -> outputs/tables/table4.tex
# ---- Figure 5: provider-level incidence (five ESCI top-50 subsets)
run main_fig4_all.py
# ---- Figures 2-5
run fig2_instrument.py
run fig3_frontier.py
run fig4_crossing.py
run fig5_forest.py
# ---- Tables 2-4 assembled and checked against the paper
run tables_assemble.py
echo "done. tables: outputs/tables/  figures: outputs/figs/  check: outputs/tables/check.txt" | tee -a "$LOG"
