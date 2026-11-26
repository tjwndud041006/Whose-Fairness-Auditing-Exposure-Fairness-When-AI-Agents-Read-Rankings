# Whose Fairness? Auditing Exposure Fairness When AI Agents Read Rankings — code and data

Reproduction bundle for the WSDM 2027 paper. It reproduces every table and figure of the
main paper (Tables 1–4, Figures 2–5) from

* the **frozen measured examination curves** of the eight LLM readers (`data/curves/`), and
* **compact derived copies** of the four public ranking testbeds (`data/derived/`),

with CPU-only linear programs. **No API or GPU call is made**: the curves were measured once (scripts and
prompts in `measurement/`) and are shipped as data. `run_all.sh` recomputes every analysis output from
scratch and ends with an automatic cell-by-cell comparison against the numbers printed in the paper
(`outputs/tables/check.txt`; the authors' run matched 106/106 cells).

```
release/
├── run_all.sh              one command: curves + derived data -> all analysis JSONs -> tables, figures, check
├── requirements.txt        numpy / scipy (HiGHS) / pandas / pyarrow / matplotlib (+ cvxpy for the P-MMF baseline)
├── code/                   analysis scripts (Python 3.12); analysis JSONs are written here by run_all.sh
├── data/curves/            measured curves (JSON): main set, direct K=10 (3 trial seeds), real-item, access/choice probes
├── data/derived/           compact dataset files built by code/prepare_data.py (see "Datasets")
├── measurement/            how the curves were measured (logit / log-prob / sampling readouts, tool-use probe); reference only
└── outputs/figs, tables    written by run_all.sh: figures (PDF), assembled tables (text), check report (Figure 1 schematic is shipped)
```

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh                 # all analyses (CPU LPs; hours, not minutes), figures, tables, check.  PY=python3.12 bash run_all.sh to pick an interpreter
cat outputs/tables/check.txt    # every Table 2-4 cell vs. the paper
```

Individual steps can be run from `code/` (`python main_tables.py`, …); `run_all.sh` lists them in dependency
order with the table/figure each one feeds.

## What each script reproduces

| Paper artifact | Script(s) | Output |
|---|---|---|
| Table 1 (curve set at K=20) | `table1_curves.py` | `outputs/tables/table1_curves.txt` |
| Table 2 (four domains) | `main_tables.py`, `rtb_main.py` (oracle / presence price), `rtb_domains.py` (RTB break-even), `main_frontier.py`, `boot_domains.py`, `steam_frontier.py` (bootstrap bands) | `outputs/tables/table2.txt` |
| Table 3 (ESCI policies; MAD / AWRF / Gini; P-MMF) | `main_esci.py`, `mixcal_merit.py`, `robust_abar.py`, `mad_table.py`, `awrf_gini.py`, `fill_sind.py` (Gini at a=0.25), `pmmf_steam.py` | `outputs/tables/table3.txt` |
| Table 4 (perturbations of the set), **as typeset** | `t4_bootstrap.py --all` then `t4_table4.py`: every row re-evaluated on the canonical workload and the 20 bootstrap workloads; cells are the canonical value ± the SD of the paired difference from the baseline | `outputs/tables/table4.tex` |
| Table 4, single-workload cross-check | `k10_analysis.py` (+`--prefix=k10s1/k10s2/real`), `k10_seeds_summary.py`, `construct_robust.py`, `eps_robust.py`, `tierb.py`, `review_fixes.py`, `fill_sind.py` (sign-indeterminacy rows) | `outputs/tables/table4.txt` |
| Figure 2 (curves; (top-3, r_H) space) | `fig2_instrument.py` | `outputs/figs/instrument.pdf` |
| Figure 3 (frontier, audit triad, certificate expiry) | `main_frontier.py`, `triad_window.py`, `rtb_main.py`, `fig_crossing_rtb.py` → `fig3_frontier.py` | `outputs/figs/frontier.pdf` |
| Figure 4 (U_max(a) vs agent share) | `main_tables.py`, `fig_crossing_rtb.py` → `fig4_crossing.py` | `outputs/figs/crossing.pdf` |
| Figure 5 (provider-level incidence, top-50) | `main_fig4_all.py` → `fig5_forest.py` | `outputs/figs/forest.pdf` |
| §5.2 query-level audit (27/200) | `main_esci.py` (`audit`) | `main_esci.json` |
| Figure 1 | hand-drawn schematic | `outputs/figs/overview.pdf` |

Shared machinery: `loaders.py` (the four workload constructions, fixed seeds), `curveset.py` (the frozen curve
manifest), `policy_structure4.py` (`build_canonical`: the LP with the MAD tie-break = the paper's canonical
policy), `core.py` (slots-formulation LPs: stage-1, lexicographic, robust tie-break; evaluation helpers).
Every policy LP is solved with HiGHS through `scipy.optimize.linprog` at default tolerances (1e-7); reported
zeros are optimal objective values at that tolerance.

## Datasets

`data/derived/` is built from the raw public datasets by `code/prepare_data.py`, which keeps only the rows
the paper's workload constructions can touch (canonical seeds and every bootstrap seed) while preserving raw
row order, so the loaders return byte-identical workloads. `python prepare_data.py --verify` (needs the raw
files, `RAW_DIR=...`) re-checks this for all 42 seeds; the check passed on the authors' copy of the raw data.

| Domain | Raw source | Derived file(s) |
|---|---|---|
| Amazon ESCI (brands) | `amazon-science/esci-data` (`shopping_queries_dataset_examples/products.parquet`) | `esci_es_us_small.parquet` — US, small version, E/S labels, joined with brand (472,718 rows) |
| TREC Fair Ranking 2022 (sub-continental regions) | `trec_2022_train_reldocs.jsonl`, `trec_2022_articles_discrete.parquet` | `trec_topics_ge30.jsonl`, `trec_articles_subset.parquet` (115,794 pages) |
| Steam (43 publishers) | FairDiverse toolkit, P-MMF Steam preprocessing (`ranking_scores.npz`, `iid2pid.json`, `steam.train`) | `steam/` (copied verbatim) |
| LastFM-360K (artists; MusicBrainz artist gender) | `LastFM360k-Le75.txt`, `LastFM360k-MB-artists.txt` | `lastfm/user_counts.parquet`, `plays_subset.parquet`, `artists_subset.tsv` |

Workloads: one canonical 200-query workload per domain (`loaders.py`, seeds 0 / 7 / 11 / 21), 20 bootstrap
ESCI query-workloads (`main_frontier.py`, seed 17), 20 bootstrap TREC/LastFM workloads (`boot_domains.py`,
seeds 1000–1019), 20 bootstrap Steam user-workloads (`steam_frontier.py`), five ESCI top-50 subsets
(`main_fig4_all.py`, seeds 100–104). Merit: graded ESCI gains (E=1, S=0.5), TREC predicted quality, Steam
MF score, LastFM log play count. The human curve is DCG (`1/log2(k+1)`) renormalized over K=10 positions.

## The measured curves (`data/curves/`)

Each JSON holds a model's position-conditioned choice distribution (`results.pick.curve`) over a K=20
(or K=10) results page of synthetic product items, item–position assignment randomized per trial and the
letter–position mapping redrawn per trial, plus the instrumental-task accuracy (`results.cheap.cheap_acc`).

| Files | Content |
|---|---|
| `lcurve_*`, `newcurve_*`, `cand_*`, `fcurve_*` | main set, K=20 (Table 1; `curveset.MAIN`); six older/smaller variants (`curveset.EXT`) are included only for the supplementary saturation check used by `main_esci.py` and `eps_robust.py` |
| `k10_*`, `k10s1_*`, `k10s2_*` | direct K=10 re-measurement, trial seeds 42 / 1 / 2 (Table 4 row 2) |
| `real_*` | re-measurement on real ESCI queries and product titles (Table 4 row 3) |
| `ccurve_*`, `acurve_*` | planted-item access curves and content-resampling influence curves (Table 4 row 4, §4 stress tests) |
| `probe_gpt-4o.json` | two-step tool-using agent on real ESCI queries (§4) |

Readouts: next-token logits over the label tokens for open-weight models (bf16, or nf4/4-bit where marked),
first-token log-probabilities for GPT-4o, a sampling protocol validated against the logit readout for the
GPT-5.6 tier (reasoning off). The exact prompts, serialization (explicit rank markers), model identifiers
and trial generation are in `measurement/` (`gate_logit*.py` for open models, `gpt_curve*.py` /
`gpt_sample*.py` for API models, `agent_probe.py` for the tool-use probe, `gate_causal.py` / `gate_ablate.py`
for the access/influence probes, `run_*.sh` for the trial-seed chains). These scripts need model weights or an
API key (read from the environment) and are provided for transparency; nothing in `run_all.sh` calls them.

## Reproducibility notes

* All LPs are deterministic; the only randomness is the workload construction (fixed seeds above) and the
  null curve sets in `tierb.py` (fixed seeds). `run_all.sh` regenerates every number in
  `outputs/tables/*.txt`; `check.txt` reports each Table 2–4 cell as PASS/FAIL against the paper.
* `run_all.sh` writes timestamps for each step to `outputs/run_all.log`; the bootstrap and null-set scripts
  (`main_frontier.py`, `triad_window.py`, `boot_domains.py`, `tierb.py`, `main_fig4_all.py`) dominate the runtime.
* Sign-indeterminacy in `k10_analysis.py --prefix=...` uses a re-solve at LP tolerance 1e-6 and can differ
  from the canonical Table 2 count by one provider on TREC22 (11 vs 10), as noted in the supplement.
* `pmmf_steam.py` is a line-by-line CPU port of the P-MMF authors' algorithm (dual mirror descent with a
  cvxpy prefix projection) run on the same 200-user Steam workload; it is the only script that needs cvxpy.

## License and citation

Code: released under the MIT License. Data files in `data/derived/` are derived
subsets of the respective public datasets and remain under their original terms (ESCI: Apache 2.0;
TREC Fair Ranking 2022 / Wikipedia: CC BY-SA; FairDiverse Steam preprocessing: its repository license;
LastFM-360K: research use). Measured curves in `data/curves/` are released with the code.
