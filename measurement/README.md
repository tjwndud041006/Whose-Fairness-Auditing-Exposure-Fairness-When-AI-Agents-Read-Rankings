# Curve measurement scripts (reference only; not run by run_all.sh)

These scripts produced the JSON files in `data/curves/`. They require model weights (open models, run with
HuggingFace transformers on a GPU, bf16 or 4-bit as marked in Table 1) or an OpenAI API key (read from the
`OPENAI_API_KEY` environment variable / a local `.env`; never stored in code).

| Script | Measurement |
|---|---|
| `gate_logit.py` | K=20 synthetic-item pages, next-token logit readout over label tokens (pick / cheapest tasks) -> `lcurve_*`, `newcurve_*`, `cand_*` |
| `gate_logit_k10.py`, `run_k10_seeds.sh` | direct K=10 re-measurement, trial seeds 42/1/2 -> `k10_*`, `k10s1_*`, `k10s2_*` |
| `gate_logit_real.py`, `real_trials.py`, `run_real_chain.sh` | real ESCI queries and product titles (letter-label instrument) -> `real_*` |
| `gate_logit_2x2.py`, `gate_logit_order.py` | serialization x rank-marker 2x2 and order controls |
| `gate_logit_wiki.py`, `gate_logit_mc.py`, `gate_sample.py` | item-domain, multiple-choice and sampling-readout controls |
| `gate_causal.py`, `gate_ablate.py` | planted-item access curves (`ccurve_*`) and content-resampling influence curves (`acurve_*`) |
| `gate_agent_curve_hf.py` | early single-model curve harness |
| `gpt_curve.py`, `gpt_sample.py`, `gpt_sample_4o.py` | API models: first-token log-probabilities (GPT-4o) and the sampling protocol (GPT-5.6 tier) -> `fcurve_*` |
| `gpt_curve_k10.py`, `gpt_sample_k10.py`, `gpt_curve_real.py`, `gpt_2x2.py` | API-model K=10, real-item and serialization runs |
| `agent_probe.py` | two-step tool-using agent (`view(k)` then `select(k)`) on real ESCI queries -> `probe_gpt-4o.json` |

Serialization: every trial is a numbered results page with explicit rank markers and letter labels whose
mapping to positions is redrawn per trial; item content is assigned to positions at random so that position
is independent of quality. See the prompt templates inside `gate_logit.py` and `gpt_curve.py`.
