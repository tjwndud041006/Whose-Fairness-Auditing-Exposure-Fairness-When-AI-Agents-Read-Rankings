#!/bin/bash
# Round-20: seed replication of the direct K=10 measurement (trial-randomization seeds 1,2; seed 42 = original).
cd "$(dirname "$0")"
STAR=${STAR:-python}
SAM3=${SAM3:-python}
export PYTORCH_ALLOC_CONF=expandable_segments:True
run(){ echo "=== $(date '+%H:%M:%S') START $*"; "$@" 2>&1 | grep -v "Warning\|warn\|pynvml" ; echo "=== $(date '+%H:%M:%S') END rc=${PIPESTATUS[0]}"; }
for S in 1 2; do
run $STAR gate_logit_k10.py microsoft/Phi-4-mini-instruct --seed=$S --prefix=k10s$S
run $STAR gate_logit_k10.py Qwen/Qwen3-VL-8B-Instruct --seed=$S --prefix=k10s$S
run $STAR gate_logit_k10.py meta-llama/Llama-3.1-8B-Instruct --seed=$S --prefix=k10s$S
run $STAR gate_logit_k10.py unsloth/Qwen3-VL-32B-Instruct-bnb-4bit --bs=4 --seed=$S --prefix=k10s$S
run $STAR gate_logit_k10.py mistralai/Mistral-Small-3.2-24B-Instruct-2506 --4bit --seed=$S --prefix=k10s$S
run $SAM3 gate_logit_k10.py google/gemma-4-12B-it --4bit --seed=$S --prefix=k10s$S
done
echo "=== CHAIN DONE $(date)"
