#!/bin/bash
# Round-24: real-item ESCI re-measurement of the six local main-set models. Waits for GPU1 to be free (guest jobs are never touched).
cd "$(dirname "$0")"
STAR=${STAR:-python}
SAM3=${SAM3:-python}
export PYTORCH_ALLOC_CONF=expandable_segments:True
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | tr -d ' ')
  if [ -n "$used" ] && [ "$used" -lt 3000 ]; then break; fi
  echo "$(date '+%H:%M:%S') waiting for GPU1 (used=${used}MiB)"; sleep 60
done
echo "=== $(date '+%H:%M:%S') GPU free, starting chain"
run(){ echo "=== $(date '+%H:%M:%S') START $*"; "$@" 2>&1 | grep -v "Warning\|warn\|pynvml" ; echo "=== $(date '+%H:%M:%S') END rc=${PIPESTATUS[0]}"; }
run $STAR gate_logit_real.py microsoft/Phi-4-mini-instruct
run $STAR gate_logit_real.py Qwen/Qwen3-VL-8B-Instruct
run $STAR gate_logit_real.py meta-llama/Llama-3.1-8B-Instruct
run $STAR gate_logit_real.py unsloth/Qwen3-VL-32B-Instruct-bnb-4bit --bs=4
run $STAR gate_logit_real.py mistralai/Mistral-Small-3.2-24B-Instruct-2506 --4bit
run $SAM3 gate_logit_real.py google/gemma-4-12B-it --4bit
echo "=== CHAIN DONE $(date)"
