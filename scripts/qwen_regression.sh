#!/bin/bash
# The Qwen3.8 path against the main worktree: same prompt, deterministic settings (no CPU share, no batched prefill),
# the logits of every prompt position compared byte for byte (the model is on the HDD: ~4 min a load). Run after changing shared code (src/cuda, src/cpu_*).
#   scripts/qwen_regression.sh [MAIN_BUILD_DIR]
set -eu
B=$(cd "$(dirname "$0")/.." && pwd)
MAIN=${1:-$B/../boundless-main-wt/build}
M=$B/../models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
T=$B/bench/corpus/chat_zh.tokens
O=$(mktemp -d)
for side in main this; do
  bin=$MAIN/bl-run; [ $side = this ] && bin=$B/build/bl-run
  BL_CPU_SHARE=0 BL_PREFILL=0 "$bin" --model "$M" --tokens-file "$T" --save-logits "$O/$side.logits" > "$O/$side.log" 2>&1
done
if cmp -s "$O/main.logits" "$O/this.logits"; then echo "IDENTICAL ($(stat -c %s "$O/this.logits") bytes of logits)"
else echo "DIFFERENT: see $O"; exit 1; fi
rm -rf "$O"
