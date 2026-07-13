#!/usr/bin/env bash
# Score a fine-tuned model on the held-out split using the OFFICIAL track3/evaluate.py
# (via track3.eval_local -> track3.official, which loads the single canonical scorer).
# vLLM can't apply LoRA to the Qwen3-VL vision tower, so an all-linear adapter must
# be merged first (swift export --merge_lora) and passed via MODEL_PATH, or scored
# with the transformers backend.
# Usage:
#   MODEL_PATH=<merged_dir> bash scripts/eval.sh                       # vLLM (fast)
#   BACKEND=transformers ADAPTER=output/.../checkpoint-xxxx bash scripts/eval.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/configs/common.sh"

# Source: a merged/full model (MODEL_PATH) OR a LoRA adapter (ADAPTER); forwarded
# to infer.sh, which handles either.
MODEL_PATH="${MODEL_PATH:-}"
if [ -z "$MODEL_PATH" ]; then
    : "${ADAPTER:?Set MODEL_PATH=<merged dir> (vLLM) or ADAPTER=<ckpt> (with BACKEND=transformers)}"
fi
VAL_GT="${VAL_GT:-$DATA_DIR/val_gt.json}"
VAL_PRED="${VAL_PRED:-$HERE/preds/val_pred.jsonl}"

cd "$HERE"
# 1) inference over the held-out GT. val videos are held out of *train*, so they
#    resolve under TRAIN_VIDEOS_ROOT (not the test video root infer.sh defaults to).
MODEL_PATH="$MODEL_PATH" ADAPTER="${ADAPTER:-}" TEST_JSON="$VAL_GT" PRED_OUT="$VAL_PRED" \
    VIDEOS_ROOT="$TRAIN_VIDEOS_ROOT" bash scripts/infer.sh "$@"
# 2) score via the official evaluator (build submission internally)
python -m track3.eval_local --gt "$VAL_GT" --pred "$VAL_PRED"
