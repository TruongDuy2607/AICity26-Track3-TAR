#!/usr/bin/env bash
# Score a model on the held-out split with the official track3/evaluate.py.
# vLLM can't apply LoRA to the Qwen3-VL vision tower: merge first (MODEL_PATH) or use
# BACKEND=transformers with ADAPTER.
# Usage:
#   MODEL_PATH=<merged_dir> bash scripts/eval.sh
#   BACKEND=transformers ADAPTER=output/.../checkpoint-xxxx bash scripts/eval.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/configs/common.sh"

MODEL_PATH="${MODEL_PATH:-}"
if [ -z "$MODEL_PATH" ]; then
    : "${ADAPTER:?Set MODEL_PATH=<merged dir> (vLLM) or ADAPTER=<ckpt> (with BACKEND=transformers)}"
fi
VAL_GT="${VAL_GT:-$DATA_DIR/val_gt.json}"
VAL_PRED="${VAL_PRED:-$HERE/preds/val_pred.jsonl}"

cd "$HERE"
# 1) inference over the held-out GT (val videos live under TRAIN_VIDEOS_ROOT)
MODEL_PATH="$MODEL_PATH" ADAPTER="${ADAPTER:-}" TEST_JSON="$VAL_GT" PRED_OUT="$VAL_PRED" \
    VIDEOS_ROOT="$TRAIN_VIDEOS_ROOT" bash scripts/infer.sh "$@"
# 2) score with the official evaluator
python -m track3.eval_local --gt "$VAL_GT" --pred "$VAL_PRED"
