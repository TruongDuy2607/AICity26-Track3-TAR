#!/usr/bin/env bash
# One Run:AI job: inference (multi-GPU) + structural post-processing -> submission.csv.
# vLLM (default) runs tensor parallel over all visible GPUs but can't apply Qwen3-VL
# ViT-LoRA: pass a merged MODEL_PATH, or ADAPTER + MERGE=1, or use BACKEND=transformers
# (device_map=auto, no merge, slower).
#
# Usage:
#   MODEL_PATH=/.../qwen3vl_32b-merged PROFILE=a100_80g_4x_32b bash scripts/infer_submit.sh
#   ADAPTER=output/qwen3vl_32b_lora_.../checkpoint-XXXX MERGE=1 bash scripts/infer_submit.sh
#   ADAPTER=output/.../checkpoint-XXXX BACKEND=transformers bash scripts/infer_submit.sh
#   MODEL_PATH=... TEMPORAL_OVERRIDE=preds/test_override.jsonl bash scripts/infer_submit.sh
#
# Export NUM_FRAMES to the training value (a frame mismatch hurts the text tasks).
set -euo pipefail
# common.sh resolves + exports HERE (repo root) from its own location.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/configs/common.sh"
MODEL_CONFIG="${MODEL_CONFIG:-qwen3vl_32b_lora}"
source "$HERE/configs/${MODEL_CONFIG}.sh"
PROFILE="${PROFILE:-a100_80g_4x_32b}"
source "$HERE/configs/profiles/${PROFILE}.sh"
# HF_TOKEN comes from configs/hf.txt via common.sh.
# Activate the conda env yourself before running (see requirements.txt).
HF_TOKEN="${HF_TOKEN:-}"
echo "[infer_submit] python: $(which python)"

if [ -n "$HF_TOKEN" ]; then
    hf auth login --token "$HF_TOKEN" --add-to-git-credential
    hf auth whoami || true
else
    echo "[WARN] HF_TOKEN is empty; skipping Hugging Face login."
fi

cd "$HERE"

# GPU / parallelism (the whole node)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NGPU=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
TENSOR_PARALLEL="${TENSOR_PARALLEL:-$NGPU}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
BACKEND="${BACKEND:-vllm}"
echo "[infer_submit] gpus=$CUDA_VISIBLE_DEVICES (NGPU=$NGPU) backend=$BACKEND TP=$TENSOR_PARALLEL"

# model source: MODEL_PATH (merged/full) or ADAPTER (+ optional MERGE)
MODEL_PATH="${MODEL_PATH:-}"
ADAPTER="${ADAPTER:-}"
if [ -z "$MODEL_PATH" ] && [ -z "$ADAPTER" ]; then
    echo "ERROR: set MODEL_PATH=<merged/full model dir> or ADAPTER=<checkpoint dir>" >&2
    exit 1
fi
if [ -n "$ADAPTER" ] && [ "${MERGE:-0}" = "1" ]; then
    MERGED_DIR="${MERGED_DIR:-${ADAPTER%/}-merged}"
    echo "[infer_submit] merging LoRA: $ADAPTER -> $MERGED_DIR (needs ~2x model size on disk) ..."
    if [ ! -f "$MERGED_DIR/config.json" ]; then
        swift export --adapters "$ADAPTER" --merge_lora true --output_dir "$MERGED_DIR"
    else
        echo "[infer_submit] reusing existing merged model at $MERGED_DIR"
    fi
    MODEL_PATH="$MERGED_DIR"
    ADAPTER=""
fi

# outputs
PRED_OUT="${PRED_OUT:-$HERE/preds/test_pred.jsonl}"
SUBMIT_CSV="${SUBMIT_CSV:-$SUBMIT_DIR/submission.csv}"

# engine args forwarded to track3.infer (via infer.sh "$@")
INFER_ARGS=()
if [ "$BACKEND" = "vllm" ]; then
    INFER_ARGS+=(--tensor-parallel-size "$TENSOR_PARALLEL" --gpu-memory-utilization "$GPU_MEM_UTIL")
    [ -n "${MAX_LORA_RANK:-}" ] && INFER_ARGS+=(--max-lora-rank "$MAX_LORA_RANK")
    # test clips are short; a smaller context frees KV-cache memory
    INFER_ARGS+=(--max-model-len "${MAX_MODEL_LEN:-4096}")
else
    INFER_ARGS+=(--device-map "${DEVICE_MAP:-auto}")
fi

# 1) inference (whole node)
echo "[infer_submit] STAGE 1/2  inference -> $PRED_OUT"
MODEL_PATH="$MODEL_PATH" ADAPTER="$ADAPTER" BACKEND="$BACKEND" \
PRED_OUT="$PRED_OUT" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
bash "$HERE/scripts/infer.sh" "${INFER_ARGS[@]}"

# 2) structural post-processing -> submission.csv (+ official validation)
echo "[infer_submit] STAGE 2/2  postprocess -> $SUBMIT_CSV"
[ -n "${TEMPORAL_OVERRIDE:-}" ] && export TEMPORAL_OVERRIDE
PRED_OUT="$PRED_OUT" SUBMIT_CSV="$SUBMIT_CSV" RULES="${RULES:---all}" \
bash "$HERE/scripts/postprocess.sh"

echo "[infer_submit] DONE"
echo "[infer_submit] predictions: $PRED_OUT"
echo "[infer_submit] submission:  $SUBMIT_CSV"
