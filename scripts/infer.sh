#!/usr/bin/env bash
# Run inference over the TAR test set with a fine-tuned adapter.
# Usage: ADAPTER=output/.../checkpoint-xxx bash scripts/infer.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/configs/common.sh"

# Source: ADAPTER or MODEL_PATH. vLLM can't apply LoRA to the Qwen3-VL vision tower:
# merge first (MODEL_PATH) or use BACKEND=transformers.
MODEL_PATH="${MODEL_PATH:-}"
if [ -n "$MODEL_PATH" ]; then
    SRC=(--model "$MODEL_PATH")
else
    : "${ADAPTER:?Set ADAPTER=<checkpoint dir> or MODEL_PATH=<merged model dir>}"
    SRC=(--adapter "$ADAPTER")
fi
PRED_OUT="${PRED_OUT:-$HERE/preds/test_pred.jsonl}"
BACKEND="${BACKEND:-vllm}"      # set BACKEND=transformers if vLLM isn't installed
VOTE_N="${VOTE_N:-5}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"  # only used by the transformers backend
# CLOSED_SCORING=vote restores legacy n-sample voting; NO_CONDITIONED_OE=1 disables
# the conditioned mcq_openended pass.
CLOSED_SCORING="${CLOSED_SCORING:-logprob}"
CONDITIONED_ARG=()
[ "${NO_CONDITIONED_OE:-0}" = "1" ] && CONDITIONED_ARG=(--no-conditioned-openended)
# MCQ_PERMUTE=4: average mcq scores over 4 cyclic option rotations (letter-bias debias).
MCQ_PERMUTE="${MCQ_PERMUTE:-0}"

# multi-GPU: vLLM tensor parallel over all visible GPUs, or transformers device_map
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NGPU=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
TENSOR_PARALLEL="${TENSOR_PARALLEL:-$NGPU}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
ENG_ARGS=()
if [ "$BACKEND" = "vllm" ]; then
    ENG_ARGS+=(--tensor-parallel-size "$TENSOR_PARALLEL" --gpu-memory-utilization "$GPU_MEM_UTIL")
    [ -n "${MAX_MODEL_LEN:-}" ] && ENG_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
    [ -n "${MAX_LORA_RANK:-}" ] && ENG_ARGS+=(--max-lora-rank "$MAX_LORA_RANK")
else
    ENG_ARGS+=(--device-map "${DEVICE_MAP:-auto}")
fi
echo "[infer] gpus=$CUDA_VISIBLE_DEVICES (NGPU=$NGPU) backend=$BACKEND TP=$TENSOR_PARALLEL"

cd "$HERE"
VIDEO_MAX_PIXELS="$VIDEO_MAX_PIXELS" \
FPS_MAX_FRAMES="$FPS_MAX_FRAMES" \
python -m track3.infer \
    "${SRC[@]}" \
    --test-json "$TEST_JSON" \
    --videos-root "$VIDEOS_ROOT" \
    --out "$PRED_OUT" \
    --backend "$BACKEND" \
    --attn-impl "$ATTN_IMPL" \
    --frames-mode "$FRAMES_MODE" \
    --num-frames "$NUM_FRAMES" \
    --max-side "$MAX_SIDE" \
    --temporal-num-frames "$TEMPORAL_NUM_FRAMES" \
    --temporal-max-side "$TEMPORAL_MAX_SIDE" \
    --closed-scoring "$CLOSED_SCORING" \
    "${CONDITIONED_ARG[@]}" \
    --mcq-permute "$MCQ_PERMUTE" \
    --vote-n "$VOTE_N" \
    "${ENG_ARGS[@]}" \
    "$@"

echo "[infer] predictions at $PRED_OUT"
