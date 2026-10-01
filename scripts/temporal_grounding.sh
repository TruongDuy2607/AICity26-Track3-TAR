#!/usr/bin/env bash
# Frame-grounded temporal_localization override (see track3/temporal_grounding.py).
#   audit on train auto-GT:
#     ADAPTER=output/.../checkpoint-xxxx MERGE=1 LIMIT=300 bash scripts/temporal_grounding.sh audit
#   test override, then splice into a submission:
#     MODEL_PATH=/.../merged bash scripts/temporal_grounding.sh test
#     TEMPORAL_OVERRIDE=preds/temporal_grounding.jsonl bash scripts/postprocess.sh
# Model: MODEL_PATH=<merged dir>, or ADAPTER=<ckpt> (+ MERGE=1 for vLLM, or BACKEND=transformers).
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
[ -n "$HF_TOKEN" ] && { hf auth login --token "$HF_TOKEN" --add-to-git-credential; hf auth whoami || true; }

cd "$HERE"
MODE="${1:-test}"          # test | audit
BACKEND="${BACKEND:-vllm}"
OUT="${OUT:-$HERE/preds/temporal_grounding.jsonl}"
LIMIT="${LIMIT:-0}"

# GPU / parallelism
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NGPU=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
TENSOR_PARALLEL="${TENSOR_PARALLEL:-$NGPU}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"

# model source
MODEL_PATH="${MODEL_PATH:-}"
ADAPTER="${ADAPTER:-}"
if [ -z "$MODEL_PATH" ] && [ -z "$ADAPTER" ]; then
    echo "ERROR: set MODEL_PATH=<merged/full dir> or ADAPTER=<checkpoint dir>" >&2
    exit 1
fi
if [ -n "$ADAPTER" ] && [ "${MERGE:-0}" = "1" ]; then
    MERGED_DIR="${MERGED_DIR:-${ADAPTER%/}-merged}"
    if [ ! -f "$MERGED_DIR/config.json" ]; then
        echo "[grounding] merging LoRA: $ADAPTER -> $MERGED_DIR ..."
        swift export --adapters "$ADAPTER" --merge_lora true --output_dir "$MERGED_DIR"
    fi
    MODEL_PATH="$MERGED_DIR"; ADAPTER=""
fi
if [ -n "$MODEL_PATH" ]; then SRC=(--model "$MODEL_PATH"); else SRC=(--adapter "$ADAPTER"); fi

# engine args
ENG_ARGS=(--backend "$BACKEND" --max-model-len "${MAX_MODEL_LEN:-4096}")
if [ "$BACKEND" = "vllm" ]; then
    ENG_ARGS+=(--tensor-parallel-size "$TENSOR_PARALLEL" --gpu-memory-utilization "$GPU_MEM_UTIL")
    [ -n "${MAX_LORA_RANK:-}" ] && ENG_ARGS+=(--max-lora-rank "$MAX_LORA_RANK")
else
    ENG_ARGS+=(--attn-impl "${ATTN_IMPL:-sdpa}" --device-map "${DEVICE_MAP:-auto}")
fi

# grounding knobs (override any GroundConfig field)
GROUND_ARGS=()
for kv in stride:STRIDE min-frames:MIN_FRAMES max-frames:MAX_FRAMES max-side:GMAX_SIDE \
          smooth-k:SMOOTH_K rel-thresh:REL_THRESH min-width:MIN_WIDTH \
          conf-thresh:CONF_THRESH margin:MARGIN agree-iou:AGREE_IOU \
          relocate-width:RELOCATE_WIDTH; do
    flag="${kv%%:*}"; var="${kv##*:}"
    [ -n "${!var:-}" ] && GROUND_ARGS+=("--${flag}" "${!var}")
done

# GROUND_VIDEOS_ROOT defaults per mode (common.sh already pre-sets VIDEOS_ROOT).
if [ "$MODE" = "audit" ]; then
    # audit on train auto-GT (train videos; scores automatically)
    AUDIT_VIDEOS="${GROUND_VIDEOS_ROOT:-$TRAIN_VIDEOS_ROOT}"
    echo "[grounding] AUDIT on train (limit=$LIMIT, videos-root=$AUDIT_VIDEOS)"
    python -m track3.temporal_grounding \
        "${SRC[@]}" --train-dir "$ANN_DIR" \
        --videos-root "$AUDIT_VIDEOS" \
        --out "$OUT" --frames-root "${FRAMES_ROOT:-$HERE/data/frames_grounding}" \
        --limit "$LIMIT" --score "${ENG_ARGS[@]}" "${GROUND_ARGS[@]}"
else
    # test override (or score a val_gt.json via TEST_JSON + SCORE=1)
    TEST_JSON="${TEST_JSON:-$HERE/data/test/test.json}"
    TEST_VIDEOS="${GROUND_VIDEOS_ROOT:-$TEST_VIDEOS_ROOT}"
    SCORE_ARG=(); [ "${SCORE:-0}" = "1" ] && SCORE_ARG=(--score)
    echo "[grounding] TEST override -> $OUT  (TEST_JSON=$TEST_JSON, videos-root=$TEST_VIDEOS)"
    python -m track3.temporal_grounding \
        "${SRC[@]}" --test-json "$TEST_JSON" \
        --videos-root "$TEST_VIDEOS" \
        --out "$OUT" --frames-root "${FRAMES_ROOT:-$HERE/data/frames_grounding}" \
        --limit "$LIMIT" "${SCORE_ARG[@]}" "${ENG_ARGS[@]}" "${GROUND_ARGS[@]}"
    echo ""
    echo "[grounding] next: splice into a submission ->"
    echo "    TEMPORAL_OVERRIDE=$OUT bash scripts/postprocess.sh"
fi
