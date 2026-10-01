#!/usr/bin/env bash
# Collision-Dossier text override for the BERTScore paragraph tasks (see track3/text_dossier.py).
#   val gate first:
#     MODEL_PATH=/.../merged TEST_JSON=data/processed/val_gt.json \
#         PRED=preds/val_pred.struct.jsonl OUT=preds/val_dossier.jsonl \
#         GROUND_VIDEOS_ROOT=$TRAIN_VIDEOS_ROOT bash scripts/text_dossier.sh
#     TEXT_OVERRIDE=preds/val_dossier.jsonl PRED_OUT=preds/val_pred.jsonl \
#         GT_JSON=data/processed/val_gt_curated.json bash scripts/postprocess.sh
#   test override, then splice into a submission:
#     MODEL_PATH=/.../merged bash scripts/text_dossier.sh
#     TEXT_OVERRIDE=preds/text_dossier.jsonl bash scripts/postprocess.sh
# Model: MODEL_PATH=<merged dir>, or ADAPTER=<ckpt> (+ MERGE=1 for vLLM, or BACKEND=transformers).
# PRED (post-structural predictions) feeds the fact sheet; optional but recommended.
set -euo pipefail
# common.sh resolves + exports HERE (repo root) from its own location.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/configs/common.sh"
MODEL_CONFIG="${MODEL_CONFIG:-qwen3vl_32b_lora}"
source "$HERE/configs/${MODEL_CONFIG}.sh"
PROFILE="${PROFILE:-a100_80g_4x_32b}"
source "$HERE/configs/profiles/${PROFILE}.sh"
# HF_TOKEN comes from configs/hf.txt via common.sh.
HF_TOKEN="${HF_TOKEN:-}"
# Activate the conda env yourself before running (see requirements.txt).

cd "$HERE"
BACKEND="${BACKEND:-vllm}"
OUT="${OUT:-$HERE/preds/text_dossier.jsonl}"
LIMIT="${LIMIT:-0}"
TEST_JSON="${TEST_JSON:-$HERE/data/test/test.json}"
# Test video_ids are tar_test/... -> TEST_VIDEOS_ROOT; a val_gt.json needs train videos.
VIDEOS="${GROUND_VIDEOS_ROOT:-$TEST_VIDEOS_ROOT}"
# fact-sheet predictions, passed only if present
PRED="${PRED:-$HERE/preds/test_pred.struct.jsonl}"

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
        echo "[dossier] merging LoRA: $ADAPTER -> $MERGED_DIR ..."
        swift export --adapters "$ADAPTER" --merge_lora true --output_dir "$MERGED_DIR"
    fi
    MODEL_PATH="$MERGED_DIR"; ADAPTER=""
fi
if [ -n "$MODEL_PATH" ]; then SRC=(--model "$MODEL_PATH"); else SRC=(--adapter "$ADAPTER"); fi

# engine args
ENG_ARGS=(--backend "$BACKEND" --max-model-len "${MAX_MODEL_LEN:-8192}"
          --num-frames "${NUM_FRAMES:-16}")
if [ "$BACKEND" = "vllm" ]; then
    ENG_ARGS+=(--tensor-parallel-size "$TENSOR_PARALLEL" --gpu-memory-utilization "$GPU_MEM_UTIL")
    [ -n "${MAX_LORA_RANK:-}" ] && ENG_ARGS+=(--max-lora-rank "$MAX_LORA_RANK")
else
    ENG_ARGS+=(--attn-impl "${ATTN_IMPL:-sdpa}" --device-map "${DEVICE_MAP:-auto}")
fi

# dossier knobs (override any DossierConfig field)
DOSS_ARGS=()
[ "${NO_RENDER_VIDEO:-0}" = "1" ] && DOSS_ARGS+=(--no-render-video)
[ -n "${LENGTH_TOL:-}" ] && DOSS_ARGS+=(--length-tol "$LENGTH_TOL")
[ -n "${DOSSIER_MAX_TOKENS:-}" ] && DOSS_ARGS+=(--dossier-max-tokens "$DOSSIER_MAX_TOKENS")
[ -n "${RENDER_MAX_TOKENS:-}" ] && DOSS_ARGS+=(--render-max-tokens "$RENDER_MAX_TOKENS")
[ -n "${TEMPERATURE:-}" ] && DOSS_ARGS+=(--temperature "$TEMPERATURE")
[ -n "${TASKS:-}" ] && DOSS_ARGS+=(--tasks $TASKS)
# DIRECT=1: render straight from the evidence sheet
[ "${DIRECT:-0}" = "1" ] && DOSS_ARGS+=(--direct)
[ -n "${FACT_POLICY:-}" ] && DOSS_ARGS+=(--fact-policy "$FACT_POLICY")
# scene probes for the evidence sheet; K-candidate MBR pool
[ -n "${SCENE_PROBES:-}" ] && DOSS_ARGS+=(--scene-probes "$SCENE_PROBES")
[ -n "${N_SAMPLES:-}" ] && DOSS_ARGS+=(--n-samples "$N_SAMPLES")
[ -n "${SAMPLE_TEMPERATURE:-}" ] && DOSS_ARGS+=(--sample-temperature "$SAMPLE_TEMPERATURE")
[ -n "${CANDIDATES_OUT:-}" ] && DOSS_ARGS+=(--candidates-out "$CANDIDATES_OUT")
[ -n "${SEED:-}" ] && DOSS_ARGS+=(--seed "$SEED")
# GEN_BATCH: requests per engine batch (bounds host RAM)
[ -n "${GEN_BATCH:-}" ] && DOSS_ARGS+=(--gen-batch "$GEN_BATCH")
PRED_ARG=(); [ -f "$PRED" ] && PRED_ARG=(--pred "$PRED")

echo "[dossier] TEXT override -> $OUT  (TEST_JSON=$TEST_JSON, videos-root=$VIDEOS, pred=${PRED_ARG[*]:-none})"
# shellcheck disable=SC2086  # TASKS is an intentional word-split list
python -m track3.text_dossier \
    "${SRC[@]}" --test-json "$TEST_JSON" \
    --videos-root "$VIDEOS" \
    --out "$OUT" --limit "$LIMIT" \
    "${PRED_ARG[@]}" "${ENG_ARGS[@]}" "${DOSS_ARGS[@]}"

echo ""
echo "[dossier] next: gate on curated-val, then splice into a submission ->"
echo "    TEXT_OVERRIDE=$OUT bash scripts/postprocess.sh"
