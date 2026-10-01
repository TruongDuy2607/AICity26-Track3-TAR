#!/usr/bin/env bash
# Option-anchored mcq_openended override (see track3/mcqoe_anchor.py).
#   MODE=anchor  deterministic "X. <chosen option text>" (no model):
#     PRED=preds/experiment-1/pred-040629-m3.struct.jsonl \
#       OUT=preds/mcqoe_anchor.jsonl MODE=anchor bash scripts/mcqoe_anchor.sh
#   MODE=render  option-anchored one-sentence render (needs a model):
#     MODEL_PATH=/.../merged PRED=preds/.../m3.struct.jsonl \
#       OUT=preds/mcqoe_render.jsonl MODE=render bash scripts/mcqoe_anchor.sh
# Append the output after the dossier override (the last line per item_index wins).
set -euo pipefail
HERE="/home/jovyan/data/Challenges/AI-City/AI-City26-Track3"
[ -d "$HERE" ] || HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$HERE/configs/common.sh"
# Activate the conda env yourself before running (see requirements.txt).
cd "$HERE"

MODE="${MODE:-anchor}"
TEST_JSON="${TEST_JSON:-$HERE/data/test/test.json}"
PRED="${PRED:?Set PRED=<predictions jsonl, post-structural preferred (the mcq letter source)>}"
OUT="${OUT:-$HERE/preds/mcqoe_anchor.jsonl}"
ARGS=(--test-json "$TEST_JSON" --pred "$PRED" --out "$OUT" --mode "$MODE")

if [ "$MODE" = "render" ]; then
    MODEL_PATH="${MODEL_PATH:-}"; ADAPTER="${ADAPTER:-}"
    if [ -n "$MODEL_PATH" ]; then ARGS+=(--model "$MODEL_PATH")
    elif [ -n "$ADAPTER" ]; then ARGS+=(--adapter "$ADAPTER")
    else echo "ERROR: MODE=render needs MODEL_PATH or ADAPTER" >&2; exit 1; fi
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    NGPU=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
    ARGS+=(--videos-root "${GROUND_VIDEOS_ROOT:-$TEST_VIDEOS_ROOT}"
           --backend "${BACKEND:-vllm}" --num-frames "${NUM_FRAMES:-32}"
           --tensor-parallel-size "${TENSOR_PARALLEL:-$NGPU}"
           --gpu-memory-utilization "${GPU_MEM_UTIL:-0.9}"
           --max-model-len "${MAX_MODEL_LEN:-8192}")
    [ "${NO_RENDER_VIDEO:-0}" = "1" ] && ARGS+=(--no-render-video)
    [ -n "${LENGTH_TOL:-}" ] && ARGS+=(--length-tol "$LENGTH_TOL")
fi

echo "[mcqoe_anchor] MODE=$MODE  PRED=$PRED -> $OUT"
python -m track3.mcqoe_anchor "${ARGS[@]}"
echo "[mcqoe_anchor] next: concat after the dossier override, then run M3 (gate on board)."
