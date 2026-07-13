#!/usr/bin/env bash
# Option-anchored mcq_openended (1) — close the one column a competitor beats us on.
# ISOLATED add-on: this script + track3/mcqoe_anchor.py (+ its test) are the whole
# feature. It only produces a {item_index,task,prediction} override jsonl (mcq_openended
# only) in the text_dossier shape, consumed by the existing
# `track3.structural --text-override` hook. Removal = git rm both files. Degrades to
# the model on any uncovered item, so gate on the BOARD.
#
# ---------------------------------------------------------------------------
# MODE=anchor  (V1, deterministic, NO model) — "X. <chosen option text>":
#     PRED=preds/experiment-1/pred-040629-m3.struct.jsonl \
#       OUT=preds/mcqoe_anchor.jsonl MODE=anchor bash scripts/mcqoe_anchor.sh
#
# MODE=render  (V2, model) — option-anchored one-sentence render (+grounding):
#     MODEL_PATH=/.../merged PRED=preds/.../m3.struct.jsonl \
#       OUT=preds/mcqoe_render.jsonl MODE=render bash scripts/mcqoe_anchor.sh
#
# Then splice into M3 alongside the dossier (mcqoe override wins for mcq_openended,
# since track3.structural._load_text_override keeps the LAST line per item_index):
#     cat preds/experiment-1/dossier-040629-ext.jsonl preds/mcqoe_anchor.jsonl \
#         > preds/experiment-1/dossier-040629-mcqoe.jsonl
#     DOSSIER_TAG=mcqoe M_TAG=mcqoe bash scripts/experiment-1/m3_full.sh   # gate on board
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
