#!/usr/bin/env bash
# Shared setup for the §5.2 controlled ablations (paper "Why it works").
#
# DESIGN INVARIANT — consistency with the test setup:
#   Every ablation runs the SAME chain stages as scripts/prove/phase3_infer.sh,
#   differing only in (a) the split — the held-out val_gt.json scored by the SAME
#   official grader, videos under TRAIN_VIDEOS_ROOT — and (b) ONE toggled variable
#   (the sheet transform / the selection rule). Nothing else changes, so a delta is
#   attributable to that one variable and not to a format/prompt difference.
#
# Source this from an eN_*.sh script; it exports MODEL_PATH, VAL_GT, VAL_VIDEOS,
# the engine args, and the helper functions used below.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$HERE/configs/common.sh"
MODEL_CONFIG="${MODEL_CONFIG:-qwen3vl_32b_lora}"
source "$HERE/configs/${MODEL_CONFIG}.sh"
PROFILE="${PROFILE:-a100_80g_4x_32b}"
source "$HERE/configs/profiles/${PROFILE}.sh"

# --- model: the Phase-0 base merged checkpoint (same one the chain runs on) ------
if [ -z "${MODEL_PATH:-}" ] && [ -f "$HERE/output/prove/BASE_MODEL_PATH" ]; then
    MODEL_PATH="$(cat "$HERE/output/prove/BASE_MODEL_PATH")"
fi
: "${MODEL_PATH:?Set MODEL_PATH=<merged dir> (or run scripts/prove/phase0_base_sft.sh)}"
export MODEL_PATH

# --- the held-out split (val is held out of TRAIN, so videos live under train) ---
export VAL_GT="${VAL_GT:-$DATA_DIR/val_gt.json}"
export VAL_VIDEOS="${VAL_VIDEOS:-$TRAIN_VIDEOS_ROOT}"
if [ ! -f "$VAL_GT" ]; then
    echo "[ablate] $VAL_GT missing — build it first: bash scripts/prepare_data.sh" >&2
    exit 1
fi

# --- output tree (kept out of the real preds/ so it never pollutes a submission) -
export ABL_DIR="${ABL_DIR:-$HERE/output/ablations}"
mkdir -p "$ABL_DIR"

# --- shared base artifacts (produced by 00_base_val.sh) --------------------------
export VAL_PRED="${VAL_PRED:-$ABL_DIR/val_pred.jsonl}"          # base infer (has 'votes')
export VAL_STRUCT0="${VAL_STRUCT0:-$ABL_DIR/val_pred.struct0.jsonl}"  # paired bcq_oe obs
export VAL_SCENE="${VAL_SCENE:-$ABL_DIR/scene_probes.jsonl}"    # P_theta scene probes

# --- engine args (mirror scripts/prove/phase3_infer.sh) --------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NGPU=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
export TENSOR_PARALLEL="${TENSOR_PARALLEL:-$NGPU}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
export PROBE_BATCH="${PROBE_BATCH:-256}"
export GEN_BATCH="${GEN_BATCH:-256}"
ENG_ARGS=(--tensor-parallel-size "$TENSOR_PARALLEL"
          --gpu-memory-utilization "$GPU_MEM_UTIL"
          --max-model-len "${MAX_MODEL_LEN:-8192}" --num-frames "${NUM_FRAMES:-32}")
export ENG_ARGS_STR="${ENG_ARGS[*]}"

# The 6 narrative render/MBR tasks — identical to phase3 (mcq_openended is anchored,
# not rendered; temporal_localization earns no mean points).
export NARR_TASKS="${NARR_TASKS:-temporal_description causal_linkage open_qa video_summarization scene_description bcq_openended}"

echo "[ablate] MODEL_PATH=$MODEL_PATH"
echo "[ablate] VAL_GT=$VAL_GT  VAL_VIDEOS=$VAL_VIDEOS  gpus=$CUDA_VISIBLE_DEVICES TP=$TENSOR_PARALLEL"
echo "[ablate] outputs -> $ABL_DIR"

# ------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------

# Score a predictions jsonl with the OFFICIAL grader -> a metrics json.
# Usage: ablate_eval <pred_jsonl> <metrics_json>
ablate_eval() {
    local pred="$1" out="$2"
    python -m track3.eval_local --gt "$VAL_GT" --pred "$pred" --out "$out" \
        --allow-missing
}

# Overlay a narrative override on the base val_pred (later line wins per item in
# make_submission) and score it. Closed-task predictions stay at the base values,
# so they cancel in any cross-condition delta. Usage: eval_override <override> <out>
eval_override() {
    local override="$1" out="$2"
    local merged="$ABL_DIR/.merged.$$.$(basename "$out").jsonl"
    cat "$VAL_PRED" "$override" > "$merged"
    ablate_eval "$merged" "$out"
    rm -f "$merged"
}

# Run one direct-greedy sheet-render condition on val and score it.
# Usage: run_sheet_condition <label> <SHEET_TRANSFORM> [extra env: PRED=, SCENE_PROBES=, SHEETS_OUT=]
# Emits $ABL_DIR/<label>.override.jsonl and $ABL_DIR/<label>.metrics.json.
run_sheet_condition() {
    local label="$1" transform="${2:-}"
    local override="$ABL_DIR/${label}.override.jsonl"
    local metrics="$ABL_DIR/${label}.metrics.json"
    echo "==================== [ablate] condition '$label' (transform='${transform}') ===================="
    DIRECT=1 TASKS="$NARR_TASKS" \
    MODEL_PATH="$MODEL_PATH" TEST_JSON="$VAL_GT" GROUND_VIDEOS_ROOT="$VAL_VIDEOS" \
    PRED="${PRED:-$VAL_STRUCT0}" SCENE_PROBES="${SCENE_PROBES:-$VAL_SCENE}" \
    OUT="$override" GEN_BATCH="$GEN_BATCH" \
    SHEET_TRANSFORM="$transform" TRANSFORM_SEED="${TRANSFORM_SEED:-0}" \
    SHEETS_OUT="${SHEETS_OUT:-}" \
        bash "$HERE/scripts/text_dossier.sh"
    eval_override "$override" "$metrics"
    echo "[ablate] '$label' -> $metrics"
}
