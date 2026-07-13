#!/usr/bin/env bash
# PROVE Phase 3 — Infer: the full chain (PROVE.md §3).
#
#   1. DO_INFER    infer.sh (W1 logprob + MCQ_PERMUTE debias)  -> preds/test_pred.jsonl
#   2. DO_STRUCT0  structural pre-pass (--all, no text)        -> preds/*.struct0.jsonl
#                  (bcq pairing fixes the observations the evidence sheet quotes)
#   3. DO_SCENE    claim_verify --scene-out                    -> preds/scene_probes.jsonl
#   4. DO_DOSSIER  text_dossier --direct --n-samples K         -> preds/text_dossier.jsonl
#                                                              +  preds/candidates.jsonl
#   5. DO_VERIFY   claim_verify --candidates                   -> preds/verify.jsonl
#   6. DO_MBR      mbr_select --lam LAM                        -> preds/text_override.jsonl
#   7. DO_ANCHOR   mcqoe_anchor (mcq_oe = letter + option)     -> preds/mcqoe_anchor.jsonl
#                  merged over the MBR override               -> preds/text_override.final.jsonl
#   8. DO_POST     postprocess.sh (TEXT_OVERRIDE)              -> submissions/submission-prove.csv
#   9. DO_TD700    re-render temporal_description (budget 0.8, K16) + MBR, overlaid on
#                  the final override + re-post -> submissions/submission-prove-td700.csv (FINAL)
#
# Every stage is toggleable (DO_X=0) and each output is an isolated, partial-safe
# override — a missing stage degrades to the previous behavior. mcq_openended is
# owned by the deterministic anchor (stage 7), NOT rendered/MBR'd: routing it
# through render+MBR measured −0.05 on the board (0.9236 -> 0.8741, then the anchor
# restored it to 0.9300). The narrative render/MBR set is the 6 free-text tasks.
#
# Usage:
#   MODEL_PATH=/path/to/merged bash scripts/prove/phase3_infer.sh
#   (MODEL_PATH also auto-reads output/prove/BASE_MODEL_PATH from phase0)
# Knobs: MCQ_PERMUTE (4) N_SAMPLES (8) SAMPLE_TEMPERATURE (0.8) LAM (0.3)
#        ANCHOR_MODE (anchor) TASKS (the 6-task render set)
#        DO_TD700 (1) TD700_LENGTH_TOL (0.8) TD700_N_SAMPLES (16)
set -euo pipefail
# common.sh resolves + exports HERE (repo root) from its own location.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/configs/common.sh"
MODEL_CONFIG="${MODEL_CONFIG:-qwen3vl_32b_lora}"
source "$HERE/configs/${MODEL_CONFIG}.sh"
# Hardware profile owns CUDA_VISIBLE_DEVICES (override by exporting either).
PROFILE="${PROFILE:-a100_80g_4x_32b}"
source "$HERE/configs/profiles/${PROFILE}.sh"
# Activate the conda env yourself before running (see requirements.txt).

# MODEL_PATH defaults to the Phase-0 base merged checkpoint recorded by
# phase0_base_sft.sh (the shipped pipeline runs the chain on the base checkpoint).
if [ -z "${MODEL_PATH:-}" ] && [ -f "$HERE/output/prove/BASE_MODEL_PATH" ]; then
    MODEL_PATH="$(cat "$HERE/output/prove/BASE_MODEL_PATH")"
    echo "[prove-p3] MODEL_PATH from output/prove/BASE_MODEL_PATH"
fi
: "${MODEL_PATH:?Set MODEL_PATH=<merged model dir> (or run phase0_base_sft.sh first)}"
export MODEL_PATH

# --- GPUs: CUDA_VISIBLE_DEVICES comes from the PROFILE (sourced above); vLLM
#     tensor-parallel spans all of them.
NGPU=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
TENSOR_PARALLEL="${TENSOR_PARALLEL:-$NGPU}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
ENG_ARGS=(--tensor-parallel-size "$TENSOR_PARALLEL"
          --gpu-memory-utilization "$GPU_MEM_UTIL"
          --max-model-len "${MAX_MODEL_LEN:-8192}" --num-frames "$NUM_FRAMES")
echo "[prove-p3] MODEL_PATH=$MODEL_PATH profile=$PROFILE gpus=$CUDA_VISIBLE_DEVICES TP=$TENSOR_PARALLEL"

# --- stage outputs --------------------------------------------------------------
PRED="${PRED:-$HERE/preds/test_pred.jsonl}"
STRUCT0="${STRUCT0:-${PRED%.jsonl}.struct0.jsonl}"
SCENE="${SCENE:-$HERE/preds/scene_probes.jsonl}"
DOSSIER="${DOSSIER:-$HERE/preds/text_dossier.jsonl}"
CANDS="${CANDS:-$HERE/preds/candidates.jsonl}"
VERIFY="${VERIFY:-$HERE/preds/verify.jsonl}"
OVERRIDE="${OVERRIDE:-$HERE/preds/text_override.jsonl}"
UNTRIMMED="${UNTRIMMED:-${OVERRIDE%.jsonl}.untrim.jsonl}"
ANCHOR_OUT="${ANCHOR_OUT:-$HERE/preds/mcqoe_anchor.jsonl}"
FINAL_OVERRIDE="${FINAL_OVERRIDE:-$HERE/preds/text_override.final.jsonl}"

MCQ_PERMUTE="${MCQ_PERMUTE:-4}"
N_SAMPLES="${N_SAMPLES:-8}"
SAMPLE_TEMPERATURE="${SAMPLE_TEMPERATURE:-0.8}"
LAM="${LAM:-0.3}"
ANCHOR_MODE="${ANCHOR_MODE:-anchor}"   # deterministic "X. <option>" (or render)
# Host-RAM guard: video-attached requests decord-decode at encode time, so engine
# batches are chunked (a full-pool batch was OOM-killed at the verify stage). Lower
# these if the pod still gets Killed; raise for more batching throughput.
PROBE_BATCH="${PROBE_BATCH:-256}"
GEN_BATCH="${GEN_BATCH:-256}"
# The 6 free-text render/MBR targets. mcq_openended is NOT here — the deterministic
# anchor (stage 7) owns it (render+MBR measured 0.9236 -> 0.8741 on the board).
TASKS="${TASKS:-temporal_description causal_linkage open_qa video_summarization scene_description bcq_openended}"

# 1) base inference: W1 logprob closed scoring + MCQ permutation debias.
if [ "${DO_INFER:-1}" = "1" ]; then
    echo "==================== [prove-p3] 1/9 infer ===================="
    MCQ_PERMUTE="$MCQ_PERMUTE" PRED_OUT="$PRED" bash "$HERE/scripts/infer.sh"
fi

# 2) structural pre-pass — the evidence sheet quotes PAIRED bcq_oe observations,
#    the anchor reads its letters from here.
if [ "${DO_STRUCT0:-1}" = "1" ]; then
    echo "==================== [prove-p3] 2/9 structural pre-pass ===================="
    python -m track3.structural --test-json "$TEST_JSON" \
        --pred "$PRED" --out "$STRUCT0" --all
fi

# 3) scene-attribute probes (evidence sheet 'Scene:' line; SD's content source).
if [ "${DO_SCENE:-1}" = "1" ]; then
    echo "==================== [prove-p3] 3/9 scene probes ===================="
    python -m track3.claim_verify --model "$MODEL_PATH" \
        --videos-root "$TEST_VIDEOS_ROOT" --test-json "$TEST_JSON" \
        --scene-out "$SCENE" --scene-min-p "${SCENE_MIN_P:-0.5}" \
        --probe-batch "$PROBE_BATCH" "${ENG_ARGS[@]}"
fi

# 4) evidence-v2 direct renders: greedy override + K-candidate MBR pool.
if [ "${DO_DOSSIER:-1}" = "1" ]; then
    echo "==================== [prove-p3] 4/9 dossier renders ===================="
    DIRECT=1 TASKS="$TASKS" PRED="$STRUCT0" OUT="$DOSSIER" \
    SCENE_PROBES="$([ -f "$SCENE" ] && echo "$SCENE")" \
    N_SAMPLES="$N_SAMPLES" SAMPLE_TEMPERATURE="$SAMPLE_TEMPERATURE" \
    GEN_BATCH="$GEN_BATCH" \
    CANDIDATES_OUT="$CANDS" bash "$HERE/scripts/text_dossier.sh"
fi

# 5) claim-level verification of every candidate (W1 probes).
if [ "${DO_VERIFY:-1}" = "1" ] && [ -f "$CANDS" ]; then
    echo "==================== [prove-p3] 5/9 claim verify ===================="
    python -m track3.claim_verify --model "$MODEL_PATH" \
        --videos-root "$TEST_VIDEOS_ROOT" \
        --candidates "$CANDS" --out "$VERIFY" \
        --probe-batch "$PROBE_BATCH" "${ENG_ARGS[@]}"
fi

# 6) verification-weighted MBR selection -> the narrative text override.
#    USE_VERIFY=1 (default) blends an EXISTING verify.jsonl even when the probe
#    stage was skipped (DO_VERIFY=0) — reuse a previous run's probes for free.
if [ "${DO_MBR:-1}" = "1" ] && [ -f "$CANDS" ]; then
    echo "==================== [prove-p3] 6/9 MBR selection ===================="
    VERIFY_ARG=(); [ "${USE_VERIFY:-1}" = "1" ] && [ -f "$VERIFY" ] && \
        VERIFY_ARG=(--verify "$VERIFY" --lam "$LAM")
    python -m track3.mbr_select --candidates "$CANDS" \
        "${VERIFY_ARG[@]}" --out "$OVERRIDE"
else
    # no MBR pool -> the greedy dossier override is the text override.
    [ -f "$DOSSIER" ] && OVERRIDE="$DOSSIER"
fi

# 6b) TD untrim — restore the truncated tails of the consensus TD picks from the
#     candidate pool + strip "Between X and Y," openers (board: +0.031 TD / −0.06
#     opener). Partial-safe: rows without a longer prefix-match pass through.
if [ "${DO_UNTRIM:-1}" = "1" ] && [ -f "$OVERRIDE" ] && [ -f "$CANDS" ]; then
    echo "==================== [prove-p3] 6b/9 TD untrim ===================="
    python -m track3.td_untrim --override-jsonl "$OVERRIDE" \
        --candidates "$CANDS" --out-jsonl "$UNTRIMMED"
    OVERRIDE="$UNTRIMMED"
fi

# 7) mcq_openended anchor ("X. <chosen option text>", letters from the structural
#    pred) merged OVER the narrative override (disjoint task sets;
#    structural._load_text_override lets later lines win per item).
if [ "${DO_ANCHOR:-1}" = "1" ]; then
    echo "==================== [prove-p3] 7/9 mcq_oe anchor ($ANCHOR_MODE) ===================="
    ANCHOR_ARGS=(--test-json "$TEST_JSON" --pred "$STRUCT0" --out "$ANCHOR_OUT"
                 --mode "$ANCHOR_MODE")
    [ "$ANCHOR_MODE" = "render" ] && ANCHOR_ARGS+=(--model "$MODEL_PATH"
                 --videos-root "$TEST_VIDEOS_ROOT" "${ENG_ARGS[@]}")
    python -m track3.mcqoe_anchor "${ANCHOR_ARGS[@]}"
fi
MERGE_SRCS=()
[ -f "$OVERRIDE" ] && MERGE_SRCS+=("$OVERRIDE")
[ "${DO_ANCHOR:-1}" = "1" ] && [ -f "$ANCHOR_OUT" ] && MERGE_SRCS+=("$ANCHOR_OUT")
if [ ${#MERGE_SRCS[@]} -gt 0 ]; then
    cat "${MERGE_SRCS[@]}" > "$FINAL_OVERRIDE"
else
    FINAL_OVERRIDE=""
fi

# 8) structural rules + submission CSV + official validation.
if [ "${DO_POST:-1}" = "1" ]; then
    echo "==================== [prove-p3] 8/9 postprocess ===================="
    TEXT_ARG=""; [ -n "$FINAL_OVERRIDE" ] && [ -f "$FINAL_OVERRIDE" ] && TEXT_ARG="$FINAL_OVERRIDE"
    TEMPORAL_ARG=""; [ -f "$HERE/preds/temporal_grounding.jsonl" ] && \
        TEMPORAL_ARG="$HERE/preds/temporal_grounding.jsonl"
    TEXT_OVERRIDE="$TEXT_ARG" TEMPORAL_OVERRIDE="$TEMPORAL_ARG" \
    PRED_OUT="$PRED" SUBMIT_CSV="${SUBMIT_CSV:-$SUBMIT_DIR/submission-prove.csv}" \
        bash "$HERE/scripts/postprocess.sh"
fi

# 9) td700 — re-render temporal_description with a larger length budget (LENGTH_TOL
#    0.8 ~ 709 chars) + a bigger pool (N_SAMPLES 16), pure-MBR select, then overlay it
#    OVER the final override (later line wins per item) and re-emit the FINAL
#    submission. Isolated to temporal_description (method §3.6; the last +0.001 lever).
if [ "${DO_TD700:-1}" = "1" ]; then
    echo "==================== [prove-p3] 9/9 td700 TD re-render ===================="
    DOSSIER_TD700="${DOSSIER_TD700:-$HERE/preds/text_dossier.td700.jsonl}"
    CANDS_TD700="${CANDS_TD700:-$HERE/preds/candidates.td700.jsonl}"
    OVERRIDE_TD700="${OVERRIDE_TD700:-$HERE/preds/text_override.td700.jsonl}"
    FINAL_TD700="${FINAL_TD700:-$HERE/preds/text_override.final.td700.jsonl}"
    # (a) re-render TD only, raised budget + bigger pool.
    DIRECT=1 TASKS="temporal_description" \
    LENGTH_TOL="${TD700_LENGTH_TOL:-0.8}" \
    N_SAMPLES="${TD700_N_SAMPLES:-16}" SAMPLE_TEMPERATURE="$SAMPLE_TEMPERATURE" \
    PRED="$STRUCT0" SCENE_PROBES="$([ -f "$SCENE" ] && echo "$SCENE")" \
    OUT="$DOSSIER_TD700" CANDIDATES_OUT="$CANDS_TD700" GEN_BATCH="$GEN_BATCH" \
        bash "$HERE/scripts/text_dossier.sh"
    # (b) pure MBR over the uncapped TD pool.
    [ -f "$CANDS_TD700" ] && python -m track3.mbr_select \
        --candidates "$CANDS_TD700" --out "$OVERRIDE_TD700"
    # (c) overlay td700 TD over the final override + re-emit the FINAL submission.
    if [ "${DO_POST:-1}" = "1" ] && [ -f "$OVERRIDE_TD700" ] \
       && [ -n "$FINAL_OVERRIDE" ] && [ -f "$FINAL_OVERRIDE" ]; then
        cat "$FINAL_OVERRIDE" "$OVERRIDE_TD700" > "$FINAL_TD700"
        TEMPORAL_ARG=""; [ -f "$HERE/preds/temporal_grounding.jsonl" ] && \
            TEMPORAL_ARG="$HERE/preds/temporal_grounding.jsonl"
        TEXT_OVERRIDE="$FINAL_TD700" TEMPORAL_OVERRIDE="$TEMPORAL_ARG" \
        PRED_OUT="$PRED" \
        SUBMIT_CSV="${SUBMIT_CSV_TD700:-$SUBMIT_DIR/submission-prove-td700.csv}" \
            bash "$HERE/scripts/postprocess.sh"
    fi
fi

echo "==================== [prove-p3] DONE ===================="
