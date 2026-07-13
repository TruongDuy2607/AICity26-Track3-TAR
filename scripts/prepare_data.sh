#!/usr/bin/env bash
# Build ms-swift train/val jsonl from the TAR annotation files.
# Usage: [FRAMES_MODE=extract] bash scripts/prepare_data.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/configs/common.sh"

cd "$HERE"
# SKIP_BASE=1: reuse an existing data/processed/{train,val}.jsonl and only run the
# optional render-SFT block below (keeps the base split IDENTICAL across phases —
# a rebuild would reshuffle nothing but wastes minutes on 44k items). Falls back to
# a full build when the base set is absent.
if [ "${SKIP_BASE:-0}" = "1" ] && [ -f "$DATA_DIR/train.jsonl" ]; then
    echo "[prepare] SKIP_BASE=1 — reusing base SFT set at $DATA_DIR/train.jsonl"
else
    # FILTER_TEMPORAL=1 drops degenerate auto-labelled temporal intervals;
    # TEMPORAL_COT=1 trains an inline timestamp chain-of-thought before the JSON.
    FILTER_ARG=()
    [ "${FILTER_TEMPORAL:-0}" = "1" ] && FILTER_ARG=(--filter-temporal)
    [ "${TEMPORAL_COT:-0}" = "1" ] && FILTER_ARG+=(--temporal-cot)
    python -m track3.build_dataset \
        --ann-dir "$ANN_DIR" \
        --videos-root "$TRAIN_VIDEOS_ROOT" \
        --out-dir "$DATA_DIR" \
        --frames-mode "$FRAMES_MODE" \
        --frames-root "$DATA_DIR/frames" \
        --num-frames "$NUM_FRAMES" \
        --max-side "$MAX_SIDE" \
        --temporal-num-frames "$TEMPORAL_NUM_FRAMES" \
        --temporal-max-side "$TEMPORAL_MAX_SIDE" \
        "${FILTER_ARG[@]}" \
        --max-per-task "$MAX_PER_TASK" \
        --val-ratio "$VAL_RATIO" \
        --skip-missing \
        "$@"

    # Validate the processed jsonl before it is ever fed to the trainer.
    echo "[prepare] validating processed dataset ..."
    python -m track3.check_dataset --data-dir "$DATA_DIR"
fi

# --- OPTIONAL: Fact-Sheet->Answer SFT distillation set (strategy C: GT facts + noise) --
# RENDER_SFT=1 also builds data/processed/render_sft.jsonl and mixes it with the base
# train.jsonl (render:base = 1:REPLAY_RATIO) into data/processed/train_render.jsonl,
# which train-runai.sh picks up automatically when RENDER_SFT=1. Default off -> unchanged.
if [ "${RENDER_SFT:-0}" = "1" ]; then
    echo "[prepare] building Fact-Sheet->Answer render-SFT set (strategy C) ..."
    # shellcheck disable=SC2086  # RENDER_TASKS is an intentional word-split list
    python -m track3.build_render_sft \
        --train-dir "$ANN_DIR" --videos-root "$TRAIN_VIDEOS_ROOT" \
        --out "$DATA_DIR/render_sft.jsonl" \
        --tasks ${RENDER_TASKS:-temporal_description causal_linkage open_qa video_summarization scene_description bcq_openended} \
        --per-task-cap "${RENDER_CAP:-2000}" --seed "${RENDER_SEED:-0}" \
        --fact-policy "${FACT_POLICY:-video-priority}" \
        --p-consequence "${P_CONS:-0.3}" --p-scene "${P_SCENE:-0.25}" \
        --p-obs-drop "${P_OBS:-0.3}" \
        --p-obs-flip "${P_FLIP:-0.0}" --max-obs "${MAX_OBS:-2}" \
        --replay-jsonl "$DATA_DIR/train.jsonl" --replay-ratio "${REPLAY_RATIO:-2}" \
        --mixed-out "$DATA_DIR/train_render.jsonl"
    echo "[prepare] render-SFT mixed set -> $DATA_DIR/train_render.jsonl"
fi
