#!/usr/bin/env bash
# Shared paths + data/frame knobs for the Track3 pipeline.
# Override any value by exporting it before calling a script in scripts/.
set -euo pipefail

# --- repo paths ---------------------------------------------------------------
# Resolved relative to THIS file, so any checkout location works (dev box, Run:AI
# pods, per-user clones). HERE is the canonical repo-root alias every script uses;
# scripts locate common.sh relative to themselves, source it, then read HERE.
TRACK3_ROOT="${TRACK3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TRACK3_ROOT
export HERE="$TRACK3_ROOT"

# --- Hugging Face token (secret; same pattern as configs/wandb.txt) -------------
# Auto-loaded from configs/hf.txt (gitignored) when HF_TOKEN isn't already
# exported; override HF_TOKEN_FILE to point elsewhere. Empty token = scripts skip
# `hf auth login` (public models still work from the local cache).
_HF_TOKEN_FILE="${HF_TOKEN_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hf.txt}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$_HF_TOKEN_FILE" ]; then
    HF_TOKEN="$(tr -d '[:space:]' < "$_HF_TOKEN_FILE")"
fi
export HF_TOKEN="${HF_TOKEN:-}"

# --- raw TAR dataset layout ---------------------------------------------------
# Expected on-disk layout (as downloaded):
#   $TAR_ROOT/train/<task>.json        (bcq.json, mcq.json, ... 10 files)
#   $TAR_ROOT/train/videos/<video_id>  (video_id resolves under here)
#   $TAR_ROOT/test/test.json
#   $TAR_ROOT/test/videos/<video_id>
export TAR_ROOT="${TAR_ROOT:-/home/jovyan/minh-workspace/duy/TAR-AICity26/data}"
export ANN_DIR="${ANN_DIR:-$TAR_ROOT/train}"                  # the <task>.json files
export TRAIN_VIDEOS_ROOT="${TRAIN_VIDEOS_ROOT:-$TAR_ROOT/train/videos}"
export TEST_JSON="${TEST_JSON:-$TAR_ROOT/test/test.json}"
export TEST_VIDEOS_ROOT="${TEST_VIDEOS_ROOT:-$TAR_ROOT/test/videos}"

# Default video root used by inference (the real test set). Local held-out eval
# (scripts/eval.sh) overrides this to TRAIN_VIDEOS_ROOT since val is held out of train.
export VIDEOS_ROOT="${VIDEOS_ROOT:-$TEST_VIDEOS_ROOT}"

# --- pipeline outputs (kept separate from the raw dataset) --------------------
export DATA_DIR="${DATA_DIR:-$TRACK3_ROOT/data/processed}"    # built train/val jsonl + val_gt.json
export OUTPUT_DIR="${OUTPUT_DIR:-$TRACK3_ROOT/output}"        # checkpoints
export SUBMIT_DIR="${SUBMIT_DIR:-$TRACK3_ROOT/submissions}"

# --- video / frame budget (the main accuracy<->memory knob) -------------------
# frames-mode: "video" (decode at train time) or "extract" (pre-extracted JPEGs)
export FRAMES_MODE="${FRAMES_MODE:-video}"
# 8 frames over a ~1-min clip is ~7.5s/frame — too coarse to localize 3s events
# (temporal IoU) or describe motion. Bumped 8->16 globally (temporal rides a denser
# budget below). Drop back toward 8 if you hit OOM on 2xA100-40G.
export NUM_FRAMES="${NUM_FRAMES:-32}"
export MAX_SIDE="${MAX_SIDE:-448}"
# Qwen3-VL video env vars consumed by ms-swift:
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-200000}"          # ~224*224
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-$NUM_FRAMES}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"

# --- vLLM sampling backend (infer) ---------------------------------------------
# Disable FlashInfer's top-k/top-p sampler: on this box its JIT (nvcc+ninja build
# of sampling.cu) fails and crashes EngineCore at startup. vLLM falls back to the
# native PyTorch sampler (no JIT, negligible perf difference for our batch sizes).
# Set to 1 only if FlashInfer's toolchain is fixed. Inherited by EngineCore procs.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# --- temporal-localization framing (the temporal IoU lever) -------------------
# temporal_localization now uses NATIVE video (not extracted JPEG lists): an image
# list is stamped with a default fps that corrupts Qwen3-VL's time-aligned position
# IDs and makes the timestamp hint a no-op (measured: mIoU stuck at 0.186). Native
# video keeps real fps/duration; the IoU lever is the global frame budget below plus
# a duration hint in the prompt. TEMPORAL_NUM_FRAMES raises FPS_MAX_FRAMES for the
# temporal pass. TEMPORAL_COT=1 trains an inline timestamp CoT before the JSON.
# FILTER_TEMPORAL=1 drops degenerate auto-labelled intervals that cap mIoU.
export TEMPORAL_NUM_FRAMES="${TEMPORAL_NUM_FRAMES:-32}"
export TEMPORAL_MAX_SIDE="${TEMPORAL_MAX_SIDE:-$MAX_SIDE}"
export FILTER_TEMPORAL="${FILTER_TEMPORAL:-1}"
export TEMPORAL_COT="${TEMPORAL_COT:-1}"

# --- data balancing / split ---------------------------------------------------
export MAX_PER_TASK="${MAX_PER_TASK:-3670}"  # equalize tasks for the mean metric
export VAL_RATIO="${VAL_RATIO:-0.02}"

echo "[common] TRACK3_ROOT=$TRACK3_ROOT"
echo "[common] ANN_DIR=$ANN_DIR  TRAIN_VIDEOS_ROOT=$TRAIN_VIDEOS_ROOT"
echo "[common] DATA_DIR=$DATA_DIR  FRAMES_MODE=$FRAMES_MODE NUM_FRAMES=$NUM_FRAMES"
