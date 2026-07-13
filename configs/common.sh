#!/usr/bin/env bash
# Shared paths + data/frame knobs for the Track3 pipeline.
# Override any value by exporting it before calling a script in scripts/.
set -euo pipefail

# Repo path
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
export ANN_DIR="${ANN_DIR:-$TAR_ROOT/train}"
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


# frames-mode: "video" (decode at train time) or "extract" (pre-extracted JPEGs)
export FRAMES_MODE="${FRAMES_MODE:-video}"


export NUM_FRAMES="${NUM_FRAMES:-32}"
export MAX_SIDE="${MAX_SIDE:-448}"

# Qwen3-VL video env vars consumed by ms-swift:
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-200000}"          # ~224*224
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-$NUM_FRAMES}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"


export TEMPORAL_NUM_FRAMES="${TEMPORAL_NUM_FRAMES:-32}"
export TEMPORAL_MAX_SIDE="${TEMPORAL_MAX_SIDE:-$MAX_SIDE}"
export FILTER_TEMPORAL="${FILTER_TEMPORAL:-1}"
export TEMPORAL_COT="${TEMPORAL_COT:-1}"

# data balancing
export MAX_PER_TASK="${MAX_PER_TASK:-3670}"  # equalize tasks for the mean metric
export VAL_RATIO="${VAL_RATIO:-0.02}"

echo "[common] TRACK3_ROOT=$TRACK3_ROOT"
echo "[common] ANN_DIR=$ANN_DIR  TRAIN_VIDEOS_ROOT=$TRAIN_VIDEOS_ROOT"
echo "[common] DATA_DIR=$DATA_DIR  FRAMES_MODE=$FRAMES_MODE NUM_FRAMES=$NUM_FRAMES"
