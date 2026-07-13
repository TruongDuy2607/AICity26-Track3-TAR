#!/usr/bin/env bash
# Model + LoRA + optimization hyper-parameters (Phase-1 Qwen3-VL-8B-Instruct SFT).
# Sourced by scripts/train-runai.sh after configs/common.sh and a hardware profile.

# base model
export MODEL="${MODEL:-/home/jovyan/minh-workspace/duy/AI-City26-Track3/models/Qwen3-VL-32B-Instruct}"
export TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
# If auto-inference ever fails for a local path, set MODEL_TYPE=qwen3_vl (passed
# through only when non-empty; see scripts/train-runai.sh).
export MODEL_TYPE="${MODEL_TYPE:-}"

# LoRA
export TUNER_TYPE="${TUNER_TYPE:-lora}"
export LORA_RANK="${LORA_RANK:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export TARGET_MODULES="${TARGET_MODULES:-all-linear}"
export FREEZE_VIT="${FREEZE_VIT:-true}"
export FREEZE_ALIGNER="${FREEZE_ALIGNER:-true}"

# optimization
export NUM_EPOCHS="${NUM_EPOCHS:-2}"
export LEARNING_RATE="${LEARNING_RATE:-2e-4}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
export MAX_LENGTH="${MAX_LENGTH:-4096}"
export LR_SCHEDULER="${LR_SCHEDULER:-cosine}"

# experiment tracking (wandb)
# wandb is ON by default. The API key is auto-loaded from configs/wandb.txt
# (gitignored) so no manual `wandb login` is needed; override WANDB_KEY_FILE to
# point elsewhere, or export WANDB_API_KEY directly to take precedence. Set
# REPORT_TO=tensorboard to opt out. WANDB_NAME is set per-run in scripts/train-runai.sh.
export REPORT_TO="${REPORT_TO:-wandb}"                # tensorboard | wandb | "wandb tensorboard"
export WANDB_PROJECT="${WANDB_PROJECT:-aicity26-track3}"
export WANDB_MODE="${WANDB_MODE:-online}"             # online | offline | disabled
_WANDB_KEY_FILE="${WANDB_KEY_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/wandb.txt}"
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$_WANDB_KEY_FILE" ]; then
    WANDB_API_KEY="$(tr -d '[:space:]' < "$_WANDB_KEY_FILE")"
    export WANDB_API_KEY
fi


if [[ "$REPORT_TO" == *wandb* ]] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "[wandb][WARN] no WANDB_API_KEY and no $_WANDB_KEY_FILE — disabling wandb." >&2
    REPORT_TO="$(echo "${REPORT_TO//wandb/}" | xargs)"
    [ -z "$REPORT_TO" ] && REPORT_TO="none"
    export REPORT_TO WANDB_MODE="disabled"
fi
# export WANDB_ENTITY="your-team"

# logging / checkpointing
export EVAL_STEPS="${EVAL_STEPS:-200}"
export SAVE_STEPS="${SAVE_STEPS:-200}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
export LOGGING_STEPS="${LOGGING_STEPS:-10}"
export DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
export DATALOADER_WORKERS="${DATALOADER_WORKERS:-8}"
