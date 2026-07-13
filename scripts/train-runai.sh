#!/usr/bin/env bash
# Fine-tune a Qwen3-VL model (LoRA SFT) on the TAR multi-task dataset (Run:AI).
# Logs into HF, auto-prepares data, then trains. Activate the conda env yourself
# (see requirements.txt) BEFORE running this script.
# The model recipe is chosen with MODEL_CONFIG (configs/<MODEL_CONFIG>.sh).
#
# Usage:
#   bash scripts/train-runai.sh                                  # default 32B config + 4xA100-80G
#   PROFILE=a100_80g_4x_32b NUM_EPOCHS=3 bash scripts/train-runai.sh  # override any knob
#   FORCE_PREPARE=1 bash scripts/train-runai.sh                 # rebuild data first
set -euo pipefail
# common.sh resolves + exports HERE (repo root) from its own location.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/configs/common.sh"
# Model recipe (defaults to the Qwen3-VL-8B-Instruct baseline).
MODEL_CONFIG="${MODEL_CONFIG:-qwen3vl_32b_lora}"
source "$HERE/configs/${MODEL_CONFIG}.sh"
PROFILE="${PROFILE:-a100_80g_4x_32b}"
source "$HERE/configs/profiles/${PROFILE}.sh"
# HF_TOKEN comes from configs/hf.txt via common.sh (gitignored, like wandb.txt).
HF_TOKEN="${HF_TOKEN:-}"
echo "[train] python: $(which python)"

if [ -n "$HF_TOKEN" ]; then
    # log "Logging into Hugging Face"
    hf auth login --token "$HF_TOKEN" --add-to-git-credential
    hf auth whoami || true
else
    echo "[WARN] HF_TOKEN is empty; skipping Hugging Face login."
fi


# Build the ms-swift jsonl (data/processed/{train,val}.jsonl + val_gt.json) if it
# isn't there yet — fresh Run AI nodes start without it. Idempotent: skipped when
# train.jsonl already exists. Force a rebuild with FORCE_PREPARE=1.
if [ "${FORCE_PREPARE:-0}" = "1" ] || [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "[train] processed data not found at $DATA_DIR — running prepare_data.sh ..."
    bash "$HERE/scripts/prepare_data.sh"
else
    echo "[train] using existing processed data at $DATA_DIR"
fi

# RENDER_SFT=1: train on the Fact-Sheet->Answer distillation mixed set (build if absent;
# prepare_data.sh with RENDER_SFT=1 emits data/processed/train_render.jsonl).
if [ "${RENDER_SFT:-0}" = "1" ] && [ ! -f "$DATA_DIR/train_render.jsonl" ]; then
    echo "[train] RENDER_SFT=1: building the render-SFT mixed set ..."
    RENDER_SFT=1 bash "$HERE/scripts/prepare_data.sh"
fi

# Training set: defaults to the prepared SFT jsonl (train_render.jsonl when RENDER_SFT=1).
_DEFAULT_JSONL="$DATA_DIR/train.jsonl"
[ "${RENDER_SFT:-0}" = "1" ] && _DEFAULT_JSONL="$DATA_DIR/train_render.jsonl"
TRAIN_JSONL="${TRAIN_JSONL:-$_DEFAULT_JSONL}"
RUN_DIR="${RUN_DIR:-$OUTPUT_DIR/${MODEL_CONFIG}_${PROFILE}_$(date +%Y%m%d_%H%M%S)}"
# Name the wandb run after the output dir (HF wandb integration reads WANDB_NAME).
export WANDB_NAME="${WANDB_NAME:-$(basename "$RUN_DIR")}"
echo "[train] profile=$PROFILE gpus=$CUDA_VISIBLE_DEVICES out=$RUN_DIR"
echo "[train] report_to='$REPORT_TO'$([[ \"$REPORT_TO\" == *wandb* ]] && echo \" project=$WANDB_PROJECT mode=$WANDB_MODE run=$WANDB_NAME\")"


cd "$HERE"
# pass --model_type only when explicitly set (else ms-swift auto-infers it)
MODEL_TYPE_ARG=()
[ -n "${MODEL_TYPE:-}" ] && MODEL_TYPE_ARG=(--model_type "$MODEL_TYPE")
# pass --deepspeed only when set (DeepSpeed requires nvcc; not needed for LoRA)
DEEPSPEED_ARG=()
[ -n "${DEEPSPEED:-}" ] && DEEPSPEED_ARG=(--deepspeed "$DEEPSPEED")

# shellcheck disable=SC2086  # REPORT_TO is an intentional word-split list
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE="$NPROC_PER_NODE" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
VIDEO_MAX_PIXELS="$VIDEO_MAX_PIXELS" \
FPS_MAX_FRAMES="$FPS_MAX_FRAMES" \
IMAGE_MAX_TOKEN_NUM="$IMAGE_MAX_TOKEN_NUM" \
swift sft \
    --model "$MODEL" \
    "${MODEL_TYPE_ARG[@]}" \
    --dataset "$TRAIN_JSONL" \
    --val_dataset "$DATA_DIR/val.jsonl" \
    --split_dataset_ratio 0 \
    --tuner_type "$TUNER_TYPE" \
    --torch_dtype "$TORCH_DTYPE" \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size 1 \
    --learning_rate "$LEARNING_RATE" \
    --lr_scheduler_type "$LR_SCHEDULER" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --target_modules "$TARGET_MODULES" \
    --freeze_vit "$FREEZE_VIT" \
    --freeze_aligner "$FREEZE_ALIGNER" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --attn_impl "$ATTN_IMPL" \
    "${DEEPSPEED_ARG[@]}" \
    --eval_steps "$EVAL_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --logging_steps "$LOGGING_STEPS" \
    --report_to $REPORT_TO \
    --max_length "$MAX_LENGTH" \
    --warmup_ratio "$WARMUP_RATIO" \
    --dataset_num_proc "$DATASET_NUM_PROC" \
    --dataloader_num_workers "$DATALOADER_WORKERS" \
    --output_dir "$RUN_DIR" \
    "$@"

echo "[train] done. checkpoints under $RUN_DIR"
