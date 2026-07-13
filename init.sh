#!/usr/bin/env bash

# REQUIRED
# 1. Dataset root (expected layout in README §1b).
export TAR_ROOT="/abs/path/to/AI-City26-TAR/data"

# 2. Base model: a local Qwen3-VL-32B-Instruct dir, or the HF id.
export MODEL="Qwen/Qwen3-VL-32B-Instruct"

# 3. GPUs to use (the default profile a100_80g_4x_32b assumes 4x A100-80G).
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# SECRETS (optional)
# Hugging Face token — only if MODEL or the dataset needs auth. Empty = skip login.
export HF_TOKEN=""

# Weights & Biases. To DISABLE wandb: leave WANDB_API_KEY empty AND uncomment the
# REPORT_TO line below.
export WANDB_API_KEY=""
# export REPORT_TO="tensorboard"

# recipe/hardware
export MODEL_CONFIG="${MODEL_CONFIG:-qwen3vl_32b_lora}"
export PROFILE="${PROFILE:-a100_80g_4x_32b}"

echo "[init] TAR_ROOT=$TAR_ROOT  MODEL=$MODEL  GPUS=$CUDA_VISIBLE_DEVICES  PROFILE=$PROFILE"
echo "[init] python: $(which python)  (activate the conda env yourself if this isn't the vlm env)"
