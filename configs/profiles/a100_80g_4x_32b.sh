#!/usr/bin/env bash
# Hardware profile: 4 x A100 SXM 80GB.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE=4
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"          # effective batch 16

_PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEEPSPEED="${DEEPSPEED:-$_PROFILE_DIR/../ds/zero3_vlm.json}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
export ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
