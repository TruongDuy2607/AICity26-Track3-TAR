#!/usr/bin/env bash
# Hardware profile: 1 x A100 40GB.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE=1
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"        # effective batch ~16
export DEEPSPEED="${DEEPSPEED:-zero2}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
export ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
